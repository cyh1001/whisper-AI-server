# main.py
from fastapi import FastAPI, HTTPException
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import requests
import tempfile
import os
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 加载Hugging Face的Whisper模型
model_name = "openai/whisper-tiny"  # 或 "openai/whisper-tiny" 根据需求选择
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 加载预下载的Whisper模型
# model_directory = "./openai/whisper-large-v3"  # 确保这与download_model.py中的保存目录相同
# processor = WhisperProcessor.from_pretrained(model_directory)
# model = WhisperForConditionalGeneration.from_pretrained(model_directory)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://podcasthighlight.com",
                   "http://localhost:3000",
                   ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    model = model.to("cuda")

def split_into_sentences(text):
    import re
    sentences = re.split('(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

@app.post("/transcribe_fastapi/")
async def transcribe_audio(url: str):
    try:
        # 下载音频文件
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # 将音频转换为16kHz采样率
        audio = AudioSegment.from_mp3(temp_file_path)
        audio = audio.set_frame_rate(16000)
        audio.export(temp_file_path, format="wav")

        # 使用Whisper模型转录
        input_features = processor(audio.raw_data, sampling_rate=16000, return_tensors="pt").input_features
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # 处理句级时间戳（注意：这里我们只能提供近似的时间戳）
        sentences = split_into_sentences(transcription[0])
        total_duration = len(audio) / 1000  # 总时长（秒）
        time_per_char = total_duration / len(transcription[0])
        
        result = []
        current_time = 0
        for sentence in sentences:
            start_time = current_time
            end_time = current_time + len(sentence) * time_per_char
            result.append({
                "text": sentence,
                "start": round(start_time, 2),
                "end": round(end_time, 2)
            })
            current_time = end_time

        # 清理临时文件
        os.unlink(temp_file_path)

        return {"sentences": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)