import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

app = FastAPI()

# 加载Whisper模型和处理器
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 如果有GPU,将模型移到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class AudioRequest(BaseModel):
    url: str

@app.post("/transcribe/")
async def transcribe_audio(request: AudioRequest):
    # 下载音频文件
    try:
        response = requests.get(request.url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

    # 将音频保存为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    try:
        # 使用Whisper进行转录
        input_features = processor(temp_file_path, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # 获取时间戳
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        result = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=448,
        )

        token_timestamps = result.token_timestamps.squeeze().tolist()

        # 处理句级时间戳
        sentences = []
        current_sentence = {"text": "", "start": None, "end": None}
        words = transcription.split()
        
        for i, word in enumerate(words):
            if current_sentence["start"] is None:
                current_sentence["start"] = token_timestamps[i]
            
            current_sentence["text"] += word + " "
            current_sentence["end"] = token_timestamps[i]
            
            if word.strip().endswith((".", "!", "?")):
                current_sentence["text"] = current_sentence["text"].strip()
                sentences.append(current_sentence)
                current_sentence = {"text": "", "start": None, "end": None}
        
        if current_sentence["text"]:
            current_sentence["text"] = current_sentence["text"].strip()
            sentences.append(current_sentence)

        return {"sentences": sentences}
    
    finally:
        import os
        os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)