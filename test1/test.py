from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None


# from transformers import pipeline

# # 创建 pipeline
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# # 指定音频文件路径
# audio_file = r"C:\Users\caoca\Music\sample1.wav"  # 替换为您的音频文件路径

# # 执行语音识别
# result = pipe(audio_file)

# # 打印识别结果
# print(result["text"])