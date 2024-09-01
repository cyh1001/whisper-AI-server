import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# 加载模型和处理器
processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

# 设置设备（如果有GPU可用，则使用GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 读取音频文件
audio_path = r"C:\Users\caoca\Music\sample1.wav"
speech, sr = librosa.load(audio_path, sr=16000)

# 处理音频
input_features = processor(speech, sampling_rate=sr, return_tensors="pt").input_features
input_features = input_features.to(device)

# 生成转录
generated_ids = model.generate(input_features)

# 解码转录
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("转录结果:")
print(transcription)