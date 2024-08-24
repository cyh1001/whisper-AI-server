from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
print("Model and processor downloaded and cached.")