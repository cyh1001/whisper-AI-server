# download_model.py
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import os
import sys
from tqdm import tqdm

def download_model(model_name, save_directory):
    print(f"Starting to download model: {model_name}")
    
    try:
        # Custom progress bar
        class CustomProgressBar(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def update(self, n=1):
                if self.total is None:
                    return
                super().update(n)

        # Download and save the tokenizer
        print("Downloading tokenizer...")
        tokenizer = WhisperTokenizer.from_pretrained(model_name, progress_bar_class=CustomProgressBar)
        tokenizer.save_pretrained(save_directory)
        print("Tokenizer saved successfully.")
        
        # Download and save the model
        print("Downloading model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name, progress_bar_class=CustomProgressBar)
        model.save_pretrained(save_directory)
        print("Model saved successfully.")
        
        print(f"Model and tokenizer have been saved to: {save_directory}")
    except Exception as e:
        print(f"An error occurred during the download process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    model_name = "openai/whisper-tiny"  # Or another model of your choice
    save_directory = "./openai/whisper-tiny"  # Directory to save the model
    
    try:
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        download_model(model_name, save_directory)
        print("Download process completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

    print("Script execution completed.")