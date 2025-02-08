import os
import sys
import subprocess
import requests
import json
from PIL import Image
from transformers import pipeline

def generate_report(image_path, pipe):
    image = Image.open(image_path)
    prompt = "You are a radiologist. I'm giving you this image, and I'm sure it has 1 or more tumors. Give me a report; write a technical report on this image."
    result = pipe(image, prompt)
    return result[0]['generated_text']

def main():
    image_path = "tumor_ex.jpeg"  # Replace with your image path
    try:
        # Install vLLM
        subprocess.run([sys.executable, "-m", "pip", "install", "vllm"], check=True)
        
        # Start the vLLM server
        subprocess.Popen(["vllm", "serve", "microsoft/llava-med-v1.5-mistral-7b"])
        
        # Generate report using vLLM
        pipe = pipeline("image-text-to-text", model="microsoft/llava-med-v1.5-mistral-7b", device=0, trust_remote_code=True)
        report = generate_report(image_path, pipe)
        print("vLLM Report:", report)
    except Exception as e:
        print(f"Error generating report with vLLM: {e}")
        print("Updating transformers library to the latest version from source...")
        os.system(f"{sys.executable} -m pip install git+https://github.com/huggingface/transformers.git")
        pipe = pipeline("image-text-to-text", model="microsoft/llava-med-v1.5-mistral-7b", device=0, trust_remote_code=True)
        try:
            report = generate_report(image_path, pipe)
            print("Transformers Report:", report)
        except Exception as e:
            print(f"Error generating report with transformers: {e}")

if __name__ == "__main__":
    main()
