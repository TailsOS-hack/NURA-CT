import os
import sys
import subprocess
import requests
import json
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor


def generate_report(image_path, model, processor):
    raw_image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a radiologist. I'm giving you this image, and I'm sure it has 1 or more tumors. Give me a report; write a technical report on this image."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)


def main():
    image_path = "tumor_ex.jpeg"  # Replace with your image path
    model_id = "microsoft/llava-med-v1.5-mistral-7b"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    processor = LlavaProcessor.from_pretrained(model_id)
    try:
        report = generate_report(image_path, model, processor)
        print("Report:", report)
    except Exception as e:
        print(f"Error generating report: {e}")


if __name__ == "__main__":
    main()