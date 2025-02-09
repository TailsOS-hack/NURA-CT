import os
import torch
import torch.distributed as dist
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
)
import requests
from PIL import Image

# Update the model ID
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",  # Automatically assign GPUs if available
)

# Directly use LlavaProcessor, it initializes the image processor internally
processor = LlavaProcessor.from_pretrained(model_id)  

# Get the image path from the user or URL
image_path_or_url = "tumor_ex.jpeg"  # Or a URL like in your provided code

# Ensure the image file exists. If not, raise a FileNotFoundError
if not os.path.exists(image_path_or_url):
    raise FileNotFoundError(f"Image file not found at: {image_path_or_url}")

try:
    # Attempt to open as a local file
    raw_image = Image.open(image_path_or_url)
except FileNotFoundError:
    try:
        # If file not found, try opening as a URL
        raw_image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    except Exception as e:
        print(f"Error loading image: {e}")
        # Handle the error appropriately, e.g., exit or use a default image
        raise  # Re-raise the exception to stop execution

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a radiologist give me a report on this image I am sure it has a tumor give me a techinal report refering to the image as a CT Scan, for exmaple a good report could look like this: 'The CT scan shows a well-defined, homogenously enhancing mass in the right frontal lobe, measuring 3.5 x 2.5 x 2.0 cm. The mass is associated with significant surrounding edema and midline shift to the left. These findings are consistent with a high-grade glioma.'"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")

# Generate the report
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print(processor.decode(output[0][2:], skip_special_tokens=True))