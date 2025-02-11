import os
import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import requests
from PIL import Image
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Load model directly
model_id = "allenai/Molmo-7B-D-0924"
cache_dir = 'G:/Model files'  # Ensure this directory has sufficient space

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir=cache_dir  # Use the external hard drive for caching
)

# Load the model with disk offload
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto',
    offload_folder=os.path.join(cache_dir, 'offload'),  # Use the external hard drive for offloading
    offload_state_dict=True,
    cache_dir=cache_dir  # Ensure all downloads go to the external hard drive
)

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

inputs = processor.process(images=[raw_image], text=prompt)

# Move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# Generate the report
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings=["."])
)

# Print the generated report
print(output)