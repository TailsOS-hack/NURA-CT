import torch
import psutil


def check_specs():
    print("Checking system specifications...\n")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get GPU details
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        print(f"GPU: {gpu_name}")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    else:
        print("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

    # Check available system memory
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    print(f"Total System Memory: {total_memory:.2f} GB")


if __name__ == "__main__":
    check_specs()