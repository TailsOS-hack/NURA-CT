import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

# Load the trained model
model = load_model("stroke_segmentation_model_3d.h5")

# Common shape for resizing (3D shape for full volumes)
target_shape = (128, 128, 64)  # Resize all volumes to this shape

# Function to load and resize NIfTI files
def load_and_resize_nifti(file_path, target_shape):
    nifti_data = nib.load(file_path)
    data = np.array(nifti_data.get_fdata(), dtype=np.float32)

    # Compute resize factors for each dimension for 3D volumes
    resize_factors = [
        target_shape / data.shape,  # Depth
        target_shape / data.shape,  # Height
        target_shape / data.shape  # Width
    ]

    resized_data = zoom(data, resize_factors, order=1)  # Bilinear interpolation
    return resized_data

# Function to segment stroke in an image
def segment_stroke(image_path):
    # Load and preprocess the image
    ct_scan = load_and_resize_nifti(image_path, target_shape)
    ct_scan = (ct_scan - np.min(ct_scan)) / (np.ptp(ct_scan))  # Normalize to
    ct_scan = np.expand_dims(ct_scan, axis=0)  # Add batch dimension
    ct_scan = np.expand_dims(ct_scan, axis=-1)  # Add channel dimension

    # Predict the segmentation mask
    predicted_mask = model.predict(ct_scan)

    # Threshold the mask to get binary segmentation
    binary_mask = (predicted_mask > 0.5).astype(np.float32)

    return binary_mask[0,:,:,:, 0]  # Remove batch and channel dimensions

# Example usage:
image_path = "Segmented Data/ct_scans/077.nii"  # Replace with the actual path to your image
segmented_mask = segment_stroke(image_path)

# You can now visualize or save the segmented mask
# For example, to save the mask as a NIfTI file:
# nib.save(nib.Nifti1Image(segmented_mask, np.eye(4)), "segmented_mask.nii")