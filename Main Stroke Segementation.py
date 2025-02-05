import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from skimage.transform import resize

def segment_stroke(image_path):
    # Load the CT scan
    ct_scan = nib.load(image_path).get_fdata()
    
    # Resize the CT scan to match the model's expected input shape
    ct_scan_resized = resize(ct_scan, (128, 128, 64), mode='constant', preserve_range=True)
    ct_scan_resized = np.expand_dims(ct_scan_resized, axis=(0, -1))  # Add batch and channel dimensions

    # Load your pre-trained model (replace 'stroke_segmentation_model_3d.h5' with the actual model file)
    model = tf.keras.models.load_model('stroke_segmentation_model_3d.h5')

    # Predict the segmentation mask
    predicted_mask = model.predict(ct_scan_resized)

    # Threshold the mask to get binary segmentation
    binary_mask = (predicted_mask > 0.5).astype(np.float32)

    return ct_scan, binary_mask[0, :, :, :, 0]  # Return original CT scan and remove batch and channel dimensions from mask

# Function to display and save the segmented part over the image
def display_segmentation(ct_scan, segmented_mask):
    # Display the middle slice of each dimension
    slice_z = ct_scan.shape[2] // 2

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original CT Scan")
    plt.imshow(ct_scan[:, :, slice_z], cmap='gray')
    plt.savefig("original_ct_scan.png")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Stroke Area")
    plt.imshow(ct_scan[:, :, slice_z], cmap='gray')
    plt.imshow(segmented_mask[:, :, slice_z], alpha=0.2, cmap='Reds')
    plt.savefig("segmented_stroke_area.png")

    plt.show()

# Example usage:
image_path = "Segmented Data/ct_scans/077.nii"  # Replace with the actual path to your image
ct_scan, segmented_mask = segment_stroke(image_path)
display_segmentation(ct_scan, segmented_mask)