import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf

def segment_stroke(image_path):
    ct_scan = nib.load(image_path).get_fdata()
    ct_scan_resized = resize(ct_scan, (128, 128, 64), mode='constant', preserve_range=True)
    ct_scan_resized = np.expand_dims(ct_scan_resized, axis=(0, -1))
    model = tf.keras.models.load_model('stroke_segmentation_model_3d.h5')
    predicted_mask = model.predict(ct_scan_resized)
    binary_mask = (predicted_mask > 0.5).astype(np.float32)
    for i in range(binary_mask.shape[1]):
        if np.any(binary_mask[0, i, :, :, 0]):
            return ct_scan, binary_mask[0, i, :, :, 0], i
    return ct_scan, None, None

def display_segmentation(ct_scan, segmented_mask, layer_index, base_filename):
    if segmented_mask is None:
        print("No stroke detected.")
        return
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original CT Scan")
    plt.imshow(ct_scan[:, :, layer_index], cmap='gray')
    plt.savefig(f"{base_filename}_original.jpg")
    plt.subplot(1, 2, 2)
    plt.title("Segmented Stroke Area")
    plt.imshow(ct_scan[:, :, layer_index], cmap='gray')
    plt.imshow(segmented_mask, alpha=0.2, cmap='Reds')
    plt.savefig(f"{base_filename}_segmented.jpg")
    plt.show()

def process_all_nii_files(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".nii"):
            image_path = os.path.join(directory, file_name)
            ct_scan, segmented_mask, layer_index = segment_stroke(image_path)
            if segmented_mask is not None:
                base_filename = os.path.splitext(file_name)[0]
                display_segmentation(ct_scan, segmented_mask, layer_index, base_filename)
                break

directory = "Segmented Data/ct_scans"
process_all_nii_files(directory)
