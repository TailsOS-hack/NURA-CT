import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split

# Set paths for the CT scans and masks
ct_scan_path = "C:\Users\Windows 10 Home\Desktop\NURA-CT\Segmented Data\ct_scans"
masks_path = "C:\Users\Windows 10 Home\Desktop\NURA-CT\Segmented Data\masks"

# Function to load NIfTI files
def load_nifti(file_path):
    nifti_data = nib.load(file_path)
    return np.array(nifti_data.get_fdata(), dtype=np.float32)

# Load data and masks
ct_scans = []
masks = []

for file_name in sorted(os.listdir(ct_scan_path)):
    if file_name.endswith(".nii"):
        ct_scan = load_nifti(os.path.join(ct_scan_path, file_name))
        ct_scans.append(ct_scan)

for file_name in sorted(os.listdir(masks_path)):
    if file_name.endswith(".nii"):
        mask = load_nifti(os.path.join(masks_path, file_name))
        masks.append(mask)

# Normalize CT scans and binarize masks
ct_scans = np.array(ct_scans)
masks = np.array(masks)

ct_scans = (ct_scans - np.min(ct_scans, axis=(1, 2, 3), keepdims=True)) / (np.ptp(ct_scans, axis=(1, 2, 3), keepdims=True))  # Normalize to [0, 1]
masks = (masks > 0).astype(np.float32)  # Binarize masks

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(ct_scans, masks, test_size=0.2, random_state=42)

# Define the U-Net model
def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    merge4 = concatenate([conv2, up4], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = concatenate([conv1, up5], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Compile the model
model = unet_model(input_size=(128, 128, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Prepare data for training
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    verbose=1
)

# Save the model
model.save("stroke_segmentation_model.h5")
