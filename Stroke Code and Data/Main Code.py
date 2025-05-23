import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

ct_scan_path = "Segmented Data/ct_scans"
masks_path = "Segmented Data/masks"
target_shape = (128, 128, 64)

def load_and_resize_nifti(file_path, target_shape):
    nifti_data = nib.load(file_path)
    data = np.array(nifti_data.get_fdata(), dtype=np.float32)
    resize_factors = [
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    ]
    resized_data = zoom(data, resize_factors, order=1)
    return resized_data

ct_scans = []
masks = []

for file_name in sorted(os.listdir(ct_scan_path)):
    if file_name.endswith(".nii"):
        ct_scan = load_and_resize_nifti(os.path.join(ct_scan_path, file_name), target_shape)
        ct_scans.append(ct_scan)

for file_name in sorted(os.listdir(masks_path)):
    if file_name.endswith(".nii"):
        mask = load_and_resize_nifti(os.path.join(masks_path, file_name), target_shape)
        masks.append(mask)

ct_scans = np.array(ct_scans)
masks = np.array(masks)

ct_scans = (ct_scans - np.min(ct_scans, axis=(1, 2, 3), keepdims=True)) / (
    np.ptp(ct_scans, axis=(1, 2, 3), keepdims=True))
masks = (masks > 0).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(ct_scans, masks, test_size=0.2, random_state=42)

def unet_model_3d(input_size=(128, 128, 64, 1)):
    inputs = Input(input_size)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    up4 = UpSampling3D(size=(2, 2, 2))(conv3)
    merge4 = concatenate([conv2, up4], axis=-1)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge4)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    merge5 = concatenate([conv1, up5], axis=-1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv5)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = unet_model_3d(input_size=(128, 128, 64, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=1,
    verbose=1
)

model.save("stroke_segmentation_model_3d.h5")
