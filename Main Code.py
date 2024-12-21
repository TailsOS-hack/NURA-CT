import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
import cv2

# Define directories
data_dir = "path_to_dataset"
normal_dir = os.path.join(data_dir, "normal")
stroke_dir = os.path.join(data_dir, "stroke")

# Image parameters
img_height = 256
img_width = 256
batch_size = 16

# Load and preprocess images
def load_images(image_dir, label):
    images = []
    labels = []
    for img_path in glob(os.path.join(image_dir, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

normal_images, normal_labels = load_images(normal_dir, 0)
stroke_images, stroke_labels = load_images(stroke_dir, 1)

# Combine datasets
images = np.concatenate((normal_images, stroke_images), axis=0)
labels = np.concatenate((normal_labels, stroke_labels), axis=0)

# Normalize pixel values
images = images / 255.0

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define U-Net model for pixel-by-pixel classification
def build_unet():
    inputs = Input((img_height, img_width, 3))

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = Concatenate()([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = Concatenate()([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

# Compile model
model = build_unet()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Prepare data generators for augmentation
data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

data_gen = ImageDataGenerator(**data_gen_args)
train_gen = data_gen.flow(X_train, y_train, batch_size=batch_size)

# Train model
history = model.fit(train_gen, epochs=20, validation_data=(X_val, y_val))

# Save model
model.save("stroke_detection_model.h5")

print("Training complete. Model saved as 'stroke_detection_model.h5'")
