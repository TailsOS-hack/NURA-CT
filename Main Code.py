import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageDraw, ImageFilter

# Paths to your data folders
stroke_dir = 'stroke'
normal_dir = 'normal'

# Parameters
img_size = 128
batch_size = 32

# Data preprocessing
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    '.',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    '.',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Evaluate the model
test_imgs, test_labels = [], []
for img_batch, label_batch in val_data:
    test_imgs.extend(img_batch)
    test_labels.extend(label_batch)
    if len(test_imgs) >= val_data.samples:
        break

preds = model.predict(np.array(test_imgs))
preds = (preds > 0.5).astype(int).flatten()
accuracy = accuracy_score(test_labels[:len(preds)], preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Draw border if stroke detected
def draw_border(image, prediction):
    if prediction == 1:
        gray = ImageOps.grayscale(image)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        draw = ImageDraw.Draw(image)
        for y, x in np.argwhere(edges_array > 128):
            draw.point((x, y), fill="red")
    return image

# Test with images from the folders
stroke_test_dir = os.path.join(stroke_dir, 'test')
normal_test_dir = os.path.join(normal_dir, 'test')

for folder in [stroke_test_dir, normal_test_dir]:
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue
        img_resized = ImageOps.fit(img, (img_size, img_size))
        img_array = np.array(img_resized) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0))[0][0] > 0.5
        bordered_img = draw_border(img, int(pred))
        bordered_img.show()
