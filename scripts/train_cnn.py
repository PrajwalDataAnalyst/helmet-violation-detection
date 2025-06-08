import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATA_DIR = './cropped_heads'
HELMET_DIR = os.path.join(DATA_DIR, 'helmet')
NO_HELMET_DIR = os.path.join(DATA_DIR, 'no_helmet')

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

def load_data():
    images = []
    labels = []

    # Load helmet images with label 0
    for img_name in os.listdir(HELMET_DIR):
        img_path = os.path.join(HELMET_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(0)  # helmet class

    # Load no_helmet images with label 1
    for img_name in os.listdir(NO_HELMET_DIR):
        img_path = os.path.join(NO_HELMET_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(1)  # no helmet class

    images = np.array(images, dtype='float32') / 255.0  # normalize
    labels = to_categorical(labels, num_classes=2)
    return images, labels

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Total samples: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=15,
                                 zoom_range=0.1,
                                 horizontal_flip=True)
    datagen.fit(X_train)

    model = build_model()
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training model...")
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              callbacks=[early_stop])

    # Save model
    os.makedirs('./models', exist_ok=True)
    model.save('./models/helmet_detector.h5')
    print("Model saved at './models/helmet_detector.h5'")

if __name__ == '__main__':
    main()
