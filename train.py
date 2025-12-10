import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# CONFIG
DATASET_DIR = r"C:\Users\rishi\Downloads\asl_small"
IMAGE_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 20

# DATA AUGMENTATION

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_data.num_classes
CLASS_LABELS = list(train_data.class_indices.keys())

# MODEL
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAIN
model.fit(
    train_data,
    steps_per_epoch=40,
    epochs=EPOCHS
)

# SAVE MODEL + LABELS
model.save("asl_model.h5")

with open("labels.txt", "w") as f:
    for label in CLASS_LABELS:
        f.write(label + "\n")

print("✅ Model and labels saved successfully")