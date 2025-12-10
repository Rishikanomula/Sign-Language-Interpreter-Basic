from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 64
BATCH_SIZE = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)
DATASET_DIR = r"C:\Users\rishi\Downloads\asl_alphabet_test\asl_alphabet_test"

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_data.num_classes
print("Classes:", train_data.class_indices)
