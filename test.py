import cv2
import numpy as np
from tensorflow.keras.models import load_model

#LOAD MODEL+LABELS
model = load_model("asl_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

#TEST IMAGE
IMAGE_PATH = r"C:\Users\rishi\Downloads\asl_alphabet_test\asl_alphabet_test\A\A_test.jpg"
IMAGE_SIZE = 64

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

#PREDICTION
pred = model.predict(img)
pred_class = np.argmax(pred)

print("✅ Predicted ASL Letter:", labels[pred_class])
