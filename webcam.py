import cv2
import numpy as np
from tensorflow.keras.models import load_model

#LOAD MODEL
model = load_model("asl_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMAGE_SIZE = 64

#WEBCAM SETUP
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    #ROI (Region of Interest)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    #Preprocess
    img = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    #Prediction
    pred = model.predict(img, verbose=0)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    label = labels[class_index]

    #DISPLAY
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.imshow("ASL Webcam Prediction", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#CLEANUP
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
