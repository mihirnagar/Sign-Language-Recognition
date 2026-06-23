import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

IMG_SIZE = 64

model = load_model("sign_language_model.h5")

with open("labels.json", "r") as f:
    class_indices = json.load(f)

labels = {value: key for key, value in class_indices.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    sign = labels[class_index]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{sign} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()