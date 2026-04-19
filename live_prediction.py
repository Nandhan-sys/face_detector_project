import cv2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
df = pd.read_csv('driver_attention_data.csv')
X = df[['face_present', 'face_size', 'face_centered']]
y = df['label']
model = DecisionTreeClassifier()
model.fit(X, y)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face_coordinates:
        # Extract features for prediction
        face_present = 1
        face_size = w * h
        face_centered = 1 if abs((x + w / 2) - frame.shape[1] / 2) < 50 else 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sample = [[face_present, face_size, face_centered]]
        prediction = model.predict(sample)[0]
        cv2.putText(frame, f'Prediction: {prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()