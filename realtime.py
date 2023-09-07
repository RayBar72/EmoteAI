import cv2
import numpy as np
from tensorflow.keras import models as K

# Cargar el modelo entrenado
model = K.load_model('FinalModel.h5')

cv2.ocl.setUseOpenCL(False)

emotions = {0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_color = frame[y:y + h, x:x + w]

        normalized_roi = roi_color / 255.0

        input_array = np.expand_dims(normalized_roi, axis=0)

        prediction = model.predict(input_array)

        maxindex = int(np.argmax(prediction))

        cv2.putText(frame,
                    emotions[maxindex],
                    (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
