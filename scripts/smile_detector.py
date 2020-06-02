from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
from config import haar_cascade, save_model_path

haar_face_detector = cv2.CascadeClassifier(haar_cascade)
model = load_model(save_model_path)

# grab the reference to the web cam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    # get the bounding box the face detected
    faces = haar_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in faces:
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (28,28))
        roi = img_to_array(roi)
        roi = roi.astype("float") / 255.0
        print("Before: ", roi.shape)
        roi = np.expand_dims(roi, axis=0) # padding with extra dimensions
        print("After: ", roi.shape)

        (not_smiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling>not_smiling else "Not Smiling"

        cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frame_clone)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()