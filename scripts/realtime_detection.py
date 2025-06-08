import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Correct absolute or relative paths to your model and cascade files
MODEL_PATH = r'P:\Deep_Learning\CNN\models\helmet_detector.h5'
CASCADE_PATH = r'P:\Deep_Learning\CNN\models\haarcascade_russian_plate_number.xml'

# Load trained model
model = load_model(MODEL_PATH)

# Load cascade for plate detection
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

labels = ['Helmet', 'No_Helmet']
IMG_SIZE = (128, 128)

def preprocess_frame(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    cap = cv2.VideoCapture(0)  # webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop heuristic region (person's head)
        h, w, _ = frame.shape
        crop = frame[int(h*0.1):int(h*0.5), int(w*0.3):int(w*0.7)]

        processed = preprocess_frame(crop)
        prediction = model.predict(processed)[0]
        label_index = np.argmax(prediction)
        label = labels[label_index]
        confidence = prediction[label_index]

        # Draw helmet detection result
        color = (0, 255, 0) if label == 'Helmet' else (0, 0, 255)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.rectangle(frame, (int(w*0.3), int(h*0.1)), (int(w*0.7), int(h*0.5)), color, 2)

        # Plate detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25,25))
        for (x, y, w_p, h_p) in plates:
            cv2.rectangle(frame, (x, y), (x+w_p, y+h_p), (255, 0, 0), 2)
            cv2.putText(frame, "Plate", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Helmet and Plate Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
