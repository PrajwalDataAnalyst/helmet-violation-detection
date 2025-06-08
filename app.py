import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pytesseract
import sys
import os

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from scripts.database import update_fine

# Configure pytesseract path if needed (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update if different

# Load helmet detection model
model = load_model('models/helmet_detector.h5')
labels = ['Helmet', 'No_Helmet']
IMG_SIZE = (128, 128)

# Load OpenCV number plate cascade
plate_cascade = cv2.CascadeClassifier('models/haarcascade_russian_plate_number.xml')

# Function to preprocess frame for helmet detection
def preprocess_frame(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Main detection function
def detect_helmet_and_plate(image):
    frame = np.array(image.convert('RGB'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape
    crop = frame[int(h*0.1):int(h*0.5), int(w*0.3):int(w*0.7)]
    processed = preprocess_frame(crop)

    prediction = model.predict(processed)[0]
    label_index = np.argmax(prediction)
    label = labels[label_index]

    if label == 'Helmet':
        return label, None, prediction[label_index]

    # If no helmet, detect plate
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    plate_number = None
    for (x, y, w_p, h_p) in plates:
        plate_img = frame[y:y+h_p, x:x+w_p]
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        ocr_result = pytesseract.image_to_string(plate_thresh, config='--psm 8')
        plate_number = ''.join(filter(str.isalnum, ocr_result))  # Clean text
        break

    return label, plate_number, prediction[label_index]

# Streamlit app
def main():
    st.title("🛵 Helmet and Number Plate Detection App")

    st.write("Upload an image of a bike rider to check for helmet and number plate.")

    uploaded_image = st.file_uploader("📤 Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, plate_number, confidence = detect_helmet_and_plate(image)

        if label == 'Helmet':
            st.success(f"✅ Helmet detected! Confidence: {confidence:.2f}")
        else:
            st.error(f"❌ No Helmet detected! Confidence: {confidence:.2f}")
            if plate_number:
                st.warning(f"🚫 Detected Number Plate: {plate_number}")
                result_msg = update_fine(plate_number)
                st.info(f"💸 {result_msg}")
            else:
                st.warning("⚠️ No number plate detected.")

if __name__ == '__main__':
    main()
