import cv2
import pytesseract

# Make sure pytesseract is installed and Tesseract OCR path is set in system environment variables

def detect_plate_and_ocr(frame, plate_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25,25))

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        text = pytesseract.image_to_string(plate_img, config='--psm 8')  # PSM 8 for single word (number plate)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame
