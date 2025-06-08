import cv2  # ✅ Import OpenCV

# ✅ Load the number plate cascade file
plate_cascade = cv2.CascadeClassifier('models/haarcascade_russian_plate_number.xml')

# ✅ Check if it loaded correctly
if plate_cascade.empty():
    print("❌ Failed to load cascade. Check path.")
else:
    print("✅ Cascade loaded successfully.")
