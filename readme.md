helmet-violation-detection App
Overview
This project is a Streamlit-based web application designed to promote road safety by detecting whether a bike rider is wearing a helmet from an uploaded image. If the rider is not wearing a helmet, the app automatically detects and extracts the bike’s number plate using computer vision and Optical Character Recognition (OCR). It then registers a fine in a MySQL database and notifies the user accordingly.

The system leverages deep learning for helmet detection, OpenCV for number plate localization, and Tesseract OCR for extracting the plate text.

Features
Helmet Detection: Uses a Convolutional Neural Network (CNN) to classify whether a rider is wearing a helmet.

Number Plate Detection: Employs OpenCV Haar cascades to detect bike number plates in the image.

OCR (Optical Character Recognition): Extracts alphanumeric characters from detected number plates using Tesseract.

Fine Management: Stores and updates fines in a MySQL database when helmet violations are detected.

User-Friendly UI: Provides an easy image upload interface and displays detection results using Streamlit.

Docker Support: Fully dockerized for easy and consistent deployment across environments.

Project Structure
bash
Copy
Edit
helmet-detection-app/
│
├── models/
│   ├── helmet_detector.h5
│   └── haarcascade_russian_plate_number.xml
│
├── scripts/
│   └── database.py             # Database connection and fine update logic
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration file
└── README.md                  # Project documentation
Prerequisites
Python 3.8+

MySQL Server (with database and tables created as per the project schema)

Tesseract OCR installed and added to your system PATH

Docker (optional, for containerized deployment)

Installation & Setup
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/PrajwalDataAnalyst/helmet-violation-detection.git
cd helmet-detection-app
2. Setup Python Environment
Create and activate a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Configure Database
Ensure MySQL Server is running.

Create the database helmet_fines and the fines table:

sql
Copy
Edit
CREATE DATABASE helmet_fines;

USE helmet_fines;

CREATE TABLE fines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) UNIQUE,
    fine_amount INT DEFAULT 0,
    last_fined DATETIME
);
Update the database connection details in scripts/database.py if needed.

5. Install Tesseract OCR
Download and install Tesseract OCR from here.

Add the Tesseract installation directory to your system PATH.

Test Tesseract installation:

bash
Copy
Edit
tesseract --version
Running the Application
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
This will start the web app locally at http://localhost:8501. Open the URL in your browser.

Using Docker (Optional)
Build Docker Image
bash
Copy
Edit
docker build -t helmet-detection-app .
Run Docker Container
bash
Copy
Edit
docker run -p 8501:8501 helmet-detection-app
Open http://localhost:8501 in your browser to access the app.

How to Use
Upload an image of a bike rider.

The app detects if the rider is wearing a helmet.

If no helmet is detected, it extracts the bike’s number plate.

The system registers or updates a fine in the database.

The UI notifies the user about helmet status and fine information.

Technologies Used
Python

TensorFlow / Keras (CNN model for helmet detection)

OpenCV (Number plate detection)

Tesseract OCR

Streamlit (UI)

MySQL (Database)

Docker (Containerization)

Troubleshooting
TesseractNotFoundError: Make sure Tesseract is installed and its path is added to your system environment variables.

Database connection errors: Confirm MySQL is running, credentials in database.py are correct, and tables are created.

Docker issues: Verify Docker daemon is running and you have permission to execute Docker commands.

