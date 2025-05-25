🔍 Face Recognition System
An interactive real-time face recognition system built with OpenCV and Python that can capture, train, and recognize faces with confidence scoring.
✨ Features

👤 User Registration: Capture face samples for new users
🧠 Model Training: Train LBPH face recognizer with collected samples
🎯 Real-time Recognition: Live face detection and recognition with confidence scores
📊 Confidence Display: Shows recognition confidence percentage
🔄 Interactive Menu: Easy-to-use command-line interface

🚀 Demo
System Interface
========= FACE RECOGNITION SYSTEM =========
1. Add New User
2. Train Model
3. Recognize Faces
4. Exit
Enter your choice:
Recognition Output

✅ High Confidence: John Doe (85%) - Green rectangle
⚠️ Low Confidence: Not recognized - Recognition threshold not met
📈 Adaptive Counting: Tracks consistent recognition for stability

🛠️ Installation
Prerequisites
bashpip install opencv-python
pip install numpy
pip install Pillow
Setup

Clone this repository

bashgit clone [your-repo-url]
cd computer-vision/face-recognition

Run the system

bashpython face_recognition.py
📋 Usage Guide
Step 1: Add New Users

Select option 1 from the main menu
Enter a unique user ID (integer)
Enter the user's name
Position your face in the camera frame
The system will capture 40 face samples automatically
Press ESC to stop early if needed

Step 2: Train the Model

Select option 2 from the main menu
The system will process all captured face samples
Training data is saved to trainer.yml
User mappings are saved to names.txt

Step 3: Start Recognition

Select option 3 from the main menu
The camera will start and begin recognizing faces
Recognized faces show name and confidence percentage
Press ESC to exit recognition mode

🔧 Technical Details
Architecture

Face Detection: Haar Cascade Classifier
Face Recognition: Local Binary Pattern Histogram (LBPH)
Image Processing: OpenCV and PIL
Data Storage: YAML model file + text mapping

Key Parameters

Confidence Threshold: 70% (adjustable)
Sample Count: 40 images per user
Recognition Stability: 21 consecutive frames for confirmation
Camera Resolution: 640x480

File Structure
face-recognition/
├── face_recognition.py      # Main application
├── dataset/                 # User face samples
│   ├── user1/
│   └── user2/
├── trainer.yml             # Trained model
├── names.txt              # User ID mappings
└── README.md              # This file
🎛️ Configuration
Adjusting Recognition Sensitivity
python# In recognize() function
if confidence < 70:  # Lower = more strict, Higher = more lenient
Camera Settings
pythoncam.set(3, 640)  # Width
cam.set(4, 480)  # Height
🔍 Troubleshooting
IssueSolutionCamera not detectedCheck camera permissions and connectionPoor recognition accuracyEnsure good lighting and capture more samples"No trained data found"Run training (option 2) before recognitionHigh false positivesLower the confidence threshold
🚀 Future Enhancements

 Web interface with Flask/Django
 Multiple face recognition in single frame
 Face verification vs identification modes
 Database integration (SQLite/PostgreSQL)
 REST API endpoints
 Mobile app integration
 Advanced anti-spoofing measures

📊 Performance

Training Time: ~2-5 seconds for 5 users
Recognition Speed: ~30 FPS on average hardware
Accuracy: ~85-95% under good lighting conditions
Memory Usage: ~50-100MB during operation


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
