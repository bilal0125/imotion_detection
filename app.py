from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# Load pre-trained face detection model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load pre-trained emotion recognition model
emotion_model = load_model('FacialEmotionR1.h5')
# Define emotions
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store emotion data
data = []

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            # Predict emotion
            predicted_emotion = emotion_model.predict(face_roi)[0]
            emotion_label = emotion_labels[np.argmax(predicted_emotion)]

            # Draw rectangle and label around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Log data with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_id = f"Student_{x}_{y}"
            data.append({"Timestamp": timestamp, "Student ID": student_id, "Emotion": emotion_label})

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/report')
def generate_report():
    df = pd.DataFrame(data)

    if df.empty:
        return jsonify({"error": "No data captured"})

    dominant_emotions = df.groupby('Student ID')['Emotion'].agg(lambda x: x.value_counts().idxmax())
    overall_dominant_emotion = dominant_emotions.mode()[0]

    # Emotion-based feedback analysis
    analysis = ""
    if overall_dominant_emotion == 'neutral':
        analysis = ("The dominant emotion was 'neutral.' This suggests the class might have been well-paced, "
                    "but could benefit from more interactive activities to increase engagement.")
    elif overall_dominant_emotion == 'happy':
        analysis = ("The dominant emotion was 'happy.' This indicates that the class was likely engaging and positive. "
                    "Continue using similar strategies to maintain this positive atmosphere.")
    elif overall_dominant_emotion == 'sad':
        analysis = ("The dominant emotion was 'sad.' This could indicate that students were disengaged or found the material challenging. "
                    "Consider reviewing the material or offering additional support.")
    # Additional emotion-based analyses...

    return jsonify({
        "dominant_emotion": overall_dominant_emotion,
        "analysis": analysis
    })

if __name__ == "__main__":
    app.run(debug=True)
