from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

# Ensure model file exists
if not os.path.exists("model.h5"):
    raise FileNotFoundError("Error: model.h5 not found. Run data_training.py first.")

# Initialize Flask app
app = Flask(__name__)

# Load model and labels
model = load_model("model.h5")
labels = np.load("labels.npy", allow_pickle=True)

emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "anger": "😠",
    "surprise": "😲",
    "rock": "🤘",
    "peace": "✌️",
    "neutral":"😐",
    "hi":"👋"
}

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Global variable to store the latest detected emoji
detected_emoji = "😐"  # Default emoji

def generate_frames():
    global detected_emoji

    while True:
        lst = []
        ret, frm = cap.read()
        if not ret:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Process left hand
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Process right hand
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Make prediction
            lst = np.array(lst).reshape(1, -1)
            pred_index = np.argmax(model.predict(lst))
            pred_label = labels[pred_index]

            # Store detected emoji
            detected_emoji = emoji_map.get(pred_label, "❓")
            
            # Display detected emoji on video frame
            cv2.putText(frm, pred_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2)
            

        '''frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Process left hand
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Process right hand
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Make prediction
            lst = np.array(lst).reshape(1, -1)
            pred_index = np.argmax(model.predict(lst))
            pred_label = labels[pred_index]

            # Store detected emoji
            detected_emoji = emoji_map.get(pred_label, "❓")

            # Display detected text on video frame
            cv2.putText(frm, detected_emoji, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (255, 0, 0), 3)'''

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# New Flask route to serve the detected emoji
@app.route('/emoji_feed')
def emoji_feed():
    global detected_emoji
    return jsonify({"emoji": detected_emoji})

# Flask route for the home page
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Live Emoji Detection</title>
        <style>
        h1 {
            font-size: 96px; /* Increase heading size */
        }
        h2 {
            font-size: 80px; /* Increase subheading size */
        }
        #emoji_display {
            font-size: 90px; /* Make emoji larger */
        }
        </style>
        <script>
        function updateEmoji() {
            fetch('/emoji_feed')
            .then(response => response.json())
            .then(data => {
                document.getElementById('emoji_display').innerText = data.emoji;
            });
        }

        setInterval(updateEmoji, 500); // Update emoji every 500ms
        </script>
    </head>
    <body>
        <h1>Live Emoji Detection</h1>
        <img src="/video_feed" />
        <h2>Detected Emoji: <span id="emoji_display">😐</span></h2>
    </body>
    </html>
    """

# Run Flask app on port 5001
if __name__ == '__main__':
    app.run(debug=True, port=5001)