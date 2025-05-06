import os
import subprocess
from flask import Flask, Response
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from tensorflow.keras.utils import to_categorical

"""Run data collection script."""
print("Starting data collection...")

# Allowed emotion categories
EMOTIONS = {"happy", "sad", "anger", "surprise", "cool", "peace"}

# Capture video
cap = cv2.VideoCapture(0)

# Get user input for the emotion name
name = input("Enter the emotion (happy, sad, anger, surprise, cool, peace): ").strip().lower()

# Check if the name is a valid emotion
if name not in EMOTIONS:
    print("Invalid emotion! Choose from: happy, sad, anger, surprise, cool, peace.")
    cap.release()
    exit()

# Remove existing file if it exists
filename = f"{name}.npy"
if os.path.exists(filename):
    os.remove(filename)
    print(f"Existing dataset '{filename}' removed.")

# Initialize MediaPipe modules
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Data storage
X = []
data_size = 0

while True:
    lst = []
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # No left hand detected

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # No right hand detected

        X.append(lst)
        data_size += 1

    # Draw landmarks
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection", frm)

    # Exit condition: 'Esc' key or 100 samples collected
    if cv2.waitKey(1) == 27 or data_size >= 100:
        break

# Cleanup
cv2.destroyAllWindows()
cv2.waitKey(1)  # Ensure all windows close
cap.release()

# Save data, replacing the file if it already exists
np.save(f"{name}.npy", np.array(X))
print(f"Data saved: {name}.npy with shape {np.array(X).shape}")
print("Data collection complete.")

"""Run data training script."""
print("Starting model training...")

from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not(is_init):
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c+1


for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1


ip = Input(shape=(X.shape[1],))  # Ensure it's a tuple

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))
print("Model training complete.")

"""Run the Flask app for inference."""
print("Starting Flask app...")

# Ensure model file exists
if not os.path.exists("model.h5"):
    raise FileNotFoundError("Error: model.h5 not found. Run data_training.py first.")

# Initialize Flask app
app = Flask(__name__)

# Load model and labels
model = load_model("model.h5")
labels = np.load("labels.npy", allow_pickle=True)

# Emotion-to-emoji mapping
emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "anger": "😠",
    "surprise": "😲",
    "cool": "😎",
    "peace": "✌️",
    "love":"❤️"
}

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Function to generate video frames
def generate_frames():
    while True:
        lst = []
        ret, frm = cap.read()
        if not ret:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        pred_label = "Detecting..."
        emoji = ""

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

            # Get corresponding emoji
            emoji = emoji_map.get(pred_label, "")

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        # Display text (emotion + emoji)
        display_text = f"{pred_label} {emoji}"
        cv2.putText(frm, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2)

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

# Flask route for the home page
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Live Emoji Detection</title>
    </head>
    <body>
        <h1>Live Emoji Detection</h1>
        <img src="/video_feed" />
    </body>
    </html>
    """

# Run Flask app on port 5001
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)