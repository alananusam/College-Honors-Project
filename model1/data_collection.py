import mediapipe as mp 
import numpy as np 
import cv2
import os
import time

cap = cv2.VideoCapture(0)

EMOTIONS = {"happy", "sad", "anger", "surprise", "rock", "peace","hi"}  # Define valid emotions

while True:
    name = input("Enter the emotion (happy, sad, anger, surprise, rock, peace, hi): ").strip().lower()
    if name in EMOTIONS:
        break  # Exit the loop if input is valid
    
    print(f"Invalid emotion! Choose from: {', '.join(EMOTIONS)}")  # Show available options

# Continue with the rest of the script
print(f"Selected emotion: {name}")

# Remove existing file if it exists
filename = f"{name}.npy"
if os.path.exists(filename):
    os.remove(filename)
    print(f"Existing dataset '{filename}' removed.")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# **Display a countdown before starting**
for i in range(3, 0, -1):
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    
    # Show countdown on screen
    cv2.putText(frm, f"Starting in {i}...", (200, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("window", frm)
    cv2.waitKey(1000)  # Wait for 1 second

cv2.destroyAllWindows()  # Close countdown window before starting

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
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        X.append(lst)
        data_size += 1

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()
        cap.release()
        break

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)
