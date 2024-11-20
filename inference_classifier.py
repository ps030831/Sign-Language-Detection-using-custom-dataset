import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe components for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary to map class labels to sign language characters
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = [0] * 42  # Ensure data_aux contains 42 features

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture a frame.")
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i, landmark in enumerate(hand_landmarks.landmark):
                if i < 21:  # Only take the first 21 landmarks
                    x = landmark.x
                    y = landmark.y
                    data_aux[i * 2] = x
                    data_aux[i * 2 + 1] = y

            x1 = int(min(data_aux[::2]) * W) - 10
            y1 = int(min(data_aux[1::2]) * H) - 10
            x2 = int(max(data_aux[::2]) * W) - 10
            y2 = int(max(data_aux[1::2]) * H) - 10

            prediction = model.predict([data_aux])  # Pass data_aux as a list

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()