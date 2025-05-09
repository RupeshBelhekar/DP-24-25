import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the pre-trained model (Assuming you have a trained ASL model)
model = tf.keras.models.load_model("sign_language_model.h5")  # Replace with your model path

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Define class labels (Modify based on your dataset)
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}  # Extend this for full ASL

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for better interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # Crop and preprocess the image for model input
            hand_crop = frame[y_min:y_max, x_min:x_max]
            hand_crop = cv2.resize(hand_crop, (64, 64))  # Resize to match model input size
            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            hand_crop = hand_crop / 255.0  # Normalize
            hand_crop = np.reshape(hand_crop, (1, 64, 64, 1))  # Reshape for model

            # Predict the letter
            prediction = model.predict(hand_crop)
            pred_class = np.argmax(prediction)
            pred_letter = class_labels.get(pred_class, "Unknown")

            # Display prediction on screen
            cv2.putText(frame, f"Predicted: {pred_letter}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language to Text", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
