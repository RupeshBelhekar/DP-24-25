import cv2
import serial
import time
from ultralytics import YOLO
import pyttsx3
import threading

# Set up Serial Communication with Arduino (Update the COM port)
arduino = serial.Serial(port="COM7", baudrate=9600, timeout=0.1)  
time.sleep(2)  # Allow connection to establish

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)  
engine.setProperty("volume", 1.0)

# Open webcam
cap = cv2.VideoCapture(0)

distance = "N/A"  # Default value for distance

def read_distance():
    """Continuously read distance from Arduino in a separate thread."""
    global distance
    while True:
        if arduino.in_waiting > 0:
            distance_data = arduino.readline().decode().strip()
            if distance_data.replace('.', '', 1).isdigit():  # Check if numeric
                distance = distance_data  # Update global distance variable
            print("Arduino Distance:", distance)

# Start a separate thread for reading distance
distance_thread = threading.Thread(target=read_distance, daemon=True)
distance_thread.start()

def speak(text):
    """Speak the given text using the TTS engine."""
    engine.say(text)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model.predict(source=frame, show=True, conf=0.5)

    detected = False  
    detected_objects = []  

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            object_name = model.names[cls]  
            detected_objects.append(object_name)

            center_x = (x1 + x2) // 2

            # Determine object position
            if center_x < frame.shape[1] // 3:
                position = "left"
            elif center_x > (2 * frame.shape[1]) // 3:
                position = "right"
            else:
                position = "center"

            # Generate voice alert
            alert_text = f"{object_name} on the {position}. Distance: {distance} cm."
            print(alert_text)
            speak(alert_text)
            detected = True  

    # If an object is detected, send alert to Arduino
    if detected:
        arduino.write(b"OBJECT_DETECTED\n")  
        print("Object detected! Alert sent to Arduino.")

    # Display distance on screen
    cv2.putText(frame, f"Distance: {distance} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
