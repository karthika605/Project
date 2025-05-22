# # Real time Traffic sign detection using image
import cv2
import numpy as np

# Load the input image
image = cv2.imread('trafficsign.png') 
if image is None:
    print("Image not found!")
    exit()

# Resize for easier visualization (optional)
image = cv2.resize(image, (400, 600))

# Blur and convert to HSV
blurred = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Define HSV color ranges
color_ranges = {
    'Red': ([0, 100, 100], [10, 255, 255]),
    'Yellow': ([20, 100, 100], [30, 255, 255]),
    'Green': ([40, 50, 50], [90, 255, 255])
}

# Action labels
actions = {
    'Red': 'STOP',
    'Yellow': 'WAIT',
    'Green': 'START'
}

for color, (lower, upper) in color_ranges.items():
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(image, f"{color}: {actions[color]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print(f"{color} Light Detected → Action: {actions[color]}")

cv2.imshow("Traffic Light Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# # Real time traffic sign detection using videocapture
import cv2
import numpy as np

# Open the webcam (use 0 for default camera)
cap = cv2.VideoCapture('trafficsignal.mp4')

# Define HSV color ranges for traffic lights
color_ranges = {
    'Red': ([0, 100, 100], [10, 255, 255]),
    'Yellow': ([20, 100, 100], [30, 255, 255]),
    'Green': ([40, 50, 50], [90, 255, 255])
}

# Action labels
actions = {
    'Red': 'STOP',
    'Yellow': 'WAIT',
    'Green': 'START'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:  # adjust sensitivity
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, f"{color}: {actions[color]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"{color} Light Detected → Action: {actions[color]}")

    cv2.imshow("Real-Time Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
