import cv2
import numpy as np
import time
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

output_folder = "img_captured"
os.makedirs(output_folder, exist_ok=True)

zoom_scale = 1.0
zoom_target = 1.0
zoom_speed = 0.05

last_capture_time = 0
capture_cooldown = 2  # seconds

def smooth_zoom(frame, center_x, center_y, zoom_factor):
    h, w = frame.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = max(center_x - new_w // 2, 0)
    y1 = max(center_y - new_h // 2, 0)
    x2 = min(center_x + new_w // 2, w)
    y2 = min(center_y + new_h // 2, h)

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    original_frame_with_rectangles = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(original_frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
        center_x, center_y = x + w // 2, y + h // 2
        zoom_target = 2.0

        current_time = time.time()
        if current_time - last_capture_time >= capture_cooldown:
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (400, 400))
            filename = os.path.join(output_folder, f"face_{int(current_time)}.jpg")
            cv2.imwrite(filename, face_crop)
            print(f"✅ Captured: {filename}")
            last_capture_time = current_time
    else:
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        zoom_target = 1.0

    zoom_scale += (zoom_target - zoom_scale) * zoom_speed
    zoomed = smooth_zoom(frame, center_x, center_y, zoom_scale)

    cv2.imshow("Original View with Face Detections", original_frame_with_rectangles)
    cv2.imshow("Face Zoom Auto Capture", zoomed)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("🚪 ESC pressed. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()