import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import shutil
import torch

YOLO_MODEL_PATH = "best.pt"  
CONF_THRESHOLD = 0.7         
ASSUMED_HEAD_WIDTH_CM = 15.5

TSHIRT_OUT_DIR = "tshirt_detections"
FACE_OUT_DIR = "face_detections"

os.makedirs(TSHIRT_OUT_DIR, exist_ok=True)
os.makedirs(FACE_OUT_DIR, exist_ok=True)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def load_yolo_ultralytics(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(model_path).to(device)
    return model

def detect_tshirt_ultralytics(image, model, conf_threshold=CONF_THRESHOLD):
    results = model(image)
    if not results:
        return None
    boxes = results[0].boxes
    filtered_boxes = [box for box in boxes if box.conf.item() >= conf_threshold]
    if len(filtered_boxes) == 0:
        return None
    box = filtered_boxes[0].xyxy[0]
    x1, y1, x2, y2 = box
    x = int(x1.item())
    y = int(y1.item())
    w = int(x2.item() - x1.item())
    h = int(y2.item() - y1.item())
    return [x, y, w, h]

def detect_face(image):
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            return (x, y, w, h)
        else:
            return None

def pose_estimation(full_img):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb_image = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        if results.pose_landmarks:
            ih, iw, _ = full_img.shape
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            ls = (int(left_shoulder.x * iw), int(left_shoulder.y * ih))
            rs = (int(right_shoulder.x * iw), int(right_shoulder.y * ih))
            return ls, rs
        else:
            return None, None

def map_chest_width_to_size(chest_width_cm):
    if chest_width_cm <= 32:
        return "Extra Small"
    elif chest_width_cm <= 34:
        return "Small"
    elif chest_width_cm <= 36:
        return "Medium"
    elif chest_width_cm <= 38:
        return "Large"
    else:
        return "Extra Large"

def process_image(image_path, yolo_model):
    clear_folder(FACE_OUT_DIR)
    clear_folder(TSHIRT_OUT_DIR)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image: {image_path}")
        return None

    tshirt_box = detect_tshirt_ultralytics(image, yolo_model, CONF_THRESHOLD)
    if tshirt_box is None:
        print(f"No T-shirt detected in {image_path}.")
        return None
    x, y, w, h = tshirt_box
    x = max(0, x)
    y = max(0, y)
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    detected_image_path = os.path.join(TSHIRT_OUT_DIR, "detected_" + os.path.basename(image_path))
    cv2.imwrite(detected_image_path, image_with_box)
    print(f"YOLO bounding box drawn image saved: {detected_image_path}")
    
    tshirt_crop = image[y:y+h, x:x+w]
    tshirt_filename = os.path.join(TSHIRT_OUT_DIR, os.path.basename(image_path))
    cv2.imwrite(tshirt_filename, tshirt_crop)
    print(f"T-shirt crop saved: {tshirt_filename}")

    face_box = detect_face(image)
    if face_box is None:
        print(f"No face detected in {image_path}; cannot compute scale factor.")
        return None
    fx, fy, fw, fh = face_box
    face_crop = image[fy:fy+fh, fx:fx+fw]
    face_filename = os.path.join(FACE_OUT_DIR, os.path.basename(image_path))
    cv2.imwrite(face_filename, face_crop)
    print(f"Face detection saved: {face_filename}")

    scale_factor = ASSUMED_HEAD_WIDTH_CM / fw

    ls, rs = pose_estimation(image)
    if ls is None or rs is None:
        print(f"Pose estimation failed to detect shoulder keypoints in {image_path}.")
        return None
    cv2.circle(image_with_box, ls, 5, (0, 0, 255), -1)
    cv2.circle(image_with_box, rs, 5, (0, 0, 255), -1)
    print("Pose keypoints drawn for visualization.")

    chest_width_pixels = abs(rs[0] - ls[0])
    print(f"Chest width in pixels: {chest_width_pixels}")

    chest_width_cm = 1.1 * chest_width_pixels * scale_factor
    size_category = map_chest_width_to_size(chest_width_cm)
    print(f"Estimated Chest Width: {chest_width_cm:.2f} cm")
    print(f"Predicted T-shirt Size for {os.path.basename(image_path)}: {size_category}")
    return size_category