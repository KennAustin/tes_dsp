# respiration.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load pose detection model only once
model_path = "models/pose_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options_image = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_image)

STANDARD_SIZE = (640, 480)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

features = None
old_gray = None
left_x = top_y = right_x = bottom_y = None

def get_initial_roi(image, x_size=100, y_size=100, shift_x=0, shift_y=0):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = pose_landmarker.detect(mp_image)

    if not detection_result.pose_landmarks:
        return None

    landmarks = detection_result.pose_landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

    center_x += shift_x
    center_y += shift_y

    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y)

    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        return None

    return (left_x, top_y, right_x, bottom_y)

def initialize_features(frame):
    global features, old_gray, left_x, top_y, right_x, bottom_y
    roi_coords = get_initial_roi(frame)
    if roi_coords is None:
        return False

    left_x, top_y, right_x, bottom_y = roi_coords
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[top_y:bottom_y, left_x:right_x]
    features = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.2, minDistance=5, blockSize=3)
    if features is None:
        return False

    features = np.float32(features)
    features[:, :, 0] += left_x
    features[:, :, 1] += top_y
    old_gray = gray.copy()
    return True

def respiration_process(frame):
    global features, old_gray
    y_disp = None
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if features is None or old_gray is None:
        if not initialize_features(frame):
            return None, frame

    new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features, None, **lk_params)
    good_old = features[status == 1]
    good_new = new_features[status == 1]

    if len(good_new) > 0:
        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
        frame = cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 2)
        y_disp = np.mean(good_new[:, 1])
        features = good_new.reshape(-1, 1, 2)
        old_gray = frame_gray.copy()
    else:
        initialize_features(frame)

    return y_disp, frame
