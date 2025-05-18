# rppg.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Load face detection model only once
base_model = "models/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=base_model)
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

options = FaceDetectorOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
)
face_detector = vision.FaceDetector.create_from_options(options)

margin_x = 10
scaling_factor = 0.8

def rppg_process(rgb_frame, frame):
    """
    Memproses satu frame untuk mendeteksi wajah dan mengambil rata-rata RGB dari ROI.
    Mengembalikan nilai RGB rata-rata dan frame dengan bbox (jika terdeteksi).
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = face_detector.detect(mp_image)
    if result.detections:
        for detection in result.detections:
            bboxC = detection.bounding_box
            x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

            new_x = int(x + margin_x)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)

            face_roi = rgb_frame[y:y+new_h, new_x:new_x+new_w]

            if face_roi.size == 0:
                return None, frame

            mean_rgb = cv2.mean(face_roi)[:3]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + new_w), int(y + new_h)), (0, 255, 0), 2)
            return mean_rgb, frame

    return None, frame