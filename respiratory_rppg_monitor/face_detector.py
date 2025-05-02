import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        """Inisialisasi detektor wajah MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
    def detect_face(self, frame):
        """Mendeteksi wajah dan landmark dalam frame"""
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        return results.multi_face_landmarks[0]
        
    def get_roi(self, frame, landmarks):
        """Mendapatkan ROI untuk analisis rPPG dari pipi"""
        if landmarks is None:
            return None

        h, w = frame.shape[:2]
        try:
            cheek_points = np.array([
                [int(landmarks.landmark[123].x * w), int(landmarks.landmark[123].y * h)],
                [int(landmarks.landmark[50].x * w), int(landmarks.landmark[50].y * h)],
                [int(landmarks.landmark[280].x * w), int(landmarks.landmark[280].y * h)],
                [int(landmarks.landmark[352].x * w), int(landmarks.landmark[352].y * h)]
            ], np.int32)
        except IndexError:
            return None
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [cheek_points], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi
        
    def draw_face_landmarks(self, frame, landmarks):
        """Menggambar landmark wajah pada frame"""
        if landmarks is None:
            return frame
            
        h, w = frame.shape[:2]
        for idx in [123, 50, 280, 352]:
            try:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            except IndexError:
                continue
        return frame
