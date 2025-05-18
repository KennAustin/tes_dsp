import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Inisialisasi dan muat model deteksi wajah dari MediaPipe (sekali saja)
base_model = "models/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=base_model)
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

# Konfigurasi detektor wajah
options = FaceDetectorOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,  # Mode IMAGE cocok untuk input frame statis
)

# Buat objek detektor wajah
face_detector = vision.FaceDetector.create_from_options(options)

# Parameter untuk menyesuaikan ROI (Region of Interest) wajah
margin_x = 10               # Margin horizontal tambahan (ke kanan)
scaling_factor = 0.8        # Skala ROI agar tidak terlalu besar

def rppg_process(rgb_frame, frame):
    """
    Memproses satu frame RGB untuk mendeteksi wajah dan menghitung nilai rata-rata RGB
    dari ROI (bagian wajah) sebagai sinyal rPPG.

    Args:
        rgb_frame (numpy.ndarray): Frame dalam format RGB.
        frame (numpy.ndarray): Frame asli (BGR) untuk menampilkan anotasi (bounding box).

    Returns:
        mean_rgb (tuple(float, float, float) | None):
            Tuple nilai rata-rata (R, G, B) pada ROI wajah, atau None jika tidak ada wajah.
        frame (numpy.ndarray): Frame dengan kotak deteksi wajah (untuk ditampilkan).
    """

    # Konversi numpy array menjadi MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Deteksi wajah dalam frame
    result = face_detector.detect(mp_image)

    # Jika terdeteksi minimal satu wajah
    if result.detections:
        for detection in result.detections:
            bboxC = detection.bounding_box  # Ambil koordinat bounding box
            x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

            # Hitung ulang posisi dan ukuran ROI dengan margin dan scaling
            new_x = int(x + margin_x)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)

            # Pastikan koordinat ROI tidak keluar dari frame
            height, width, _ = rgb_frame.shape
            new_y = max(0, y)
            new_x = max(0, new_x)
            new_w = min(new_w, width - new_x)
            new_h = min(new_h, height - new_y)

            # Ekstraksi ROI dari frame RGB
            face_roi = rgb_frame[new_y:new_y+new_h, new_x:new_x+new_w]

            # Cek apakah ROI valid
            if face_roi.size == 0:
                return None, frame

            # Hitung nilai rata-rata RGB di ROI
            mean_rgb = cv2.mean(face_roi)[:3]

            # Gambar bounding box pada frame output (dalam warna hijau)
            cv2.rectangle(frame, (int(new_x), int(new_y)), (int(new_x + new_w), int(new_y + new_h)), (0, 255, 0), 2)

            return mean_rgb, frame  # Return mean RGB dan frame dengan bounding box

    # Jika tidak ada wajah terdeteksi, kembalikan None dan frame asli
    return None, frame