import cv2
import numpy as np
from face_detector import FaceDetector
from signal_processor import SignalProcessor
from visualizer import SignalVisualizer

def main():
    face_detector = FaceDetector()
    signal_processor = SignalProcessor()
    visualizer = SignalVisualizer()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            frame = cv2.flip(frame, 1)
            landmarks = face_detector.detect_face(frame)
            roi = face_detector.get_roi(frame, landmarks)
            
            resp_value = signal_processor.process_respiration(frame)
            rppg_value = signal_processor.process_rppg(roi)

            # Estimasi BPM & RPM
            bpm = signal_processor.estimate_bpm()
            rpm = signal_processor.estimate_rpm()
            
            signal_plot = visualizer.update_plot(resp_value, rppg_value)

            if landmarks:
                frame = face_detector.draw_face_landmarks(frame, landmarks)
            
            # Tambahkan teks BPM dan RPM
            cv2.putText(frame, f"BPM: {bpm}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"RPM: {rpm}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            combined = np.vstack((frame, signal_plot))
            cv2.imshow('Respiratory & rPPG Monitoring', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()