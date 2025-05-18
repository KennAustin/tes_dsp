# main.py
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from rppg import rppg_process
from respiration import respiration_process

def main():
    fps = 30
    duration = 60  # in seconds
    frame_buffer = []

    r_signal, g_signal, b_signal = [], [], []
    resp_signal = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("[INFO] Starting real-time signal acquisition for 60 seconds...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process rPPG frame (returns mean RGB and frame with bbox)
        mean_rgb, frame = rppg_process(rgb_frame, frame)
        if mean_rgb is not None:
            r_signal.append(mean_rgb[0])
            g_signal.append(mean_rgb[1])
            b_signal.append(mean_rgb[2])

        # Process respiration frame (returns displacement signal and updated frame)
        y_disp, frame = respiration_process(frame)
        if y_disp is not None:
            resp_signal.append(y_disp)

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Processing and visualizing results...")

    # Show RGB signals
    plt.figure(figsize=(20, 5))
    plt.subplot(3, 1, 1)
    plt.plot(r_signal, color='red')
    plt.title('Red Signal')
    plt.subplot(3, 1, 2)
    plt.plot(g_signal, color='green')
    plt.title('Green Signal')
    plt.subplot(3, 1, 3)
    plt.plot(b_signal, color='blue')
    plt.title('Blue Signal')
    plt.tight_layout()
    plt.show()

    # Show respiration signal
    plt.figure(figsize=(20, 5))
    plt.plot(resp_signal, color='black')
    plt.title('Respiration Signal')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
