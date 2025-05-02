import numpy as np
import cv2
from scipy import signal
from scipy.signal import find_peaks

class SignalProcessor:
    def __init__(self):
        """Inisialisasi parameter pemrosesan sinyal"""
        self.resp_buffer = []
        self.rppg_buffer = []
        self.resp_b, self.resp_a = signal.butter(4, [0.1, 0.5], 'bandpass', fs=30)
        self.rppg_b, self.rppg_a = signal.butter(4, [0.6, 4], 'bandpass', fs=30)

    def process_respiration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[h//3:2*h//3, w//3:2*w//3]
        avg_intensity = np.mean(roi)
        self.resp_buffer.append(avg_intensity)
        if len(self.resp_buffer) > 300:
            self.resp_buffer.pop(0)

        if len(self.resp_buffer) > 30:
            filtered = signal.filtfilt(self.resp_b, self.resp_a, self.resp_buffer)
            return filtered[-1]
        return 0

    def process_rppg(self, roi):
        if roi is None or roi.size == 0:
            return 0

        avg_r = np.mean(roi[:,:,2])
        avg_g = np.mean(roi[:,:,1])
        avg_b = np.mean(roi[:,:,0])
        mean_rgb = np.mean([avg_r, avg_g, avg_b])
        norm_signal = avg_g / (mean_rgb + 1e-6)
        self.rppg_buffer.append(norm_signal)
        if len(self.rppg_buffer) > 300:
            self.rppg_buffer.pop(0)

        if len(self.rppg_buffer) > 30:
            filtered = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_buffer)
            return filtered[-1]
        return 0

    def estimate_bpm(self):
        if len(self.rppg_buffer) < 150:
            return 0
        filtered = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_buffer)
        peaks, _ = find_peaks(filtered, distance=15)
        duration_sec = len(self.rppg_buffer) / 30
        bpm = len(peaks) * (60 / duration_sec)
        return int(bpm)

    def estimate_rpm(self):
        if len(self.resp_buffer) < 150:
            return 0
        filtered = signal.filtfilt(self.resp_b, self.resp_a, self.resp_buffer)
        peaks, _ = find_peaks(filtered, distance=30)
        duration_sec = len(self.resp_buffer) / 30
        rpm = len(peaks) * (60 / duration_sec)
        return int(rpm)
