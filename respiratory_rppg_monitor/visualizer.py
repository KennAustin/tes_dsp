import numpy as np
import cv2

class SignalVisualizer:
    def __init__(self):
        """Inisialisasi plot untuk visualisasi sinyal dengan OpenCV"""
        self.plot_width = 640
        self.plot_height = 480
        self.plot = np.zeros((self.plot_height, self.plot_width, 3), dtype=np.uint8)
        
        # Buffer data
        self.resp_data = []
        self.rppg_data = []
        
    def update_plot(self, resp_value, rppg_value):
        """Memperbarui plot dengan nilai baru"""
        # Update data
        self.resp_data.append(resp_value)
        self.rppg_data.append(rppg_value)
        
        # Batasi jumlah data yang ditampilkan
        max_points = 100
        if len(self.resp_data) > max_points:
            self.resp_data.pop(0)
            self.rppg_data.pop(0)
            
        # Buat plot baru
        self.plot = np.zeros((self.plot_height, self.plot_width, 3), dtype=np.uint8)
        
        # Gambar sinyal pernapasan (biru)
        for i in range(1, len(self.resp_data)):
            y1 = int(self.plot_height/2 - self.resp_data[i-1] * 100)
            y2 = int(self.plot_height/2 - self.resp_data[i] * 100)
            x1 = int((i-1) * self.plot_width / max_points)
            x2 = int(i * self.plot_width / max_points)
            cv2.line(self.plot, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        # Gambar sinyal rPPG (merah)
        for i in range(1, len(self.rppg_data)):
            y1 = int(self.plot_height/2 - self.rppg_data[i-1] * 100)
            y2 = int(self.plot_height/2 - self.rppg_data[i] * 100)
            x1 = int((i-1) * self.plot_width / max_points)
            x2 = int(i * self.plot_width / max_points)
            cv2.line(self.plot, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        return self.plot