import tkinter as tk
from tkinter import messagebox
from threading import Thread
import cv2
import time
import matplotlib.pyplot as plt

from rppg import rppg_process
from respiration import respiration_process

class RPPGApp:
    """
    Kelas utama untuk GUI sistem deteksi rPPG dan respirasi.

    Atribut:
        root (tk.Tk): Objek root dari Tkinter.
        running (bool): Status proses akuisisi sedang berjalan atau tidak.
        duration_entry (tk.Entry): Input durasi dari user.
        start_btn (tk.Button): Tombol untuk memulai proses.
        log_text (tk.Text): Area log untuk menampilkan status aplikasi.

    Metode:
        log(message): Menambahkan log ke tampilan GUI.
        start_process(): Validasi input dan memulai thread proses akuisisi.
        run_acquisition(duration): Fungsi utama untuk akuisisi video dan analisis sinyal.
    """

    def __init__(self, root):
        """
        Inisialisasi GUI, membuat elemen-elemen antarmuka dan mengatur layout.
        """
        self.root = root
        self.root.title("Sistem rPPG & Respirasi")
        self.running = False  # Flag untuk menandai apakah proses sedang berjalan

        # Input durasi
        tk.Label(root, text="Durasi (detik):").grid(row=0, column=0, padx=10, pady=10)
        self.duration_entry = tk.Entry(root)
        self.duration_entry.insert(0, "60")  # Default durasi 60 detik
        self.duration_entry.grid(row=0, column=1)

        # Tombol mulai
        self.start_btn = tk.Button(root, text="Mulai", command=self.start_process)
        self.start_btn.grid(row=1, column=0, columnspan=2, pady=10)

        # Area log untuk menampilkan status
        self.log_text = tk.Text(root, height=10, width=50, state="disabled")
        self.log_text.grid(row=2, column=0, columnspan=2, padx=10)

    def log(self, message):
        """
        Menambahkan pesan ke area log di GUI.

        Parameter:
            message (str): Teks yang akan ditampilkan di log.
        """
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_process(self):
        """
        Memvalidasi input durasi dan memulai proses akuisisi video di thread terpisah.
        """
        if self.running:
            return
        try:
            duration = int(self.duration_entry.get())
            self.running = True
            self.start_btn.config(state="disabled")  # Disable tombol saat berjalan
            Thread(target=self.run_acquisition, args=(duration,)).start()
        except ValueError:
            messagebox.showerror("Error", "Durasi harus berupa angka.")

    def run_acquisition(self, duration):
        """
        Proses utama untuk akuisisi video dari kamera, ekstraksi sinyal rPPG dan respirasi,
        serta menampilkan hasil dalam bentuk grafik.

        Parameter:
            duration (int): Durasi proses pengambilan data dalam detik.
        """
        self.log("[INFO] Memulai akuisisi sinyal...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("Error: Tidak bisa mengakses kamera.")
            self.running = False
            return

        # Buffer untuk sinyal RGB dan pernapasan
        r_signal, g_signal, b_signal = [], [], []
        resp_signal = []

        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                self.log("Error: Gagal membaca frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ekstraksi sinyal rPPG
            mean_rgb, frame = rppg_process(rgb_frame, frame)
            if mean_rgb is not None:
                r_signal.append(mean_rgb[0])
                g_signal.append(mean_rgb[1])
                b_signal.append(mean_rgb[2])

            # Ekstraksi sinyal pernapasan
            y_disp, frame = respiration_process(frame)
            if y_disp is not None:
                resp_signal.append(y_disp)

            # Tampilkan feed video
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.log("[INFO] Akuisisi selesai. Menampilkan grafik...")

        # Plot sinyal RGB
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

        # Plot sinyal pernapasan
        plt.figure(figsize=(20, 5))
        plt.plot(resp_signal, color='black')
        plt.title('Respiration Signal')
        plt.tight_layout()
        plt.show()

        # Reset status
        self.running = False
        self.start_btn.config(state="normal")

if __name__ == "__main__":
    # Inisialisasi dan jalankan GUI
    root = tk.Tk()
    app = RPPGApp(root)
    root.mainloop()