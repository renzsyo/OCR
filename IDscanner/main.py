import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("IDscanner.ui", self)

        # Force start page
        self.Form1.setCurrentIndex(0)
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)


        # Connect button
        self.continuep1.clicked.connect(self.go_next)

    def start_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        self.timer.start(30)  # refresh every 30ms

    def stop_camera(self):
        self.timer.stop()
        self.cap.release()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Show in QLabel (cameraView)
        scaled = pixmap.scaled(
            self.cameraView.size(),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.cameraView.setPixmap(scaled)

    def go_next(self):
        selected_id = self.idOption.currentText()

        if not (self.uploadOption.isChecked() or self.cameraOption.isChecked()):
            print("Select camera or upload")
            return

        # Passport
        if selected_id == "Passport":
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(1)
                self.start_camera()
            elif self.uploadOption.isChecked():
                self.stop_camera()
                self.Form1.setCurrentIndex(2)

        # National ID
        elif selected_id == "National ID":
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(4)
                self.start_camera()
            elif self.uploadOption.isChecked():
                self.stop_camera()
                self.Form1.setCurrentIndex(5)

        # Driver's License
        elif selected_id == "Driver's License":
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(4)
                self.start_camera()
            elif self.uploadOption.isChecked():
                self.stop_camera()
                self.Form1.setCurrentIndex(5)

    def closeEvent(self, event):
        # Stop camera when app closes
        self.stop_camera()
        super().closeEvent(event)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
