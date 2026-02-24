import sys, cv2 ,os, requests
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("IDscanner.ui", self)

        # Force start page
        self.Form1.setCurrentIndex(0)
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect button
        self.continuep1.clicked.connect(self.go_next)
        self.captureButtonp2.clicked.connect(self.capture_image)
        self.recaptureButtonp2.clicked.connect(self.recapture_image)

        self.uploaded_files = []  # list to store uploaded file data
        self.current_index = -1  # tracks which image is currently shown
        self.deleteButton.clicked.connect(self.delete_current_file)

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

        # Store latest frame (for capture)
        self.current_frame = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.cameraView.size(),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.cameraView.setPixmap(scaled)

    def capture_image(self):
        if not hasattr(self, "current_frame"):
            print("No frame available")
            return

        self.timer.stop()
        self.captured_frame = self.current_frame.copy()

        save_path = "captured_id.jpg"
        cv2.imwrite(save_path, self.captured_frame)

        print(f"Image captured and saved to {save_path}")

        # ---- Send to FastAPI ----
        try:
            with open(save_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://127.0.0.1:5000/ocr", files=files)

            if response.status_code == 200:
                print("OCR Result:", response.json())
            else:
                print("Error:", response.text)

        except Exception as e:
            print("Request failed:", e)

        # ---- Display frozen frame ----
        frame = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.cameraView.size(),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.cameraView.setPixmap(scaled)

        print("Image captured and preview frozen.")


    def recapture_image(self):
        # Resume camera preview
        self.timer.start(30)

        print("Camera resumed for recapture.")

    #Upload Img
    def upload_image(self, target_label):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ID Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            file_name = os.path.basename(file_path)

            file_info = {
                "path": file_path,
                "name": file_name,
                "size": f"{file_size:.2f} MB",
                "status": "Completed"
            }

            self.uploaded_files.append(file_info)
            self.current_index = len(self.uploaded_files) - 1

            self.display_file_details(target_label)

        except Exception as e:
            print("Upload failed:", e)

        #Display Img details
    def display_file_details(self, target_label):
        if self.current_index < 0 or self.current_index >= len(self.uploaded_files):
            return

        file_info = self.uploaded_files[self.current_index]

        frame = cv2.imread(file_info["path"])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            target_label.size(),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        target_label.setPixmap(scaled)

        # 👇 Update UI labels
        self.fileNameLabel.setText(file_info["name"])
        self.fileSizeLabel.setText(file_info["size"])
        self.fileStatusLabel.setText(file_info["status"])
        self.deleteButton.clicked.connect(self.delete_current_file)

    def delete_current_file(self):
            if not self.uploaded_files:
                return

            self.uploaded_files.pop(self.current_index)

            if self.uploaded_files:
                self.current_index = max(0, self.current_index - 1)
                self.display_uploaded_file(self.uploadedImageView)
            else:
                self.current_index = -1
                self.uploadedImageView.clear()
                self.fileNameLabel.setText("")
                self.fileSizeLabel.setText("")
                self.fileStatusLabel.setText("")
        # Optional: show on label or message
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
                self.upload_image(self.uploadedImageView)

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