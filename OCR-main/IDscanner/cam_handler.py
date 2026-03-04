import cv2, time
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class CamHandler:
    def __init__(self, parent):
        self.parent = parent
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(30)

    def stop_camera(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        except Exception:
            pass
    def update_frame(self):
        # Safe camera read and QImage creation
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        try:
            # Keep a copy for capture
            self.parent.current_frame = frame.copy()

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            box_w = int(w * 0.75)
            box_h = int(box_w / 1.586)
            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Align ID within the box", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            if qimg.isNull():
                return
            pixmap = QPixmap.fromImage(qimg)

            view_to_attr = {
                "cameraView": None,
                "cameraView1": "captured_front_frame",
                "cameraView2": "captured_back_frame",
            }
            for view_name, frozen_attr in view_to_attr.items():
                view = getattr(self.parent, view_name, None)
                if view is None:
                    continue
                if frozen_attr and hasattr(self.parent, frozen_attr):
                    continue
                if view.width() == 0 or view.height() == 0:
                    continue
                scaled = pixmap.scaled(
                    view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                view.setPixmap(scaled)
        except Exception as e:
            print("update_frame error:", e)

    def capture_image(self):
        p = self.parent
        # Capture current frame and send to OCR in background
        if not hasattr(self.parent, "current_frame"):
            return

        # Stop timer so preview shows captured frame
        try:
            self.timer.stop()
        except Exception:
            pass

        p.captured_frame = self.parent.current_frame.copy()

        # Save with unique filename to avoid overwriting
        selected_id = p.idOption.currentText()
        folder = p.get_output_folder(selected_id, "Capture")
        save_path = f"{folder}/{int(time.time())}.jpg"
        cv2.imwrite(save_path, p.captured_frame)
        print("Saved capture to:", save_path)

        # Start background thread to send OCR request (non-blocking)
        # Show captured image in cameraView
        try:
            rgb = cv2.cvtColor(p.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            if not qimg.isNull():
                pixmap = QPixmap.fromImage(qimg)
                p.cameraView.setPixmap(pixmap.scaled(
                    p.cameraView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        except Exception as e:
            print("capture_image display error:", e)

    def recapture_image(self):
        p = self.parent
        if hasattr(p, "captured_frame"):
            del p.captured_frame

        self.start_camera()

    def toggle_capture(self, frame_attr, display_label, button):
        p = self.parent
        selected_id = p.idOption.currentText()
        if hasattr(p,frame_attr):
            delattr(p, frame_attr)
            button.setText("Capture Image")
            display_label.clear()
            self.start_camera()
        else:
            if not hasattr(p, "current_frame"):
                return
            frame = p.current_frame.copy()
            setattr(p, frame_attr, frame)
            button.setText("Recapture Image")

            folder = p.get_output_folder(selected_id, "Capture")
            save_path = f"{folder}/{frame_attr}_{int(time.time())}.jpg"

            cv2.imwrite(save_path, frame)
            print("Saved to:", save_path)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
