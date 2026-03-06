import cv2, time, sys,os
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

try:
    import mvsdk
    MVSDK_AVAILABLE = True
    print("[CamHandler] MindVision SDK found.")
except Exception as e:
    MVSDK_AVAILABLE = False
    print("[CamHandler] MindVision SDK failed:", e)


class CamHandler:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self.cap = None
        self.timer = QTimer()
        self._mv_handle: int | None = None
        self._mv_buffer: int | None = None
        self._mv_buffer_size: int = 0
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self) -> None:
        if MVSDK_AVAILABLE:
            self._mv_start()
        else:
            self._cv_start()
        self.timer.start(30)

    def stop_camera(self) -> None:
        try:
            self.timer.stop()
        except Exception as e:
            print("[CamHandler/stop_camera] Failed to stop timer:", e)

        if MVSDK_AVAILABLE:
            self._mv_stop()
        else:
            self._cv_stop()

    def _mv_start(self) -> None:
        if self._mv_handle is not None:
            return

        try:
            device_list = mvsdk.CameraEnumerateDevice()
            if not device_list:
                print("[CamHandler/_mv_start] No MindVision camera found.")
                self._cv_start()
                return
            device_info = device_list[0]
            handle = mvsdk.CameraInit(device_info, -1, -1)
            self._mv_handle = handle

            mvsdk.CameraSetIspOutFormat(handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            capability = mvsdk.CameraGetCapability(handle)
            buf_size = (
                capability.sResolutionRange.iWidthMax
                * capability.sResolutionRange.iHeightMax
                * 3
            )
            self._mv_buffer = mvsdk.CameraAlignMalloc(buf_size, 16)
            self._mv_buffer_size = buf_size

            mvsdk.CameraPlay(handle)
            print("[CamHandler/_mv_start] MindVision camera start.")

        except mvsdk.CameraException as e:
            print("[CamHandler/_mv_start] SDK error:", e.error_code, e.message)
            self._mv_handle = None
            self._cv_start()

    def _mv_stop(self) -> None:
        try:
            if self._mv_handle is not None:
                mvsdk.CameraStop(self._mv_handle)
                mvsdk.CameraUnInit(self._mv_handle)
                self._mv_handle = None
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to stop MV camera", e)

        try:
            if self._mv_buffer is not None:
                mvsdk.CameraAlignFree(self._mv_buffer)
                self._mv_buffer = None
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to free MV buffer:", e)

    def _mv_read_frame(self) -> np.ndarray | None:
        if self._mv_handle is None or self._mv_buffer is None:
            return None
        try:
            raw_data, frame_head = mvsdk.CameraGetImageBuffer(self._mv_handle, 200)
            mvsdk.CameraImageProcess(self._mv_handle, raw_data, self._mv_buffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(self._mv_handle, raw_data)

            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self._mv_buffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (frame_head.iHeight, frame_head.iWidth, 3)
            )
            return cv2.flip(frame.copy(), 1)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("[CamHandler/_mv_read_frame] SDK error:", e.error_code, e.message)
            return None

    def _cv_start(self) -> None:
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _cv_stop(self) -> None:
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        except Exception as e:
            print("[CamHandler/_cv_stop] Failed to release OpenCV camera", e)

    def _cv_read_frame(self) -> np.ndarray | None:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret and frame is not None else None

    def _read_frame(self) -> np.ndarray | None:
        if MVSDK_AVAILABLE and self._mv_handle is not None:
            return self._mv_read_frame()
        return self._cv_read_frame()

    def update_frame(self) -> None:
        frame = self._read_frame()
        if frame is None:
            return

        try:
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
            print("[CamHandler/update_frame] Error:", e)

    def capture_image(self) -> None:
        p = self.parent
        # Capture current frame and send to OCR in background
        if not hasattr(self.parent, "current_frame"):
            return

        # Stop timer so preview shows captured frame
        try:
            self.timer.stop()
        except Exception as e:
            print("[CamHandler/capture_image] Failed to stop timer:", e)

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
                p.continuep2.setEnabled(False)
                QTimer.singleShot(100, p.inference.infer_page2_camera_passport)
        except Exception as e:
            print("capture_image display error:", e)

    def recapture_image(self) -> None:
        p = self.parent
        if hasattr(p, "captured_frame"):
            del p.captured_frame

        self.start_camera()

    def toggle_capture(self, frame_attr: str, display_label: QLabel, button: QPushButton) -> None:
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
            selected_id = p.idOption.currentText()
            if selected_id == "Driver's License":
                if hasattr(p, "captured_front_frame") and hasattr(p, "captured_back_frame"):
                    p.continuep5.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_driver_license_camera)
            if selected_id == "National ID":
                if hasattr(p, "captured_front_frame") and hasattr(p, "captured_back_frame"):
                    p.continuep5.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_national_id_camera)
