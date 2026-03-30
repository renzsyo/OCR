"""
cam_handler.py
--------------
CHANGES FROM PREVIOUS VERSION:
  - CHANGED [lines 200-241]:capture_image() — now calls inference.infer_front_camera()
                             instead of infer_page2_camera_passport(); detection is
                             now automatic; removed unused selected_method line
  - FIXED   [lines 205-207]:capture_image() — now calls full stop_camera() immediately
                             after capture instead of just timer.stop()
  - FIXED   [lines 54-67]:  stop_camera() — now always calls cv_stop() regardless of
                             MVSDK_AVAILABLE; when MindVision SDK is present but no
                             camera is found, mv_start() falls back to cv_start() but
                             _mv_handle stays None so mv_stop() does nothing — OpenCV
                             VideoCapture was never released and the camera light
                             stayed on; cv_stop() is now called unconditionally
  - CHANGED [lines 251-298]:toggle_capture() — uses detected_id_type instead of
                             reading idOption for inference routing on dual-cam page
  - CHANGED [lines 242-249]:recapture_image() — now calls inference.reset_detection()
                             and clears detected_id_type for a fresh attempt
  - KEPT    [lines 30-199]: all MindVision SDK logic, cv_start, update_frame unchanged
  - ADDED   [line 234]:     capture_image() — stores save_path as p._captured_front_save_path
                             so review_handler can upload the captured image to Supabase
  - ADDED   [lines 290-292]:toggle_capture() — stores save_path as p._captured_front_save_path
                             or p._captured_back_save_path depending on frame_attr,
                             so review_handler can upload dual-cam captures to Supabase

BUGFIXES (latest):
  - FIXED   [update_frame]:  QImage(rgb.data, ...) now calls .copy() to prevent dangling
                             pointer crash (0xC0000409) — Qt was holding a raw pointer into
                             a numpy buffer that Python GC'd before Qt was done with it
  - FIXED   [capture_image]: Same QImage .copy() fix applied
  - FIXED   [toggle_capture]:Same QImage .copy() fix applied
  - FIXED   [stop_camera]:   Added _stopping guard flag so any in-flight mv_read_frame()
                             call during timer teardown does not access a freed buffer
  - FIXED   [mv_stop]:       Handle and buffer are nulled out before SDK calls so
                             update_frame() sees them as gone immediately; both cleanup
                             steps run independently even if the first raises
  - FIXED   [start_camera]:  Resets _stopping=False on restart for clean state
"""
import cv2, time, os
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


# ── YOLO detector (live preview bounding box + crop-on-capture) ──────────────
# This is the DETECTION model that finds the ID card in the frame.
# It is different from the CLASSIFICATION model in id_classifier.py.
_DETECTOR_PATH = os.path.join(os.path.dirname(__file__), "AI models", "best.pt")
_DETECT_CONF   = 0.25
_DETECT_IOU    = 0.45

_detector       = None   # YOLO instance, loaded lazily on first capture
_detector_error = None   # set on load failure; suppresses repeated attempts

def get_detector():
    """Lazy-load the YOLO detector once. Returns the model or None on failure."""
    global _detector, _detector_error
    if _detector_error:
        return None
    if _detector is not None:
        return _detector
    try:
        import os
        if not os.path.exists(_DETECTOR_PATH):
            _detector_error = f"Detector weights not found: {_DETECTOR_PATH}"
            print(f"[CamHandler] {_detector_error}")
            return None
        from ultralytics import YOLO
        print("[CamHandler] Loading YOLO detector...")
        _detector = YOLO(_DETECTOR_PATH)
        print("[CamHandler] YOLO detector ready.")
        return _detector
    except Exception as e:
        _detector_error = str(e)
        print(f"[CamHandler] Failed to load YOLO detector: {e}")
        return None


def run_detector(frame: np.ndarray) -> list[dict]:
    """
    Run YOLO detection on a BGR frame.
    Returns [{"bbox": (x1,y1,x2,y2), "confidence": float}, ...] or [].
    Falls back to empty list if model unavailable — caller handles gracefully.
    """
    model = get_detector()
    if model is None:
        return []
    try:
        results = model(frame, conf=_DETECT_CONF, iou=_DETECT_IOU, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})
        return detections
    except Exception as e:
        print(f"[CamHandler] Detector error: {e}")
        return []


class CamHandler:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self.cap = None
        self.timer = QTimer()
        self._mv_handle: int | None = None
        self._mv_buffer: int | None = None
        self._mv_buffer_size: int = 0
        self._stopping: bool = False  # guard against in-flight frame reads during teardown
        self._last_detections: list[dict] = []  # latest YOLO results; used by capture_image()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self) -> None:
        self._stopping = False  # ensure clean state on (re)start
        if MVSDK_AVAILABLE:
            self.mv_start()
        else:
            self.cv_start()
        self.timer.start(30)

    def stop_camera(self) -> None:
        # Signal in-flight mv_read_frame() calls to bail out immediately
        # before the buffer is freed below.
        self._stopping = True
        try:
            self.timer.stop()
        except Exception as e:
            print("[CamHandler/stop_camera] Failed to stop timer:", e)

        if MVSDK_AVAILABLE:
            self.mv_stop()
        # Always call cv_stop — when MVSDK is available but no MindVision
        # camera is found, mv_start() falls back to cv_start(), so
        # _mv_handle stays None and mv_stop() does nothing. Without this
        # explicit cv_stop() call the OpenCV VideoCapture is never released
        # and the camera light stays on.
        self.cv_stop()

    def mv_start(self) -> None:
        if self._mv_handle is not None:
            return

        try:
            device_list = mvsdk.CameraEnumerateDevice()
            if not device_list:
                print("[CamHandler/_mv_start] No MindVision camera found.")
                self.cv_start()
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
            self.cv_start()

    def mv_stop(self) -> None:
        # Grab local refs and null out the instance attrs first so that
        # any concurrent update_frame() call sees them as gone immediately.
        handle = self._mv_handle
        buffer = self._mv_buffer
        self._mv_handle = None
        self._mv_buffer = None

        try:
            if handle is not None:
                mvsdk.CameraStop(handle)
                mvsdk.CameraUnInit(handle)
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to stop MV camera", e)

        try:
            if buffer is not None:
                mvsdk.CameraAlignFree(buffer)
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to free MV buffer:", e)

    def mv_read_frame(self) -> np.ndarray | None:
        # Bail out immediately if teardown has started — the buffer may be
        # in the process of being freed by stop_camera() / mv_stop().
        if self._stopping or self._mv_handle is None or self._mv_buffer is None:
            return None
        try:
            raw_data, frame_head = mvsdk.CameraGetImageBuffer(self._mv_handle, 200)
            mvsdk.CameraImageProcess(self._mv_handle, raw_data, self._mv_buffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(self._mv_handle, raw_data)

            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self._mv_buffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (frame_head.iHeight, frame_head.iWidth, 3)
            )
            return cv2.flip(frame.copy(), 0)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("[CamHandler/_mv_read_frame] SDK error:", e.error_code, e.message)
            return None

    def cv_start(self) -> None:
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def cv_stop(self) -> None:
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        except Exception as e:
            print("[CamHandler/_cv_stop] Failed to release OpenCV camera", e)

    def cv_read_frame(self) -> np.ndarray | None:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret and frame is not None else None

    def read_frame(self) -> np.ndarray | None:
        if MVSDK_AVAILABLE and self._mv_handle is not None:
            return self.mv_read_frame()
        return self.cv_read_frame()

    def update_frame(self) -> None:
        frame = self.read_frame()
        if frame is None:
            return

        try:
            self.parent.current_frame = frame.copy()

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Run YOLO detector for live bounding box.
            # Falls back to a static alignment guide if model unavailable.
            detections = run_detector(display_frame)
            if detections:
                # Draw each detected box; highlight the best one
                best = max(detections, key=lambda d: d["confidence"])
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    is_best = det is best
                    color     = (0, 255, 0) if is_best else (0, 165, 255)
                    thickness = 2 if is_best else 1
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(display_frame, f"ID {det['confidence']:.2f}",
                                (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Store latest detections so capture_image() can use them
                self._last_detections = detections
            else:
                # Fallback: static alignment guide
                box_w = int(w * 0.75)
                box_h = int(box_w / 1.586)
                gx1   = (w - box_w) // 2
                gy1   = (h - box_h) // 2
                cv2.rectangle(display_frame,
                              (gx1, gy1), (gx1 + box_w, gy1 + box_h),
                              (100, 100, 100), 1)
                cv2.putText(display_frame, "Align ID within the box",
                            (gx1, gy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                self._last_detections = []

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            # FIXED: .copy() forces Qt to own its data — without it Qt holds a raw
            # pointer into the numpy buffer which Python may GC before Qt is done,
            # causing the 0xC0000409 stack-buffer-overrun crash.
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
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
        """
        Captures the current frame on the single-cam page (page 1).
        CHANGED: crops to the best YOLO detection before saving and passing
                 to inference — this gives the classifier a clean ID crop
                 instead of a full camera frame.
        Falls back to the full frame if no detection is available.
        """
        p = self.parent
        if not hasattr(self.parent, "current_frame"):
            return

        raw_frame = self.parent.current_frame.copy()

        # ── Crop to best YOLO detection if available ──────────────────
        detections = getattr(self, "_last_detections", [])
        if detections:
            best        = max(detections, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["bbox"]
            fh, fw      = raw_frame.shape[:2]
            x1, y1      = max(0, x1), max(0, y1)
            x2, y2      = min(fw, x2), min(fh, y2)
            crop = raw_frame[y1:y2, x1:x2].copy()
            if crop.size > 0:
                p.captured_frame = crop
                print(f"[capture_image] Cropped to detection bbox "
                      f"({x1},{y1})-({x2},{y2}) conf={best['confidence']:.2f}")
            else:
                p.captured_frame = raw_frame
                print("[capture_image] Crop was empty, using full frame.")
        else:
            # No YOLO detection — use full frame (model may not be loaded yet)
            p.captured_frame = raw_frame
            print("[capture_image] No detection available, using full frame.")

        # Stop camera fully (releases hardware) immediately after capture
        self.stop_camera()
        folder    = p.get_output_folder("Capture", "Front")
        save_path = f"{folder}/{int(time.time())}.jpg"
        cv2.imwrite(save_path, p.captured_frame)
        print("Saved capture to:", save_path)
        # Store path so review_handler can upload it to Supabase
        p._captured_front_save_path = save_path

        # Display captured frame in cameraView
        try:
            rgb = cv2.cvtColor(p.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # FIXED: .copy() — same dangling pointer fix as update_frame()
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            if not qimg.isNull():
                pixmap = QPixmap.fromImage(qimg)
                p.cameraView.setPixmap(pixmap.scaled(
                    p.cameraView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        except Exception as e:
            print("[capture_image] Display error:", e)

        # CHANGED: auto-detect ID type from captured frame
        p.continuep2.setEnabled(False)
        QTimer.singleShot(100, p.inference.infer_front_camera)

    def recapture_image(self) -> None:
        p = self.parent
        if hasattr(p, "captured_frame"):
            del p.captured_frame
        # Reset detection state so a fresh attempt can be made
        p.inference.reset_detection()
        p.detected_id_type = None
        self.start_camera()

    def toggle_capture(self, frame_attr: str, display_label: QLabel, button: QPushButton) -> None:
        """
        Used on the dual-cam page (page 4) for front and back captures.
        CHANGED: uses detected_id_type instead of reading idOption for inference routing.
        """

        p = self.parent
        if hasattr(p, frame_attr):
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

            # Save
            folder = p.get_output_folder("Capture", frame_attr)
            save_path = f"{folder}/{frame_attr}_{int(time.time())}.jpg"
            cv2.imwrite(save_path, frame)
            print("Saved to:", save_path)
            # Store path so review_handler can upload it to Supabase
            if frame_attr == "captured_front_frame":
                p._captured_front_save_path = save_path
            elif frame_attr == "captured_back_frame":
                p._captured_back_save_path = save_path

            # Display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # FIXED: .copy() — same dangling pointer fix as update_frame()
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

            id_type = getattr(p, "detected_id_type", None)
            both_captured = (hasattr(p, "captured_front_frame")
                             and hasattr(p, "captured_back_frame"))

            if both_captured:
                if id_type == "Driver's License":
                    p.continuep5.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_driver_license_camera)
                elif id_type in ("National ID", "UMID"):
                    p.continuep5.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_national_id_camera)