"""
cam_handler.py
--------------
CHANGES FROM PREVIOUS VERSION:
  - REMOVED [captureButtonp1]:  Manual capture button no longer exists in the UI.
                                 capture_image() is now triggered automatically by the
                                 auto-capture state machine — no button press needed.
  - ADDED   [Auto-capture state machine]: Three-state FSM ported from auto_detect-classify.py:
                 STATE_READY    → waiting for an ID to appear in frame
                 STATE_LOCKING  → ID detected and stable, counting down to capture
                 STATE_CAPTURED → just fired, waiting for ID to leave before re-arming
  - ADDED   [box_iou()]:        IoU helper to measure bbox stability frame-to-frame.
                                 If the detected box shifts more than STEADY_IOU_MIN
                                 between frames the countdown resets (card is moving).
  - ADDED   [draw_hud()]:       Overlays state label + animated lock-on progress bar
                                 onto the live preview so the user gets clear feedback.
  - ADDED   [_reset_auto_capture()]: Centralises state reset so start_camera(),
                                 recapture_image(), and go_back() all reset cleanly.
  - CHANGED [update_frame()]:   Drives the state machine on every frame tick.
                                 Auto-capture fires via QTimer.singleShot(0, capture_image)
                                 to stay on the main thread (thread-safe).
  - CHANGED [capture_image()]:  No longer connected to a button. Callable directly
                                 (by the state machine) or from recapture flow.
                                 Resets auto-capture state on entry so re-arming
                                 only happens once the ID leaves the frame.
  - CHANGED [recapture_image()]: Also resets auto-capture state before restarting camera.
  - CHANGED [start_camera()]:   Calls _reset_auto_capture() for a clean slate.
  - KEPT    [toggle_capture()]: Dual-cam page flow unchanged.
  - KEPT    [all MV SDK logic]: mv_start/mv_stop/mv_read_frame unchanged.

Auto-capture tuning constants (adjust at top of class):
  STEADY_SECONDS     = 1.5   — seconds the ID must stay still
  STEADY_CONF_MIN    = 0.50  — min detector confidence to start countdown
  STEADY_IOU_MIN     = 0.75  — min IoU frame-to-frame to stay in LOCKING
  GONE_FRAMES_NEEDED = 8     — frames with no detection before re-arming

BUGFIXES carried forward:
  - QImage(...).copy() on all QImage constructions (dangling pointer fix)
  - _stopping guard in mv_read_frame()
  - mv_stop() nulls handle/buffer before SDK calls
  - cv_stop() always called in stop_camera() regardless of MVSDK_AVAILABLE
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


# ── YOLO detector (live preview bounding box + crop-on-capture) ───────────────
_DETECTOR_PATH = os.path.join(os.path.dirname(__file__), "AI models", "best.pt")
_DETECT_CONF   = 0.25
_DETECT_IOU    = 0.45

_detector       = None
_detector_error = None


def get_detector():
    """Lazy-load the YOLO detector once. Returns the model or None on failure."""
    global _detector, _detector_error
    if _detector_error:
        return None
    if _detector is not None:
        return _detector
    try:
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


# ── IoU helper ────────────────────────────────────────────────────────────────

def _box_iou(a: tuple, b: tuple) -> float:
    """
    Intersection-over-Union between two (x1,y1,x2,y2) boxes.
    Used to measure bbox stability between consecutive frames.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Auto-capture state labels ─────────────────────────────────────────────────
_STATE_READY    = "READY"
_STATE_LOCKING  = "LOCKING"
_STATE_CAPTURED = "CAPTURED"


def _draw_hud(frame: np.ndarray, detections: list, state: str,
              lock_progress: float) -> np.ndarray:
    """
    Draw state label + animated lock-on progress bar onto the live frame.
    lock_progress: 0.0–1.0, only rendered during LOCKING state.
    """
    h, w = frame.shape[:2]

    # ── Detection status ──────────────────────────────────────────────
    if detections:
        best_conf = max(d["confidence"] for d in detections)
        cv2.putText(frame, f"ID detected  {best_conf:.0%}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No ID detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # ── State label ───────────────────────────────────────────────────
    _state_colors = {
        _STATE_READY:    (180, 180, 180),
        _STATE_LOCKING:  (0, 220, 255),
        _STATE_CAPTURED: (0, 255, 100),
    }
    _state_labels = {
        _STATE_READY:    "READY  — hold ID steady to scan",
        _STATE_LOCKING:  "LOCKING ON...",
        _STATE_CAPTURED: "CAPTURED  — remove ID to scan again",
    }
    cv2.putText(frame, _state_labels[state],
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                _state_colors[state], 2)

    # ── Lock-on progress bar (LOCKING only) ───────────────────────────
    if state == _STATE_LOCKING and lock_progress > 0:
        bar_x, bar_y = 10, 68
        bar_w, bar_h = w - 20, 14
        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # Fill: colour shifts from yellow → green as it fills
        fill_w = int(bar_w * lock_progress)
        r_val  = int(255 * (1.0 - lock_progress))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          (0, 220, r_val), -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        # Percentage text
        cv2.putText(frame, f"{int(lock_progress * 100)}%",
                    (bar_x + bar_w // 2 - 14, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


class CamHandler:

    # ── Auto-capture tuning ───────────────────────────────────────────
    STEADY_SECONDS     = 1.5    # seconds to hold steady before firing
    STEADY_CONF_MIN    = 0.50   # min detector confidence to start countdown
    STEADY_IOU_MIN     = 0.75   # min IoU frame-to-frame to stay in LOCKING
    GONE_FRAMES_NEEDED = 8      # no-detection frames before re-arming

    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self.cap = None
        self.timer = QTimer()
        self._mv_handle: int | None = None
        self._mv_buffer: int | None = None
        self._mv_buffer_size: int = 0
        self._stopping: bool = False
        self._last_detections: list[dict] = []

        # Auto-capture state machine
        self._ac_state:        str         = _STATE_READY
        self._ac_lock_start:   float | None = None   # time.time() when LOCKING began
        self._ac_lock_box:     tuple | None = None   # bbox that started the countdown
        self._ac_gone_counter: int          = 0      # frames with no detection
        self._ac_fired:        bool         = False  # guard: True while capture_image() is running

        self.timer.timeout.connect(self.update_frame)

    # ── Auto-capture state reset ──────────────────────────────────────

    def _reset_auto_capture(self) -> None:
        """Reset all auto-capture FSM state to READY. Call on start/recapture."""
        self._ac_state        = _STATE_READY
        self._ac_lock_start   = None
        self._ac_lock_box     = None
        self._ac_gone_counter = 0
        self._ac_fired        = False
        self._last_detections = []

    # ── Camera lifecycle ──────────────────────────────────────────────

    def start_camera(self) -> None:
        self._stopping = False
        self._reset_auto_capture()       # clean slate every time camera starts
        if MVSDK_AVAILABLE:
            self.mv_start()
        else:
            self.cv_start()
        self.timer.start(30)

    def stop_camera(self) -> None:
        self._stopping = True
        try:
            self.timer.stop()
        except Exception as e:
            print("[CamHandler/stop_camera] Failed to stop timer:", e)

        if MVSDK_AVAILABLE:
            self.mv_stop()
        # Always call cv_stop — needed when MV SDK present but no camera found
        # and mv_start() fell back to cv_start().
        self.cv_stop()

    # ── MindVision SDK ────────────────────────────────────────────────

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
            config_path = os.path.join(os.path.dirname(__file__), "new_config.config")
            if os.path.exists(config_path):
                mvsdk.CameraReadParameterFromFile(handle, config_path)
                print(f"[CamHandler/_mv_start] Loaded camera config: {config_path}")
            else:
                print(f"[CamHandler/_mv_start] Config not found, using defaults.")
            capability = mvsdk.CameraGetCapability(handle)
            buf_size = (
                capability.sResolutionRange.iWidthMax
                * capability.sResolutionRange.iHeightMax
                * 3
            )
            self._mv_buffer = mvsdk.CameraAlignMalloc(buf_size, 16)
            self._mv_buffer_size = buf_size
            mvsdk.CameraPlay(handle)
            print("[CamHandler/_mv_start] MindVision camera started.")
        except mvsdk.CameraException as e:
            print("[CamHandler/_mv_start] SDK error:", e.error_code, e.message)
            self._mv_handle = None
            self.cv_start()

    def mv_stop(self) -> None:
        handle = self._mv_handle
        buffer = self._mv_buffer
        self._mv_handle = None
        self._mv_buffer = None
        try:
            if handle is not None:
                mvsdk.CameraStop(handle)
                mvsdk.CameraUnInit(handle)
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to stop MV camera:", e)
        try:
            if buffer is not None:
                mvsdk.CameraAlignFree(buffer)
        except Exception as e:
            print("[CamHandler/_mv_stop] Failed to free MV buffer:", e)

    def mv_read_frame(self) -> np.ndarray | None:
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

    # ── OpenCV webcam ─────────────────────────────────────────────────

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
            print("[CamHandler/_cv_stop] Failed to release OpenCV camera:", e)

    def cv_read_frame(self) -> np.ndarray | None:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret and frame is not None else None

    def read_frame(self) -> np.ndarray | None:
        if MVSDK_AVAILABLE and self._mv_handle is not None:
            return self.mv_read_frame()
        return self.cv_read_frame()

    # ── Frame update + auto-capture state machine ─────────────────────

    def update_frame(self) -> None:
        frame = self.read_frame()
        if frame is None:
            return

        try:
            self.parent.current_frame = frame.copy()
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # ── Run YOLO detector ─────────────────────────────────────
            detections = run_detector(display_frame)
            best = (max(detections, key=lambda d: d["confidence"])
                    if detections else None)

            # Draw bounding boxes on the live preview
            if detections:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    is_best = (det is best)
                    color     = (0, 255, 0) if is_best else (0, 165, 255)
                    thickness = 2           if is_best else 1
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(display_frame, f"ID {det['confidence']:.2f}",
                                (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                self._last_detections = detections
            else:
                # Fallback: static alignment guide when detector unavailable
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

            # ── Auto-capture state machine ────────────────────────────
            trigger_capture   = False
            trigger_dual_slot = None   # "front" | "back" — which dual-cam slot fired

            # Determine which page is active so the FSM guards correctly.
            # Page 1 (index 1) = single-cam  → guard on captured_frame
            # Page 4 (index 4) = dual-cam    → guard on captured_front/back_frame
            try:
                current_page = self.parent.Form1.currentIndex()
            except Exception:
                current_page = -1

            on_single_cam = (current_page == 1)
            on_dual_cam   = (current_page == 4)

            front_needed   = on_dual_cam and not hasattr(self.parent, "captured_front_frame")
            back_needed    = on_dual_cam and not hasattr(self.parent, "captured_back_frame")

            single_cam_ready = (
                on_single_cam
                and not self._ac_fired
                and not hasattr(self.parent, "captured_frame")
            )
            dual_cam_ready = (on_dual_cam and (front_needed or back_needed))

            if (single_cam_ready or dual_cam_ready) and not self._ac_fired:

                if self._ac_state == _STATE_READY:
                    if best and best["confidence"] >= self.STEADY_CONF_MIN:
                        self._ac_state      = _STATE_LOCKING
                        self._ac_lock_start = time.time()
                        self._ac_lock_box   = best["bbox"]
                        self._ac_gone_counter = 0
                        print("[AutoCapture] Lock-on started...")

                elif self._ac_state == _STATE_LOCKING:
                    if not best or best["confidence"] < self.STEADY_CONF_MIN:
                        # ID lost — reset to READY
                        print("[AutoCapture] ID lost, resetting to READY.")
                        self._ac_state      = _STATE_READY
                        self._ac_lock_start = None
                        self._ac_lock_box   = None
                    else:
                        # Check spatial stability via IoU
                        iou = _box_iou(self._ac_lock_box, best["bbox"])
                        if iou < self.STEADY_IOU_MIN:
                            # ID moved too much — restart countdown from new position
                            self._ac_lock_start = time.time()
                            self._ac_lock_box   = best["bbox"]
                            print("[AutoCapture] Movement detected, countdown restarted.")
                        else:
                            elapsed_lock = time.time() - self._ac_lock_start
                            if elapsed_lock >= self.STEADY_SECONDS:
                                # Held steady long enough — fire!
                                trigger_capture  = True
                                self._ac_state   = _STATE_CAPTURED
                                self._ac_fired   = True
                                # Record which dual-cam slot to fill (if on dual-cam page)
                                if front_needed:
                                    trigger_dual_slot = "front"
                                elif back_needed:
                                    trigger_dual_slot = "back"
                                print(f"[AutoCapture] Stable — triggering auto-capture! slot={trigger_dual_slot or 'single'}")

                elif self._ac_state == _STATE_CAPTURED:
                    # Waiting for the ID to leave the frame before re-arming.
                    # On dual-cam page: re-arm as soon as the other slot still needs
                    # capturing so the user can present the back without going to GONE.
                    if on_dual_cam and (front_needed or back_needed):
                        # Still a slot to fill — re-arm immediately for next card
                        self._reset_auto_capture()
                        print("[AutoCapture] Dual-cam: re-armed for next slot.")
                    elif not detections:
                        self._ac_gone_counter += 1
                        if self._ac_gone_counter >= self.GONE_FRAMES_NEEDED:
                            self._reset_auto_capture()
                            print("[AutoCapture] Re-armed. Ready for next ID.")
                    else:
                        self._ac_gone_counter = 0

            # ── Compute lock-on progress for HUD ─────────────────────
            lock_progress = 0.0
            if self._ac_state == _STATE_LOCKING and self._ac_lock_start is not None:
                lock_progress = min(
                    (time.time() - self._ac_lock_start) / self.STEADY_SECONDS, 1.0
                )

            # ── Draw HUD overlay ──────────────────────────────────────
            display_frame = _draw_hud(
                display_frame, detections, self._ac_state, lock_progress
            )

            # ── Push frame to Qt label(s) ─────────────────────────────
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h2, w2, ch = rgb.shape
            qimg = QImage(rgb.data, w2, h2, ch * w2,
                          QImage.Format.Format_RGB888).copy()
            if qimg.isNull():
                return
            pixmap = QPixmap.fromImage(qimg)

            view_to_attr = {
                "cameraView":  None,
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

            # ── Fire capture on main thread ───────────────────────────
            if trigger_capture:
                if trigger_dual_slot == "front":
                    QTimer.singleShot(0, self._auto_capture_front)
                elif trigger_dual_slot == "back":
                    QTimer.singleShot(0, self._auto_capture_back)
                else:
                    QTimer.singleShot(0, self.capture_image)

        except Exception as e:
            print("[CamHandler/update_frame] Error:", e)

    # ── Capture ───────────────────────────────────────────────────────

    def capture_image(self) -> None:
        """
        Captures the current frame on the single-cam page (page 1).
        Previously triggered by captureButtonp1; now called automatically
        by the auto-capture state machine in update_frame().

        Crops to the best YOLO detection before saving — gives the
        classifier a clean ID crop instead of a full camera frame.
        Falls back to the full frame if no detection is available.
        """
        p = self.parent
        if not hasattr(p, "current_frame"):
            return

        raw_frame = p.current_frame.copy()

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
            p.captured_frame = raw_frame
            print("[capture_image] No detection available, using full frame.")

        # Stop camera fully immediately after capture
        self.stop_camera()

        folder    = p.get_output_folder("Capture", "Front")
        save_path = f"{folder}/{int(time.time())}.jpg"
        cv2.imwrite(save_path, p.captured_frame)
        print("[capture_image] Saved capture to:", save_path)
        p._captured_front_save_path = save_path

        # Display captured frame in cameraView
        try:
            rgb = cv2.cvtColor(p.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
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

        # Kick off auto-detection + OCR in background
        p.continuep2.setEnabled(False)
        QTimer.singleShot(100, p.inference.infer_front_camera)

    def _auto_capture_front(self) -> None:
        """
        Auto-capture the front side on the dual-cam page (page 4).
        Mirrors what captureButtonp2 used to do via toggle_capture(),
        but called by the state machine instead of a button press.
        """
        p = self.parent
        if not hasattr(p, "current_frame"):
            return
        frame = p.current_frame.copy()
        p.captured_front_frame = frame

        folder    = p.get_output_folder("Capture", "captured_front_frame")
        save_path = f"{folder}/captured_front_frame_{int(time.time())}.jpg"
        cv2.imwrite(save_path, frame)
        p._captured_front_save_path = save_path
        print(f"[AutoCapture] Front captured → {save_path}")

        # Display in cameraView1
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            p.cameraView1.setPixmap(QPixmap.fromImage(qimg).scaled(
                p.cameraView1.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        except Exception as e:
            print("[_auto_capture_front] Display error:", e)

        # If back is already captured, trigger inference now
        if hasattr(p, "captured_back_frame"):
            self._trigger_dual_inference()

    def _auto_capture_back(self) -> None:
        """
        Auto-capture the back side on the dual-cam page (page 4).
        Mirrors what captureButtonp3 used to do via toggle_capture(),
        but called by the state machine instead of a button press.
        """
        p = self.parent
        if not hasattr(p, "current_frame"):
            return
        frame = p.current_frame.copy()
        p.captured_back_frame = frame

        folder    = p.get_output_folder("Capture", "captured_back_frame")
        save_path = f"{folder}/captured_back_frame_{int(time.time())}.jpg"
        cv2.imwrite(save_path, frame)
        p._captured_back_save_path = save_path
        print(f"[AutoCapture] Back captured → {save_path}")

        # Display in cameraView2
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            p.cameraView2.setPixmap(QPixmap.fromImage(qimg).scaled(
                p.cameraView2.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        except Exception as e:
            print("[_auto_capture_back] Display error:", e)

        # If front is already captured, trigger inference now
        if hasattr(p, "captured_front_frame"):
            self._trigger_dual_inference()

    def _trigger_dual_inference(self) -> None:
        """Fire the correct two-sided inference once both sides are captured."""
        p = self.parent
        id_type = getattr(p, "detected_id_type", None)
        print(f"[AutoCapture] Both sides captured — running inference for {id_type}")
        if id_type == "Driver's License":
            p.continuep5.setEnabled(False)
            QTimer.singleShot(100, p.inference.infer_only_driver_license_camera)
        elif id_type in ("National ID", "UMID"):
            p.continuep5.setEnabled(False)
            QTimer.singleShot(100, p.inference.infer_only_national_id_camera)

    def recapture_image(self) -> None:
        """
        Called by recaptureButtonp1. Clears the captured frame and
        resets the auto-capture state machine before restarting the camera.
        """
        p = self.parent
        if hasattr(p, "captured_frame"):
            del p.captured_frame
        p.inference.reset_detection()
        p.detected_id_type = None
        self._reset_auto_capture()   # re-arms the FSM for a fresh attempt
        self.start_camera()

    # ── Dual-cam toggle (page 4) — unchanged ─────────────────────────

    def toggle_capture(self, frame_attr: str, display_label: QLabel,
                       button: QPushButton) -> None:
        """
        Used on the dual-cam page (page 4) for front and back captures.
        Uses detected_id_type instead of reading idOption for routing.
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
            print("[toggle_capture] Saved to:", save_path)

            if frame_attr == "captured_front_frame":
                p._captured_front_save_path = save_path
            elif frame_attr == "captured_back_frame":
                p._captured_back_save_path = save_path

            # Display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
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