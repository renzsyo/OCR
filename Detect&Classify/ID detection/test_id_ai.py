"""
test_yolo_live.py
─────────────────────────────────────────────────────────────
YOLOv8 Live Inference Test — Industrial Camera
Tests ID card detection in real time using MindVision camera

Controls:
    ENTER — capture current frame and save result
    Q     — quit
─────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import os
import sys
import time
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mvsdk
    MVSDK_AVAILABLE = True
    print("[Camera] MindVision SDK found.")
except Exception as e:
    MVSDK_AVAILABLE = False
    print("[Camera] MindVision SDK not found, falling back to webcam:", e)


# ─────────────────────────────────────────────
#  CONFIG — edit this path to match your setup
# ─────────────────────────────────────────────
OUTPUT_DIR     = "live_test_results"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
MODEL_PATH     = "D:/IDscanner/ID detection/runs/detect/runs/id_card/train_v1/weights/best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  CAMERA STATE
# ─────────────────────────────────────────────
mv_handle = None
mv_buffer = None
cap       = None

def start_camera():
    global mv_handle, mv_buffer, cap
    if MVSDK_AVAILABLE:
        try:
            device_list = mvsdk.CameraEnumerateDevice()
            if not device_list:
                print("[Camera] No MindVision camera found, falling back to webcam.")
                start_webcam()
                return
            device_info = device_list[0]
            mv_handle   = mvsdk.CameraInit(device_info, -1, -1)
            mvsdk.CameraReadParameterFromFile(mv_handle, "conf.config")
            print("[Camera] Config loaded from conf.config")
            mvsdk.CameraSetIspOutFormat(mv_handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            capability = mvsdk.CameraGetCapability(mv_handle)
            buf_size   = (capability.sResolutionRange.iWidthMax
                          * capability.sResolutionRange.iHeightMax * 3)
            mv_buffer  = mvsdk.CameraAlignMalloc(buf_size, 16)
            mvsdk.CameraPlay(mv_handle)
            print("[Camera] MindVision camera started.")
        except mvsdk.CameraException as e:
            print("[Camera] SDK error:", e.error_code, e.message)
            start_webcam()
    else:
        start_webcam()

def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[Camera] Webcam started.")

def read_frame():
    global mv_handle, mv_buffer, cap
    if MVSDK_AVAILABLE and mv_handle is not None and mv_buffer is not None:
        try:
            raw_data, frame_head = mvsdk.CameraGetImageBuffer(mv_handle, 200)
            mvsdk.CameraImageProcess(mv_handle, raw_data, mv_buffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(mv_handle, raw_data)
            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(mv_buffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (frame_head.iHeight, frame_head.iWidth, 3))
            return cv2.flip(frame.copy(), 0)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("[Camera] SDK error:", e.error_code, e.message)
            return None
    if cap and cap.isOpened():
        ret, frame = cap.read()
        return frame if ret else None
    return None

def stop_camera():
    global mv_handle, mv_buffer, cap
    if MVSDK_AVAILABLE and mv_handle is not None:
        try:
            mvsdk.CameraStop(mv_handle)
            mvsdk.CameraUnInit(mv_handle)
            mv_handle = None
        except Exception as e:
            print("[Camera] Failed to stop MV camera:", e)
    if mv_buffer is not None:
        try:
            mvsdk.CameraAlignFree(mv_buffer)
            mv_buffer = None
        except Exception as e:
            print("[Camera] Failed to free buffer:", e)
    if cap and cap.isOpened():
        cap.release()
        cap = None


# ─────────────────────────────────────────────
#  RUN INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, frame):
    start   = time.time()
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    elapsed = (time.time() - start) * 1000

    detections   = []
    result_frame = frame.copy()

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})
            color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, f"id_card {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result_frame, detections, elapsed


# ─────────────────────────────────────────────
#  DRAW HUD
# ─────────────────────────────────────────────
def draw_hud(frame, detections, elapsed):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Inference: {elapsed:.1f}ms",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if detections:
        best_conf = max(d["confidence"] for d in detections)
        cv2.putText(frame, f"ID DETECTED — conf: {best_conf:.2f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO ID DETECTED",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "ENTER — save | Q — quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("YOLOv8 Live Camera Test — best.pt")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found: {MODEL_PATH}")
        return

    print("[Model] Loading best.pt...")
    model = YOLO(MODEL_PATH)
    print("[Model] Loaded!\n")

    start_camera()

    print("[Info] Camera running.")
    print("[Info] Place an ID card in frame.")
    print("[Info] Press ENTER to save capture.")
    print("[Info] Press Q to quit.\n")

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue

            display_frame            = cv2.resize(frame, (1280, 960))
            result_frame, detections, elapsed = run_inference(model, display_frame)
            result_frame             = draw_hud(result_frame, detections, elapsed)

            cv2.imshow('YOLOv8 Live Test', result_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:
                save_path = os.path.join(OUTPUT_DIR, f"capture_{int(time.time())}.jpg")
                cv2.imwrite(save_path, result_frame)
                print(f"[Capture] Saved: {save_path}")
                if detections:
                    for d in detections:
                        print(f"         bbox: {d['bbox']} | conf: {d['confidence']:.2f}")
                else:
                    print("[Capture] No detections in this frame")

            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")

    finally:
        print("[Info] Releasing camera...")
        stop_camera()
        cv2.destroyAllWindows()
        print("[Info] Done.")


if __name__ == "__main__":
    main()