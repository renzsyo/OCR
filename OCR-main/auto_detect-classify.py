"""
=============================================================
  ID Scanner — Integrated Pipeline (Auto-Capture)
  ─────────────────────────────────────────────────────────
  Two YOLO models working together:
    • YOLO DETECTOR   → finds and crops the ID card in frame
    • YOLO CLASSIFIER → identifies which type of ID it is
    • Grad-CAM        → shows what the classifier looked at

  Auto-capture logic:
    • ID must be detected with high confidence AND held
      steady for 1.5 seconds to trigger automatically
    • A lock-on progress bar fills on the HUD as it counts
    • After capture, scanner re-arms only once the ID
      fully leaves the frame
    • ENTER still works at any time as a manual fallback

  Controls:
    ENTER  → manual capture immediately
    Q      → quit

  Output layout (saved JPG):
  ┌─────────────────────────────────────────────────────┐
  │  Title: ID Type | Confidence | Detection conf       │
  ├──────────────┬──────────────┬────────────────────────┤
  │  Full frame  │  ID crop     │  Grad-CAM overlay      │
  │  (YOLO box)  │  (original)  │  (attention heatmap)   │
  ├──────────────┴──────────────┴────────────────────────┤
  │  Confidence bar chart (all 6 classes)               │
  └─────────────────────────────────────────────────────┘
=============================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import torch
from torchvision import transforms
from PIL import Image
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
#  CONFIG
# ─────────────────────────────────────────────
YOLO_DETECTOR_PATH   = "D:/IDscanner/ID detection/runs/detect/runs/id_card/train_v1/weights/best.pt"
YOLO_CLASSIFIER_PATH = r"C:\Users\Renzo\Documents\MindVision\models\yolo_v4\yolo_final\weights\classifybest.pt"
OUTPUT_DIR           = "live_test_results"
CLASS_NAMES          = ["drivers_license", "passport", "philhealth", "philid", "senior", "sss","tin"]

# Detection thresholds
DETECT_CONF  = 0.25
DETECT_IOU   = 0.45

# Classification confidence gate — below this → "Uncertain"
CLASSIFY_CONF_THRESHOLD = 0.80

# ── Auto-capture tuning ──────────────────────
STEADY_SECONDS     = 1.5    # how long the ID must stay still before triggering
STEADY_CONF_MIN    = 0.50   # minimum detector confidence to start the countdown
STEADY_IOU_MIN     = 0.75   # how much the box must overlap frame-to-frame (stability)
GONE_FRAMES_NEEDED = 8      # consecutive frames with no detection before re-arming

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")


# ─────────────────────────────────────────────
#  CAMERA
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
            mvsdk.CameraReadParameterFromFile(mv_handle, "new_config.config")
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
#  BOX HELPERS
# ─────────────────────────────────────────────
def box_iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes. Used to measure stability."""
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


# ─────────────────────────────────────────────
#  YOLO DETECTOR
# ─────────────────────────────────────────────
def run_detector(detector, frame):
    start   = time.time()
    results = detector(frame, conf=DETECT_CONF, iou=DETECT_IOU, verbose=False)
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
#  YOLO CLASSIFIER
# ─────────────────────────────────────────────
def classify_crop(classifier, crop_bgr):
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_crop.jpg")
    cv2.imwrite(tmp_path, crop_bgr)

    result     = classifier(tmp_path, verbose=False)[0]
    probs      = result.probs.data.cpu().numpy()
    class_idx  = int(result.probs.top1)
    confidence = float(probs[class_idx])
    class_name = CLASS_NAMES[class_idx] if confidence >= CLASSIFY_CONF_THRESHOLD else "Uncertain"

    return class_name, confidence, probs, class_idx


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class YOLOGradCAM:
    def __init__(self, classifier):
        self.model       = classifier.model
        self.gradients   = None
        self.activations = None
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _get_target_layer(self):
        layers = list(self.model.model.children())
        for layer in reversed(layers):
            if hasattr(layer, 'conv'):
                return layer.conv
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        return list(self.model.modules())[-3]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        self.model.eval()
        output = self.model(img_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        score = output[0, class_idx]
        score.backward()
        if self.gradients is None or self.activations is None:
            return None
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]
        for i, grad in enumerate(pooled_grads):
            activations[i] *= grad
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


_gradcam_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def build_gradcam_overlay(gradcam, crop_bgr, class_idx):
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)
        tensor   = _gradcam_transform(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensor.requires_grad_(True)
        heatmap = gradcam.generate(tensor, class_idx)
        if heatmap is None:
            return crop_bgr.copy(), False
        h, w            = crop_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop_bgr, 0.6, heatmap_colored, 0.4, 0)
        return overlay, True
    except Exception as e:
        print(f"[GradCAM] Warning — {e} (showing plain crop instead)")
        return crop_bgr.copy(), False


# ─────────────────────────────────────────────
#  HUD  (with lock-on progress bar)
# ─────────────────────────────────────────────
# Scanner states
STATE_READY    = "READY"     # waiting for an ID to appear
STATE_LOCKING  = "LOCKING"   # ID detected, counting down
STATE_CAPTURED = "CAPTURED"  # just fired — waiting for ID to leave

def draw_hud(frame, detections, elapsed, state, lock_progress):
    """
    lock_progress: 0.0 → 1.0, only shown during LOCKING state.
    """
    h, w = frame.shape[:2]

    # ── Inference time ──
    cv2.putText(frame, f"Detect: {elapsed:.1f}ms",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # ── Detection status ──
    if detections:
        best_conf = max(d["confidence"] for d in detections)
        cv2.putText(frame, f"ID DETECTED  conf: {best_conf:.2f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO ID DETECTED",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # ── State label ──
    state_colors = {
        STATE_READY:    (180, 180, 180),
        STATE_LOCKING:  (0, 220, 255),
        STATE_CAPTURED: (0, 255, 100),
    }
    state_labels = {
        STATE_READY:    "READY — hold ID steady to scan",
        STATE_LOCKING:  "LOCKING ON...",
        STATE_CAPTURED: "CAPTURED — remove ID to scan again",
    }
    cv2.putText(frame, state_labels[state],
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                state_colors[state], 2)

    # ── Lock-on progress bar (only during LOCKING) ──
    if state == STATE_LOCKING and lock_progress > 0:
        bar_x, bar_y = 10, 118
        bar_w, bar_h = w - 20, 18
        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # Fill — colour shifts green→yellow as it fills
        fill_w  = int(bar_w * lock_progress)
        r_val   = int(255 * (1 - lock_progress))
        g_val   = 220
        bar_col = (0, g_val, r_val)          # BGR
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        # Percentage text inside bar
        pct_text = f"{int(lock_progress * 100)}%"
        cv2.putText(frame, pct_text,
                    (bar_x + bar_w // 2 - 15, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Bottom hint ──
    cv2.putText(frame, "ENTER — manual capture | Q — quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    return frame


# ─────────────────────────────────────────────
#  BUILD RESULT IMAGE
# ─────────────────────────────────────────────
def build_result_image(full_frame, detection, class_name, confidence,
                        gradcam_overlay, probs, gradcam_ok):
    TARGET_H = 400

    x1, y1, x2, y2 = detection["bbox"]
    detect_conf     = detection["confidence"]

    panel1 = full_frame.copy()
    cv2.rectangle(panel1, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(panel1, f"DETECT {detect_conf:.2f}",
                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    panel2 = full_frame[y1:y2, x1:x2].copy()
    panel3 = gradcam_overlay.copy()

    def resize_to_height(img, h):
        if img is None or img.size == 0:
            return np.zeros((h, h, 3), dtype=np.uint8)
        scale = h / max(img.shape[0], 1)
        return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))

    p1 = resize_to_height(panel1, TARGET_H)
    p2 = resize_to_height(panel2, TARGET_H)
    p3 = resize_to_height(panel3, TARGET_H)

    def add_label(img, top_text, sub_text=""):
        img = img.copy()
        for color, th in [((0, 0, 0), 3), ((255, 255, 255), 2)]:
            cv2.putText(img, top_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, th)
        if sub_text:
            for color, th in [((0, 0, 0), 3), ((0, 255, 200), 2)]:
                cv2.putText(img, sub_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, th)
        return img

    p1 = add_label(p1, "Detection",   f"conf: {detect_conf:.2%}")
    p2 = add_label(p2, "ID Crop")
    p3 = add_label(p3,
                   "Grad-CAM" if gradcam_ok else "Classifier",
                   f"{class_name.upper()} {confidence:.1%}")

    top_row = np.hstack([p1, p2, p3])

    title_bar   = np.zeros((55, top_row.shape[1], 3), dtype=np.uint8)
    color_title = (0, 255, 200) if class_name != "Uncertain" else (0, 165, 255)
    cv2.putText(title_bar,
                f"ID Type: {class_name.upper()}  |  "
                f"Classifier: {confidence:.1%}  |  "
                f"Detector: {detect_conf:.2%}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color_title, 2)

    bar_h     = 130
    bar_panel = np.zeros((bar_h, top_row.shape[1], 3), dtype=np.uint8)
    slot_w    = top_row.shape[1] // len(CLASS_NAMES)
    for i, name in enumerate(CLASS_NAMES):
        prob    = float(probs[i])
        x_s     = i * slot_w
        x_e     = x_s + slot_w - 4
        bar_top = bar_h - 35 - int(prob * (bar_h - 50))
        color   = (0, 220, 0) if name == class_name else (80, 80, 200)
        cv2.rectangle(bar_panel, (x_s + 4, bar_top), (x_e, bar_h - 35), color, -1)
        cv2.putText(bar_panel, f"{prob:.0%}",
                    (x_s + 4, max(bar_top - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(bar_panel, name[:9],
                    (x_s + 4, bar_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return np.vstack([title_bar, top_row, bar_panel])


# ─────────────────────────────────────────────
#  CAPTURE PIPELINE  (shared by auto + manual)
# ─────────────────────────────────────────────
def do_capture(display_frame, best_detection, classifier, gradcam):
    """Run classify + Grad-CAM on the best detection and save result."""
    x1, y1, x2, y2 = best_detection["bbox"]
    h_f, w_f        = display_frame.shape[:2]
    x1, y1          = max(0, x1), max(0, y1)
    x2, y2          = min(w_f, x2), min(h_f, y2)
    crop             = display_frame[y1:y2, x1:x2]

    if crop.size == 0:
        print("[Capture] Crop is empty — skipping.")
        return None

    print("[Classify] Running YOLO classifier...")
    class_name, confidence, probs, class_idx = classify_crop(classifier, crop)

    print("[GradCAM]  Generating heatmap...")
    gradcam_overlay, gradcam_ok = build_gradcam_overlay(gradcam, crop, class_idx)

    print(f"\n[Result]  ID Type    : {class_name.upper()}")
    print(f"[Result]  Confidence : {confidence:.2%}")
    print(f"[Result]  Detector   : {best_detection['confidence']:.2%}")
    print("[Result]  Breakdown  :")
    for i, name in enumerate(CLASS_NAMES):
        bar    = "█" * int(probs[i] * 30)
        marker = " <" if name == class_name else ""
        print(f"           {name:16s} {probs[i]:.2%} {bar}{marker}")

    combined  = build_result_image(
        display_frame, best_detection,
        class_name, confidence,
        gradcam_overlay, probs, gradcam_ok
    )
    timestamp = int(time.time())
    save_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.jpg")
    cv2.imwrite(save_path, combined)
    print(f"\n[Saved]   {save_path}\n")

    cv2.imshow("Capture Result", combined)
    cv2.waitKey(1)
    return combined


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ID Scanner — Auto-Capture + YOLO Classify + Grad-CAM")
    print("=" * 55)

    for label, path in [("Detector",   YOLO_DETECTOR_PATH),
                         ("Classifier", YOLO_CLASSIFIER_PATH)]:
        if not os.path.exists(path):
            print(f"[Error] {label} model not found: {path}")
            return

    print("[Detector]   Loading...")
    detector = YOLO(YOLO_DETECTOR_PATH)
    print("[Detector]   Loaded!")

    print("[Classifier] Loading...")
    classifier = YOLO(YOLO_CLASSIFIER_PATH)
    print("[Classifier] Loaded!")

    print("[Grad-CAM]   Attaching hooks...")
    gradcam = YOLOGradCAM(classifier)
    print("[Grad-CAM]   Ready!\n")

    start_camera()
    print("[Info] Camera running. Hold an ID card steady for 1.5s to auto-scan.")
    print("[Info] Press ENTER for manual capture. Press Q to quit.\n")

    # ── Auto-capture state ──────────────────────
    state          = STATE_READY
    lock_start     = None       # time.time() when lock-on began
    lock_box       = None       # bbox that started the countdown
    gone_counter   = 0          # frames with no detection (for re-arm)

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue

            display_frame          = cv2.resize(frame, (1280, 960))
            result_frame, detections, elapsed = run_detector(detector, display_frame)

            # ── Best detection this frame ──
            best = (max(detections, key=lambda d: d["confidence"])
                    if detections else None)

            # ── State machine ───────────────────
            trigger_capture = False

            if state == STATE_READY:
                if best and best["confidence"] >= STEADY_CONF_MIN:
                    # Start lock-on
                    state      = STATE_LOCKING
                    lock_start = time.time()
                    lock_box   = best["bbox"]
                    gone_counter = 0
                    print("[Lock-on] Started countdown...")

            elif state == STATE_LOCKING:
                if not best or best["confidence"] < STEADY_CONF_MIN:
                    # Lost the ID — reset
                    print("[Lock-on] ID lost, resetting.")
                    state      = STATE_READY
                    lock_start = None
                    lock_box   = None
                else:
                    # Check stability: new box must overlap well with starting box
                    iou = box_iou(lock_box, best["bbox"])
                    if iou < STEADY_IOU_MIN:
                        # Moved too much — restart countdown from new position
                        lock_start = time.time()
                        lock_box   = best["bbox"]
                        print("[Lock-on] Movement detected, restarting countdown.")
                    else:
                        elapsed_lock = time.time() - lock_start
                        if elapsed_lock >= STEADY_SECONDS:
                            # Held steady long enough — fire!
                            trigger_capture = True
                            state           = STATE_CAPTURED
                            print("[Lock-on] Stable — triggering auto-capture!")

            elif state == STATE_CAPTURED:
                if not detections:
                    gone_counter += 1
                    if gone_counter >= GONE_FRAMES_NEEDED:
                        # ID has left the frame — re-arm
                        state        = STATE_READY
                        lock_start   = None
                        lock_box     = None
                        gone_counter = 0
                        print("[Scanner] Re-armed. Ready for next ID.\n")
                else:
                    gone_counter = 0   # ID still in frame, keep waiting

            # ── Compute progress bar fill ──
            lock_progress = 0.0
            if state == STATE_LOCKING and lock_start is not None:
                lock_progress = min((time.time() - lock_start) / STEADY_SECONDS, 1.0)

            # ── Draw HUD ──
            result_frame = draw_hud(result_frame, detections, elapsed,
                                    state, lock_progress)
            cv2.imshow("ID Scanner Live", result_frame)

            # ── Handle capture (auto or manual ENTER) ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            manual = (key == 13)
            if manual and not detections:
                print("[Manual] No ID detected — try again.")
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0),
                              (flash.shape[1]-1, flash.shape[0]-1),
                              (0, 0, 255), 8)
                cv2.putText(flash, "NO ID DETECTED — try again",
                            (50, flash.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("ID Scanner Live", flash)
                cv2.waitKey(800)
                continue

            if trigger_capture or manual:
                capture_target = best if best else (
                    max(detections, key=lambda d: d["confidence"])
                    if detections else None)
                if capture_target:
                    source = "AUTO" if trigger_capture else "MANUAL"
                    print(f"\n[Capture] {source} capture triggered.")
                    do_capture(display_frame, capture_target, classifier, gradcam)
                    if manual:
                        # Manual capture also puts us into CAPTURED state
                        state        = STATE_CAPTURED
                        gone_counter = 0

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")

    finally:
        tmp = os.path.join(OUTPUT_DIR, "_tmp_crop.jpg")
        if os.path.exists(tmp):
            os.remove(tmp)
        print("[Info] Releasing camera...")
        stop_camera()
        cv2.destroyAllWindows()
        print("[Info] Done.")


if __name__ == "__main__":
    main()