"""
id_detect_classify.py
─────────────────────────────────────────────────────────────
Standalone integration of:
  - canny.py      → ID detection, crop, post processing
  - heatmap_mob.py → ID type classification via MobileNetV3

Flow:
  Camera feed → Detect ID rectangle → Perspective crop
  → Post process → Save temp file → Classify ID type
  → Show result with Grad-CAM heatmap

Controls:
  ENTER — manually trigger detection and classification
  Q     — quit
─────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── pytorch / model imports ───────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── mvsdk ─────────────────────────────────────────────────────────────────────
try:
    import mvsdk
    MVSDK_AVAILABLE = True
    print("[Camera] MindVision SDK found.")
except Exception as e:
    MVSDK_AVAILABLE = False
    print("[Camera] MindVision SDK not found, falling back to webcam:", e)


# ─────────────────────────────────────────────
#  MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Renzo\Documents\MindVision\models\mobilenetv3\mobilenet_best.pth"
OUTPUT_DIR  = r"C:\Users\Renzo\Documents\MindVision\gradcam_results"
CLASS_NAMES = ["driver's license", "passport", "philhealth", "philid", "senior", "sss"]
IMG_SIZE    = (224, 224)
TEMP_PATH   = "temp_id_crop.jpg"  # temp file bridge between detection and classification

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Model] Device: {device}")

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    print("[Model] Loading MobileNetV3...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    print("[Model] Model loaded!")
    return model


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]
        for i, grad in enumerate(pooled_grads):
            activations[i] *= grad
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap, class_idx, output


# ─────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────
def classify_id(model, gradcam, image_path):
    """Takes an image path, returns class_name, confidence and visualization."""
    filename     = os.path.basename(image_path)
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    heatmap, class_idx, output = gradcam.generate(input_tensor)
    probs      = torch.softmax(output, dim=1)[0]
    confidence = probs[class_idx].item()
    class_name = CLASS_NAMES[class_idx]

    # build heatmap overlay
    img_w, img_h     = original_img.size
    heatmap_resized  = cv2.resize(heatmap, (img_w, img_h))
    heatmap_colored  = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_bgr          = cv2.cvtColor(np.array(original_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay          = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # print breakdown
    print(f"\n[Classify] Result: {class_name.upper()} ({confidence:.2%})")
    print(f"[Classify] Breakdown:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {name:15s} {probs[i].item():.2%} {bar}")

    return class_name, confidence, overlay, heatmap_colored, original_img


def build_result_display(original_img, overlay, heatmap_colored, class_name, confidence):
    """Builds a display image showing original, heatmap and overlay side by side."""
    img_w, img_h = original_img.size
    target_h = 400
    scale    = target_h / img_h
    target_w = int(img_w * scale)

    def add_label(img, label):
        img = img.copy()
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        return img

    orig_panel    = cv2.resize(cv2.cvtColor(np.array(original_img.convert("RGB")), cv2.COLOR_RGB2BGR), (target_w, target_h))
    heatmap_panel = cv2.resize(heatmap_colored, (target_w, target_h))
    overlay_panel = cv2.resize(overlay, (target_w, target_h))

    orig_panel    = add_label(orig_panel,    "Cropped ID")
    heatmap_panel = add_label(heatmap_panel, "Attention Map")
    overlay_panel = add_label(overlay_panel, f"{class_name.upper()} ({confidence:.1%})")

    combined  = np.hstack([orig_panel, heatmap_panel, overlay_panel])
    title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"ID Type: {class_name.upper()}  |  Confidence: {confidence:.1%}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    return np.vstack([title_bar, combined])


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
#  CANNY DETECTION PIPELINE
# ─────────────────────────────────────────────
CR80_RATIO = 1.586

def aspect_ratio_score(ratio):
    portrait_ratio  = 1.0 / CR80_RATIO
    landscape_score = 1.0 - abs(ratio - CR80_RATIO) / CR80_RATIO
    portrait_score  = 1.0 - abs(ratio - portrait_ratio) / portrait_ratio
    return max(landscape_score, portrait_score)

def get_edges(frame):
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray     = clahe.apply(gray)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    median   = np.median(filtered)
    sigma    = 0.33
    low      = int(max(0,   (1.0 - sigma) * median))
    high     = int(min(255, (1.0 + sigma) * median))
    edges    = cv2.Canny(filtered, low, high)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges

def find_id_rectangle(frame, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contours   = sorted(contours, key=cv2.contourArea, reverse=True)
    frame_area = frame.shape[0] * frame.shape[1]
    best_contour = None
    best_score   = -1
    for cnt in contours:
        perimeter    = cv2.arcLength(cnt, True)
        epsilon      = 0.04 * perimeter
        approx       = cv2.approxPolyDP(cnt, epsilon, True)
        area         = cv2.contourArea(cnt)
        if area < 10000 or area > frame_area * 0.80:
            continue
        if len(approx) != 4:
            continue
        x, y, w, h   = cv2.boundingRect(approx)
        aspect_ratio = w / h
        score        = aspect_ratio_score(aspect_ratio)
        if score > 0.75 and score > best_score:
            best_score   = score
            best_contour = approx
    if best_contour is None:
        return None, None
    result_frame = frame.copy()
    cv2.drawContours(result_frame, [best_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(best_contour)
    cv2.putText(result_frame, f"ID Detected (score: {best_score:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result_frame, best_contour

def order_corners(pts):
    pts     = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff       = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def perspective_crop(frame, contour):
    corners      = order_corners(contour)
    tl, tr, br, bl = corners
    OUTPUT_WIDTH  = 856
    OUTPUT_HEIGHT = 540
    dst = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return warped

def post_process(image):
    # denoise — removes camera grain while keeping edges sharp
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    # sharpening using unsharp mask
    blurred  = cv2.GaussianBlur(denoised, (0, 0), 2)
    return cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # load model first before starting camera
    model   = load_model()
    gradcam = GradCAM(model)

    start_camera()

    print("\n[Info] Camera running.")
    print("[Info] Place an ID card in frame.")
    print("[Info] Press ENTER to detect, crop and classify.")
    print("[Info] Press Q to quit.\n")

    current_frame   = None
    current_contour = None

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue

            display_frame   = cv2.resize(frame, (1280, 960))
            current_frame   = display_frame.copy()
            edges           = get_edges(display_frame)

            # show edge mask window
            cv2.imshow('Edge Mask', edges)

            result_frame, current_contour = find_id_rectangle(display_frame, edges)

            # show detection window
            if result_frame is not None:
                cv2.imshow('ID Detection', result_frame)
            else:
                no_detect = display_frame.copy()
                cv2.putText(no_detect, "No ID Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('ID Detection', no_detect)

            key = cv2.waitKey(1) & 0xFF

            # press Enter to crop and classify
            if key == 13:
                if current_contour is not None:
                    print("\n[Pipeline] ID detected — cropping...")
                    warped    = perspective_crop(current_frame, current_contour)
                    processed = post_process(warped)

                    # save temp file as bridge to classifier
                    cv2.imwrite(TEMP_PATH, processed)
                    print(f"[Pipeline] Crop saved to temp file: {TEMP_PATH}")

                    # classify
                    print("[Pipeline] Sending to classifier...")
                    class_name, confidence, overlay, heatmap_colored, original_img = \
                        classify_id(model, gradcam, TEMP_PATH)

                    # build and show result display
                    result_display = build_result_display(
                        original_img, overlay, heatmap_colored,
                        class_name, confidence
                    )
                    cv2.imshow('Classification Result', result_display)

                    # save result
                    save_name = f"result_{int(time.time())}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    cv2.imwrite(save_path, result_display)
                    print(f"[Pipeline] Result saved to: {save_path}")

                else:
                    print("[Pipeline] No ID detected — make sure the ID is clearly in frame.")

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")

    finally:
        print("[Info] Releasing camera...")
        stop_camera()
        cv2.destroyAllWindows()
        # clean up temp file
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)
        print("[Info] Done.")

if __name__ == "__main__":
    main()