import cv2
import numpy as np
import sys
import os

# add parent directory to path so mvsdk can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mvsdk
    MVSDK_AVAILABLE = True
    print("[Camera] MindVision SDK found.")
except Exception as e:
    MVSDK_AVAILABLE = False
    print("[Camera] MindVision SDK not found, falling back to webcam:", e)

# ── camera state ──────────────────────────────────────────────────────────────
mv_handle = None
mv_buffer = None
cap = None

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
            mv_handle = mvsdk.CameraInit(device_info, -1, -1)

            mvsdk.CameraReadParameterFromFile(mv_handle, "conf.config")
            print("[Camera] Config loaded from conf.config")

            mvsdk.CameraSetIspOutFormat(mv_handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            capability = mvsdk.CameraGetCapability(mv_handle)
            buf_size = (
                capability.sResolutionRange.iWidthMax
                * capability.sResolutionRange.iHeightMax
                * 3
            )
            mv_buffer = mvsdk.CameraAlignMalloc(buf_size, 16)
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
                (frame_head.iHeight, frame_head.iWidth, 3)
            )
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

# ── canny edge detection ──────────────────────────────────────────────────────
def get_edges(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE — boosts local contrast independently per tile
    # helps in dim rooms by making ID border edges stronger before filtering
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # bilateral filter — preserves strong edges (ID border) while smoothing
    # internal noise (holographic patterns, text, background texture)
    # d=9, sigmaColor=75, sigmaSpace=75 is a good starting point
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # adaptive canny thresholds based on median pixel brightness
    # works automatically in dim or bright rooms
    median = np.median(filtered)
    sigma = 0.33
    low  = int(max(0,   (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    print(f"[Canny] Room brightness median: {median:.0f} | Low: {low} | High: {high}")

    edges = cv2.Canny(filtered, low, high)

    # morphological closing — connects broken border lines
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return edges


# ── rectangle detection ───────────────────────────────────────────────────────

# ideal CR80 ID card aspect ratio
CR80_RATIO = 1.586

def aspect_ratio_score(ratio):
    # score how close the aspect ratio is to ideal CR80 ratio
    # returns 0.0 to 1.0 where 1.0 is a perfect match
    portrait_ratio = 1.0 / CR80_RATIO  # 0.630 for portrait orientation
    landscape_score = 1.0 - abs(ratio - CR80_RATIO) / CR80_RATIO
    portrait_score  = 1.0 - abs(ratio - portrait_ratio) / portrait_ratio
    return max(landscape_score, portrait_score)

def find_id_rectangle(frame, edges):
    # find all contours from the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # sort contours by area largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # frame area used for max area threshold
    frame_area = frame.shape[0] * frame.shape[1]

    best_contour = None
    best_score   = -1

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon   = 0.04 * perimeter
        approx    = cv2.approxPolyDP(cnt, epsilon, True)
        area      = cv2.contourArea(cnt)

        # min area — ignore tiny noise contours
        # max area — ignore contours that are 80%+ of the frame (too large to be an ID)
        if area < 10000 or area > frame_area * 0.80:
            continue

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h
        score = aspect_ratio_score(aspect_ratio)

        print(f"[Contour] Area: {area:.0f} | Ratio: {aspect_ratio:.2f} | Score: {score:.2f}")

        # only consider contours with a reasonable aspect ratio score
        if score > 0.75 and score > best_score:
            best_score   = score
            best_contour = approx

    if best_contour is None:
        return None, None

    # draw the detected rectangle on a copy of the frame
    result_frame = frame.copy()
    cv2.drawContours(result_frame, [best_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(best_contour)
    cv2.putText(result_frame, f"ID Detected (score: {best_score:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result_frame, best_contour


# ── order corners for perspective transform ───────────────────────────────────
def order_corners(pts):
    # reshape to 4x2 array
    pts = pts.reshape(4, 2).astype(np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right

    # top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


# ── perspective transform crop ────────────────────────────────────────────────
def perspective_crop(frame, contour):
    # order the 4 corners correctly
    corners = order_corners(contour)
    tl, tr, br, bl = corners

    # calculate width of the output — take the max of top and bottom widths
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    output_width = int(max(width_top, width_bottom))

    # calculate height of the output — take the max of left and right heights
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    output_height = int(max(height_left, height_right))

    # standardized CR80 ID card output size at 100 DPI
    # ensures consistent dimensions for OCR pipeline regardless of camera distance
    OUTPUT_WIDTH  = 856
    OUTPUT_HEIGHT = 540

    # define destination rectangle corners at standard size
    dst = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1]
    ], dtype=np.float32)

    # calculate perspective transform matrix and apply warp
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    return warped


# ── post processing ───────────────────────────────────────────────────────────
def post_process(image):
    # step 1 — denoise: lower h value to avoid over-smoothing
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    # step 2 — illumination normalization in LAB color space
    # LAB separates lightness (L) from color (A, B channels)
    # so we only normalize the lightness and color is fully preserved
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(l, kernel, iterations=7)
    l_normalized = cv2.divide(l, dilated, scale=255)
    lab_normalized = cv2.merge([l_normalized, a, b])
    denoised = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    # step 3 — sharpening using unsharp mask
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2)
    sharpened = cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)

    return sharpened


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    start_camera()

    print("[Info] Camera running.")
    print("[Info] Place an ID card on a white surface.")
    print("[Info] Press ENTER to capture the detected ID.")
    print("[Info] Press Q to quit.")

    current_edges    = None
    current_contour  = None
    current_frame    = None

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue

            # resize for display and processing
            display_frame = cv2.resize(frame, (1280, 960))
            current_frame = display_frame.copy()

            # get edges
            current_edges = get_edges(display_frame)

            # find ID rectangle
            result_frame, current_contour = find_id_rectangle(display_frame, current_edges)

            # show windows
            cv2.imshow('Original', display_frame)
            cv2.imshow('Canny Edges', current_edges)

            if result_frame is not None:
                cv2.imshow('Rectangle Detection', result_frame)
            else:
                # show original with no detection text if no rectangle found
                no_detect = display_frame.copy()
                cv2.putText(no_detect, "No ID Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Rectangle Detection', no_detect)

            key = cv2.waitKey(1) & 0xFF

            # press Enter to crop and save
            if key == 13:
                if current_contour is not None:
                    warped = perspective_crop(current_frame, current_contour)
                    processed = post_process(warped)
                    cv2.imshow('Perspective Crop (Raw)', warped)
                    cv2.imshow('Perspective Crop (Processed)', processed)
                    cv2.imwrite('canny_extracted_id.jpg', processed)
                    print("[Extract] Saved as canny_extracted_id.jpg")
                else:
                    print("[Extract] No ID rectangle detected — make sure the ID is clearly in frame.")

            # press Q to quit
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")

    finally:
        print("[Info] Releasing camera...")
        stop_camera()
        cv2.destroyAllWindows()
        print("[Info] Camera released.")

if __name__ == "__main__":
    main()