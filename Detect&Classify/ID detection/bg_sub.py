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

            # load saved camera config (exposure, white balance, frame rate etc)
            mvsdk.CameraReadParameterFromFile(mv_handle, "conf.config")
            print("[Camera] Config loaded from conf.config")

            # set output format to BGR so OpenCV can use it directly
            mvsdk.CameraSetIspOutFormat(mv_handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            # allocate frame buffer based on max resolution
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

    # use industrial camera if available
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

    # fallback to webcam
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

# ── mask cleanup ──────────────────────────────────────────────────────────────
def clean_mask(mask):
    # step 1 — catch both white (255) and grey shadows (127) as foreground
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    # step 2 — median blur to remove noise speckles
    mask = cv2.medianBlur(mask, 5)

    # step 3 — morphological closing to fill dark holes inside the ID
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # step 4 — morphological opening to remove small stray blobs outside the ID
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return mask


# ── shadow correction (dilate and divide) ────────────────────────────────────
def remove_shadows(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # dilate heavily to fill in the foreground object (ID card)
    # what remains is essentially just the background lighting information
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(gray, kernel, iterations=10)

    # divide original by dilated — normalizes lighting and removes shadows
    # result is a uniformly lit image where background becomes white
    divided = cv2.divide(gray, dilated, scale=255)

    # convert back to BGR for display and background subtractor
    result = cv2.cvtColor(divided, cv2.COLOR_GRAY2BGR)
    return result, divided  # return both BGR and grayscale versions


def extract_foreground(frame, mask):
    # find contours on the cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[Extract] No contours found.")
        return None

    # get the largest contour — most likely the ID card
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # ignore if too small — probably noise
    if area < 5000:
        print("[Extract] Detected region too small, ignoring.")
        return None

    # get bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest)
    print(f"[Extract] Bounding box — x:{x} y:{y} w:{w} h:{h} area:{area:.0f}")

    # crop the ID from the original color frame
    cropped = frame[y:y+h, x:x+w]
    return cropped


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    start_camera()

    # initialize background subtractor
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)

    print("[Info] Camera running. Let it sit for a few seconds to learn the background.")
    print("[Info] Then place an ID card in frame.")
    print("[Info] Press ENTER to extract the foreground.")
    print("[Info] Press Q to quit.")

    current_frame = None
    current_mask  = None

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue

            current_frame = frame.copy()

            # resize for display and processing — 1280x960
            display_frame = cv2.resize(frame, (1280, 960))

            # apply shadow correction (dilate and divide)
            shadow_removed, shadow_gray = remove_shadows(display_frame)

            # apply background subtractor on shadow corrected grayscale
            raw_mask = backSub.apply(shadow_removed)

            # clean up the mask
            current_mask = clean_mask(raw_mask)

            cv2.imshow('Original', display_frame)
            cv2.imshow('Shadow Corrected', shadow_removed)
            cv2.imshow('FG Mask (Cleaned)', current_mask)

            key = cv2.waitKey(1) & 0xFF

            # press Enter to extract
            if key == 13:
                if current_frame is not None and current_mask is not None:
                    cropped = extract_foreground(shadow_removed, current_mask)
                    if cropped is not None:
                        cv2.imshow('Extracted ID', cropped)
                        cv2.imwrite('extracted_id.jpg', cropped)
                        print("[Extract] Saved as extracted_id.jpg")
                    else:
                        print("[Extract] Could not extract — make sure the ID is clearly in frame.")

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