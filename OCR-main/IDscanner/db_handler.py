"""
db_handler.py
-------------
Handles all Supabase operations for the ID Scanner app.

CHANGES FROM PREVIOUS VERSION:
  - FIXED   [lines 14-22]:  _get_client() — return _client was inside the try block
                             so on every call after the first the function returned
                             None; moved return outside the if block
  - FIXED   [line 82]:      _upload_image() — changed content_type to content-type
  - ADDED   [line 64]:      _upload_image() — delete_after parameter; when True
                             the local temp file is deleted after a successful upload
                             so no permanent local copies are kept
"""
import time, os, threading, sys
from typing import Optional
from dotenv import load_dotenv


# --- FIX FOR EXE AND FOLDER PATHING ---
def get_env_path():
    # 1. Check if running as a PyInstaller EXE
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, ".env")

    # 2. For PyCharm: Look in the same folder as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_path = os.path.join(current_dir, ".env")

    if os.path.exists(potential_path):
        return potential_path

    return ".env"  # Fallback to current working directory


env_path = get_env_path()
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "id-scans"

# Define the global client variable
_client = None
# --------------------------------------

if not SUPABASE_URL:
    print(f"[DB] CRITICAL: SUPABASE_URL not found. Searched at: {os.path.abspath(env_path)}")


def init_client_on_main_thread() -> None:
    global _client
    if _client is not None:
        return
    try:
        from supabase import create_client
        # Double check the vars right before init
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing from environment")

        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[DB] Supabase client initialised.")
    except Exception as e:
        print(f"[DB] Failed to initialise Supabase client: {e}")

def get_client():
    # Returns the cached client. Client should already be initialized
    # via _init_client_on_main_thread() at app startup.
    # Falls back to lazy init if called before startup (safe on main thread).
    global _client
    if _client is None:
        init_client_on_main_thread()
    return _client

def save_scan(
    id_type, method, result_text,
    front_path=None, back_path=None, debug_path=None, back_debug_path=None, gradcam_path=None, gradcam_back_path=None
) -> None:
    threading.Thread(
        target=save_scan_worker,
        args=(id_type, method, result_text, front_path, back_path, debug_path, back_debug_path, gradcam_path, gradcam_back_path),
        daemon=True,
    ).start()

def save_scan_worker(
    id_type, method, result_text, front_path, back_path, debug_path, back_debug_path, gradcam_path, gradcam_back_path
) -> None:
    try:
        client=get_client()
        if client is None:
            print("[DB] Skipping save — Supabase client not available.")
            return

        ts = int(time.time())

        front_url = upload_image(client, front_path, ts, "front", method=method, delete_after=True) if front_path else None
        back_url = upload_image(client, back_path, ts, "back", method=method, delete_after=True) if back_path else None
        debug_url = upload_image(client, debug_path, ts, "debug", method=method, delete_after=True) if debug_path else None
        back_debug_url = upload_image(client, back_debug_path, ts, "debug_back", method=method,delete_after=True) if back_debug_path else None
        gradcam_url = upload_image(client, gradcam_path, ts, "gradcam", method=method,delete_after=False) if gradcam_path else None
        gradcam_back_url = upload_image(client, gradcam_back_path, ts, "gradcam_back", method=method,delete_after=False) if gradcam_back_path else None
        record = {
            "id_type":     id_type,
            "method":      method,
            "front_url":   front_url,
            "back_url":    back_url,
            "debug_url":   debug_url,
            "back_debug_url": back_debug_url,
            "result_text": result_text,
            "gradcam_url": gradcam_url,
            "gradcam_back_url": gradcam_back_url,
        }

        response = client.table("scans").insert(record).execute()
        print(f"[DB] Scan record saved. id={response.data[0].get('id') if response.data else '?'}")
    except Exception as e:
        print(f"[DB] Error saving scan: {e}")

def upload_image(
    client,
    local_path: str,
    timestamp: int,
    label: str,
    method: str = "Upload",
    delete_after: bool = False,
) -> Optional[str]:
    if not local_path or not os.path.exists(local_path):
        print(f"[DB] Skipping upload — file not found: {local_path}")
        return None

    try:
        filename     = os.path.basename(local_path)
        folder_map = {
            "Camera": "Camera",
            "Upload": "Upload",
            "PDF": "PDF",
        }
        debug_folder_map = {
            "Camera": "Camera_debug",
            "Upload": "Upload_debug",
            "PDF": "PDF_debug",
        }
        gradcam_folder_map = {
            "Camera": "Camera_grad",
            "Upload": "Upload_grad",
            "PDF": "PDF_grad",
        }
        method_folder = folder_map.get(method, "Upload")
        debug_method_folder = debug_folder_map.get(method, "Upload_debug")
        gradcam_method_folder = gradcam_folder_map.get(method, "Upload_grad")

        if label in ("debug", "debug_back"):
            storage_path = f"Debug/{debug_method_folder}/{timestamp}_{label}_{filename}"
        elif label in ("gradcam", "gradcam_back"):
            ext = os.path.splitext(filename)[1].lower() or ".jpg"
            storage_path = f"Grad-CAM/{gradcam_method_folder}/{timestamp}_{label}{ext}"
        else:
            storage_path = f"{method_folder}/{timestamp}_{label}_{filename}"

        with open(local_path, "rb") as f:
            file_bytes = f.read()

        ext  = os.path.splitext(filename)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        client.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": mime},
        )

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{storage_path}"
        )
        print(f"[DB] Uploaded {label}: {public_url}")

        # Delete the temp file after successful upload
        if delete_after:
            try:
                os.remove(local_path)
                print(f"[DB] Deleted temp file: {local_path}")
            except Exception as e:
                print(f"[DB] Could not delete temp file {local_path}: {e}")

        return public_url

    except Exception as e:
        print(f"[DB] Failed to upload {label} image ({local_path}): {e}")
        return None