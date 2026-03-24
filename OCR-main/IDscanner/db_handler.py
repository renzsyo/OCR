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
import time, os, threading
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "id-scans"

_client = None

def init_client_on_main_thread() -> None:
    """
    Call this once from the main thread at startup (before any background
    threads run) so the Supabase/httpx/asyncio initialization never races
    with PyTorch or PaddleOCR CUDA context on a background thread.
    On Windows, initializing asyncio-based libraries from a worker thread
    while CUDA is active can corrupt the process (0xC0000409).
    """
    global _client
    if _client is not None:
        return
    try:
        from supabase import create_client
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
    id_type:     str,
    method:      str,
    result_text: str,
    front_path:  Optional[str] = None,
    back_path:   Optional[str] = None,
    debug_path:  Optional[str] = None,
) -> None:

    threading.Thread(
        target=save_scan_worker,
        args=(id_type, method, result_text, front_path, back_path, debug_path),
        daemon=True,
    ).start()

def save_scan_worker(
    id_type, method, result_text, front_path, back_path, debug_path
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

        record = {
            "id_type":     id_type,
            "method":      method,
            "front_url":   front_url,
            "back_url":    back_url,
            "debug_url":   debug_url,
            "result_text": result_text,
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
        storage_folder = folder_map.get(method, "Upload")
        storage_path = f"{storage_folder}/{timestamp}_{label}_{filename}"

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