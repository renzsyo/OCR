"""
file_handler.py
---------------
CHANGES FROM PREVIOUS VERSION:
  - REMOVED [n/a]:          uploaded_files list, current_index, refresh_file_list,
                             list_item_clicked, on_current_row_changed, show_list_menu,
                             delete_selected_file, display_file_details
                             (fileListWidget no longer exists on upload page)
  - CHANGED [lines 84-86]:  FileManager.__init__() — removed uploaded_files and
                             current_index attributes
  - CHANGED [lines 120-142]:handle_image_upload() — now saves to a temp file in
                             IDscanner/output/temp/ instead of a permanent local
                             copy; temp file is deleted by db_handler after upload
                             to Supabase; deployment-safe since original file path
                             is only read, never stored permanently
  - CHANGED [lines 91-113]: upload_image() — id_type now read from detected_id_type
                             instead of idOption dropdown
  - CHANGED [lines 205-228]:finalise_upload() — side=None now stores to p.front_file
                             (same as side='front'); no more uploaded_files list
  - CHANGED [lines 230-253]:trigger_inference() — side=None triggers infer_front_upload();
                             side front/back uses detected_id_type for routing
  - KEPT    [lines 27-82]:  all module-level helpers unchanged
  - KEPT    [lines 127-204]:all PDF handling and crop dialog logic unchanged
"""

import os, shutil, cv2
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow


# ---------------------------------------------------------------------------
# Module-level helpers — UNCHANGED
# ---------------------------------------------------------------------------

def convert_pdf_pages(pdf_path: str, dpi: int = 200) -> list[np.ndarray]:
    try:
        from pdf2image import convert_from_path
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        frames = []
        for page in pil_pages:
            arr = np.array(page)
            frames.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f"[convert_pdf_pages] {len(frames)} page(s) converted.")
        return frames
    except Exception as e:
        print(f"[convert_pdf_pages] Conversion failed: {e}")
        return []


def save_image(image: np.ndarray, folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    return path


def make_file_info(path: str, side: str | None) -> dict:
    return {
        "path":   path,
        "name":   os.path.basename(path),
        "size":   f"{os.path.getsize(path) / (1024 * 1024):.2f} MB",
        "status": "Completed",
        "side":   side,
    }


def display_frame_on_label(frame: np.ndarray, label: QLabel) -> None:
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    except Exception as e:
        print(f"[display_frame_on_label] Failed: {e}")


# ---------------------------------------------------------------------------
# FileManager
# ---------------------------------------------------------------------------

class FileManager:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent

    # ------------------------------------------------------------------
    # Upload entry point
    # ------------------------------------------------------------------

    def upload_image(self, target_label: QLabel, side: str | None) -> None:
        p = self.parent
        file_paths, _ = QFileDialog.getOpenFileNames(
            p, "Select ID Image", "",
            "All Supported Files (*.png *.jpg *.jpeg *.bmp *.pdf);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp);;"
            "PDF Files (*.pdf)",
        )
        if not file_paths:
            return
        try:
            file_path = file_paths[-1]
            if not os.path.exists(file_path):
                return
            id_type = getattr(p, "detected_id_type", None) or "Upload"
            folder = p.get_output_folder(id_type, "Upload")
            if file_path.lower().endswith(".pdf"):
                self.handle_pdf_upload(file_path, folder, side, target_label, id_type)
            else:
                self.handle_image_upload(file_path, folder, side, target_label, id_type)
        except Exception as e:
            print("Upload failed:", e)
            QMessageBox.warning(p, "Upload Error", f"Upload failed: {e}")

    def handle_image_upload(self, file_path, folder, side, target_label, selected_id):
        """
        Copies the uploaded file to a temp location so the path is always
        valid regardless of where the user picked the file from.
        The temp file is deleted by db_handler after upload to Supabase.
        """
        import tempfile
        p = self.parent
        ext = os.path.splitext(file_path)[1] or ".jpg"
        try:
            # Create temp dir if needed
            temp_dir = os.path.join("IDscanner", "output", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=ext, dir=temp_dir
            )
            with open(file_path, "rb") as src:
                tmp.write(src.read())
            tmp.close()
            temp_path = tmp.name
        except Exception as e:
            QMessageBox.warning(p, "Copy Error", str(e))
            return
        self.finalise_upload(temp_path, side, target_label, selected_id)

    # ------------------------------------------------------------------
    # PDF upload handling — UNCHANGED logic
    # ------------------------------------------------------------------

    def handle_pdf_upload(self, pdf_path, folder, side, target_label, selected_id):
        p = self.parent
        pages = convert_pdf_pages(pdf_path)
        if not pages:
            QMessageBox.warning(p, "PDF Error",
                "Could not convert the PDF.\n\nMake sure pdf2image and poppler are installed.")
            return
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        if len(pages) == 1:
            self.handle_single_page_pdf(pages[0], folder, base, side, target_label, selected_id)
        else:
            from .inference import detect_and_crop_id
            id_pages = [pg for pg in pages if detect_and_crop_id(pg) is not None]
            if len(id_pages) >= 2:
                self.handle_two_page_pdf(id_pages[:2], folder, base, side, target_label, selected_id)
            elif len(id_pages) == 1:
                self.handle_single_page_pdf(id_pages[0], folder, base, side, target_label, selected_id)
            else:
                self.handle_two_page_pdf(pages[:2], folder, base, side, target_label, selected_id)

    def handle_single_page_pdf(self, page, folder, base, side, target_label, selected_id):
        from .inference import detect_and_crop_id, detect_two_ids
        from .crop_dialog import CropPreviewDialog
        p = self.parent
        left_crop, right_crop = detect_two_ids(page)
        if left_crop is not None and right_crop is not None:
            dialog = CropPreviewDialog(parent=p, front=left_crop, back=right_crop,
                                       full_front=page, full_back=page)
            dialog.exec()
            final_front = dialog.accepted_front if dialog.accepted_front is not None else page
            final_back  = dialog.accepted_back  if dialog.accepted_back  is not None else page
            front_path  = save_image(final_front, folder, f"{base}_front.jpg")
            back_path   = save_image(final_back,  folder, f"{base}_back.jpg")
            p.front_file = make_file_info(front_path, "front")
            p.back_file  = make_file_info(back_path,  "back")
            display_frame_on_label(final_front, target_label)
            self.trigger_inference(selected_id, side="front")
            return
        crop = detect_and_crop_id(page)
        if crop is not None:
            dialog = CropPreviewDialog(parent=p, single=crop, full_page=page)
            dialog.exec()
            final = dialog.accepted_crop if dialog.accepted_crop is not None else page
        else:
            final = page
        dest = save_image(final, folder, f"{base}_page1.jpg")
        self.finalise_upload(dest, side, target_label, selected_id)

    def handle_two_page_pdf(self, pages, folder, base, side, target_label, selected_id):
        from .inference import detect_and_crop_id
        from .crop_dialog import CropPreviewDialog
        p = self.parent
        front_page, back_page = pages[0], pages[1]
        front_crop = detect_and_crop_id(front_page)
        back_crop  = detect_and_crop_id(back_page)
        if front_crop is not None or back_crop is not None:
            dialog = CropPreviewDialog(
                parent=p,
                front=front_crop or front_page,
                back=back_crop   or back_page,
                full_front=front_page, full_back=back_page,
            )
            dialog.exec()
            final_front = dialog.accepted_front if dialog.accepted_front is not None else front_page
            final_back  = dialog.accepted_back  if dialog.accepted_back  is not None else back_page
        else:
            final_front, final_back = front_page, back_page
        front_path = save_image(final_front, folder, f"{base}_front.jpg")
        back_path  = save_image(final_back,  folder, f"{base}_back.jpg")
        p.front_file = make_file_info(front_path, "front")
        p.back_file  = make_file_info(back_path,  "back")
        display_frame_on_label(final_front, target_label)
        self.trigger_inference(selected_id, side="front")

    # ------------------------------------------------------------------
    # Finalise — CHANGED: side=None now also stores to p.front_file
    # ------------------------------------------------------------------

    def finalise_upload(self, dest, side, target_label, selected_id):
        """
        All uploads now go to p.front_file or p.back_file.
        side=None means it's a single-image upload (passport / auto-detect)
        and is treated the same as side='front'.
        """
        p = self.parent
        file_info = make_file_info(dest, side)

        if side == "back":
            p.back_file = file_info
        else:
            # side='front' or side=None — both store as front
            p.front_file = file_info

        frame = cv2.imread(dest)
        if frame is not None and target_label is not None:
            display_frame_on_label(frame, target_label)

        self.trigger_inference(selected_id, side)

    # ------------------------------------------------------------------
    # Trigger inference — CHANGED: side=None uses infer_front_upload()
    # ------------------------------------------------------------------

    def trigger_inference(self, selected_id: str, side: str | None = None) -> None:
        p = self.parent

        if side in (None, "front") and p.front_file and not p.back_file:
            # Single upload or just front uploaded — run auto-detection
            p.continuep3.setEnabled(False)
            QTimer.singleShot(100, p.inference.infer_front_upload)
            return

        if side in ("front", "back"):
            # Both sides present — run full OCR based on detected type
            id_type = getattr(p, "detected_id_type", None) or selected_id

            if p.front_file and p.back_file:
                if id_type == "Driver's License":
                    p.continuep6.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_driver_license_upload)
                elif id_type in ("National ID", "UMID"):
                    p.continuep6.setEnabled(False)
                    QTimer.singleShot(100, p.inference.infer_only_national_id_upload)