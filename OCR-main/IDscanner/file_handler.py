"""
file_handler.py
---------------
Handles image and PDF uploads via the manual upload buttons (pages 2 & 5).
PDF-to-scan flow is handled by PdfPreviewHandler + pdf_preview_handler.py.
"""

import os, shutil, cv2
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QMenu, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow


# ---------------------------------------------------------------------------
# Module-level helpers
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
        "path": path,
        "name": os.path.basename(path),
        "size": f"{os.path.getsize(path) / (1024 * 1024):.2f} MB",
        "status": "Completed",
        "side": side,
    }


def display_frame_on_label(frame: np.ndarray, label: QLabel) -> None:
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
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
        self.uploaded_files = []
        self.current_index = -1

    # ------------------------------------------------------------------
    # Manual upload buttons (pages 2 & 5)
    # ------------------------------------------------------------------

    def upload_image(self, target_label: QLabel, side: str | None) -> None:
        p = self.parent
        file_paths, _ = QFileDialog.getOpenFileNames(
            p, "Select ID Images", "",
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
            selected_id = p.idOption.currentText()
            folder = p.get_output_folder(selected_id, "Upload")
            if file_path.lower().endswith(".pdf"):
                self._handle_pdf_upload(file_path, folder, side, target_label, selected_id)
            else:
                self.handle_image_upload(file_path, folder, side, target_label, selected_id)
        except Exception as e:
            print("Upload failed:", e)
            QMessageBox.warning(p, "Upload Error", f"Upload failed: {e}")

    def handle_image_upload(self, file_path, folder, side, target_label, selected_id):
        p = self.parent
        filename = os.path.basename(file_path)
        dest = os.path.join(folder, filename)
        try:
            shutil.copyfile(file_path, dest)
        except Exception as e:
            QMessageBox.warning(p, "Copy Error", str(e))
            return
        self.finalise_upload(dest, side, target_label, selected_id)

    def _handle_pdf_upload(self, pdf_path, folder, side, target_label, selected_id):
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

    def finalise_upload(self, dest, side, target_label, selected_id):
        p = self.parent
        file_info = make_file_info(dest, side)
        if side == "front":
            p.front_file = file_info
        elif side == "back":
            p.back_file = file_info
        else:
            self.uploaded_files.append(file_info)
            self.current_index = len(self.uploaded_files) - 1
            self.refresh_file_list()
            if self.current_index >= 0:
                p.fileListWidget.setCurrentRow(self.current_index)
        frame = cv2.imread(dest)
        if frame is not None:
            display_frame_on_label(frame, target_label)
        self.trigger_inference(selected_id, side)

    def trigger_inference(self, selected_id: str, side: str | None = None) -> None:
        p = self.parent
        if side is None:
            p.continuep3.setEnabled(False)
            QTimer.singleShot(100, p.inference.infer_page3_upload_passport)
        if side in ("front", "back"):
            if selected_id == "Driver's License" and p.front_file and p.back_file:
                p.continuep6.setEnabled(False)
                QTimer.singleShot(100, p.inference.infer_only_driver_license_upload)
            if selected_id == "National ID" and p.front_file and p.back_file:
                p.continuep6.setEnabled(False)
                QTimer.singleShot(100, p.inference.infer_only_national_id_upload)

    # ------------------------------------------------------------------
    # File list management — unchanged
    # ------------------------------------------------------------------

    def refresh_file_list(self) -> None:
        p = self.parent
        try:
            p.fileListWidget.clear()
            for file in self.uploaded_files:
                p.fileListWidget.addItem(
                    f"{file['name']} | {file['size']} | {file['status']}"
                )
            if 0 <= self.current_index < len(self.uploaded_files):
                p.fileListWidget.setCurrentRow(self.current_index)
            else:
                p.fileListWidget.setCurrentRow(-1)
        except Exception as e:
            print("refresh_file_list error:", e)

    def list_item_clicked(self, item) -> None:
        p = self.parent
        row = p.fileListWidget.row(item)
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(p.uploadedImageView)

    def on_current_row_changed(self, row: int) -> None:
        p = self.parent
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(p.uploadedImageView)
        else:
            self.current_index = -1
            p.uploadedImageView.clear()
            try:
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[on_current_row_changed] Failed to clear labels:", e)

    def show_list_menu(self, position) -> None:
        p = self.parent
        item = p.fileListWidget.itemAt(position)
        menu = QMenu(p)
        if item is None:
            add_action = menu.addAction("Add files")
            action = menu.exec(p.fileListWidget.mapToGlobal(position))
            if action == add_action:
                self.upload_image(p.uploadedImageView, None)
            return
        row = p.fileListWidget.row(item)
        if row < 0 or row >= len(self.uploaded_files):
            return
        p.fileListWidget.setCurrentRow(row)
        self.current_index = row
        delete_action = menu.addAction("Delete")
        action = menu.exec(p.fileListWidget.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected_file()

    def delete_selected_file(self) -> None:
        p = self.parent
        row = p.fileListWidget.currentRow()
        if row < 0 or row >= len(self.uploaded_files):
            return
        try:
            removed = self.uploaded_files.pop(row)
            print("Removed:", removed["name"])
        except Exception as e:
            print("delete_selected_file error:", e)
            return
        self.current_index = -1 if not self.uploaded_files else min(row, len(self.uploaded_files) - 1)
        self.refresh_file_list()
        if self.current_index >= 0:
            p.fileListWidget.setCurrentRow(self.current_index)
            self.display_file_details(p.uploadedImageView)
        else:
            try:
                p.uploadedImageView.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[delete_selected_file] Failed to clear UI:", e)

    def display_file_details(self, target_label: QLabel) -> None:
        p = self.parent
        if self.current_index < 0 or self.current_index >= len(self.uploaded_files):
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception:
                pass
            return
        file_info = self.uploaded_files[self.current_index]
        path = file_info.get("path")
        if not path or not os.path.exists(path):
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception:
                pass
            return
        frame = cv2.imread(path)
        if frame is None:
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception:
                pass
            return
        try:
            display_frame_on_label(frame, target_label)
            p.fileNameLabel.setText(file_info["name"])
            p.fileSizeLabel.setText(file_info["size"])
            p.fileStatusLabel.setText(file_info["status"])
        except Exception as e:
            print("[display_file_details] Failed to render:", e)
            try:
                target_label.clear()
            except Exception:
                pass