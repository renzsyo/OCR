import cv2
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

class ReviewHandler:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent

    @staticmethod
    def frame_to_tab(frame) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        pictureView = QLabel()
        pictureView.setFixedSize(512, 384)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        pictureView.setPixmap(pixmap.scaled(
            pictureView.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        layout.addWidget(pictureView)
        return tab

    def show_review_page(self) -> None:
        p = self.parent
        p.reviewTabWidget.clear()

        print("[REVIEW DEBUG] debug_mode:", getattr(p, "debug_mode", False))
        print("[REVIEW DEBUG] pendingDebugImage:", getattr(p, "pendingDebugImage", None))
        print("[REVIEW DEBUG] pendingResponse:", p.pendingResponse)

        # Front/back uploaded files (NID or DL upload flow)
        if p.front_file:
            self.add_file_tab(p.front_file, "Front Side")
        if p.back_file:
            self.add_file_tab(p.back_file, "Back Side")

        # Populate the shared resultbox from pendingResponse
        if p.pendingResponse is not None:
            selected_id = p.idOption.currentText()
            formatted = p.inference.format_pending_response(p.pendingResponse, selected_id)
            p.resultbox.setPlainText(formatted)
            p.pendingResponse = None
            print("[DEBUG] resultbox populated from pendingResponse")

        # Single captured frame (passport camera flow)
        if hasattr(p, "captured_frame"):
            tab = ReviewHandler.frame_to_tab(p.captured_frame)
            p.reviewTabWidget.addTab(tab, "Captured Image")

        # Front/back captured frames (NID or DL camera flow)
        for frame_attr, tab_label in (("captured_front_frame", "Front Capture"),
                                      ("captured_back_frame", "Back Capture")):
            if not hasattr(p, frame_attr):
                continue
            tab = ReviewHandler.frame_to_tab(getattr(p, frame_attr))
            p.reviewTabWidget.addTab(tab, tab_label)

        # Uploaded files list (passport upload flow)
        for file in p.files.uploaded_files:
            path = file.get("path")
            frame = cv2.imread(path)
            if frame is None:
                continue
            tab = ReviewHandler.frame_to_tab(frame)
            p.reviewTabWidget.addTab(tab, file["name"])

        if getattr(p, "debug_mode", False) and getattr(p, "pendingDebugImage", None):
            debug_path = p.pendingDebugImage
            frame = cv2.imread(debug_path)
            if frame is not None:
                tab = ReviewHandler.frame_to_tab(frame)
                p.reviewTabWidget.addTab(tab, "Debug - Bounding Boxes")
            p.pendingDebugImage = None


    def add_file_tab(self, file_info: dict, tab_name: str) -> None:
        p = self.parent
        path = file_info.get("path")
        frame = cv2.imread(path)
        if frame is None:
            return

        tab = ReviewHandler.frame_to_tab(frame)
        p.reviewTabWidget.addTab(tab, tab_name)

    def download_text(self, text_box, default_name: str ="extracted_text") -> None:
        p = self.parent
        text = text_box.toPlainText()
        if not text.strip():
            QMessageBox.warning(p, "No text", "There is no text to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            p,
            "Save Extracted Text",
            default_name + ".txt",
            "Text Files (*.txt)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(p, "Saved", f"File saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(p, "Error", f"Could not save text file: {e}")