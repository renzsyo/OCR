import cv2
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class ReviewHandler:
    def __init__(self, parent):
        self.parent = parent

    def show_review_page(self):

        p = self.parent
        p.reviewTabWidget.clear()

        print("[REVIEW DEBUG] debug_mode:", getattr(p, "debug_mode", False))
        print("[REVIEW DEBUG] pendingDebugImage:", getattr(p, "pendingDebugImage", None))
        print("[REVIEW DEBUG] pendingResponse:", p.pendingResponse)

        # Front/back uploaded files (NID or DL upload flow)
        if p.front_file:
            self._add_file_tab(p.front_file, "Front Side")
        if p.back_file:
            self._add_file_tab(p.back_file, "Back Side")

        # Populate the shared resultbox from pendingResponse
        if p.pendingResponse is not None:
            selected_id = p.idOption.currentText()
            formatted = p.inference._format_pending_response(p.pendingResponse, selected_id)
            p.resultbox.setPlainText(formatted)
            p.pendingResponse = None
            print("[DEBUG] resultbox populated from pendingResponse")

        # Single captured frame (passport camera flow)
        if hasattr(p, "captured_frame"):
            tab = QWidget()
            layout = QHBoxLayout(tab)
            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(p.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(pixmap.scaled(
                pictureView.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            layout.addWidget(pictureView)
            p.reviewTabWidget.addTab(tab, "Captured Image")

        # Front/back captured frames (NID or DL camera flow)
        for frame_attr, tab_label in (("captured_front_frame", "Front Capture"),
                                      ("captured_back_frame", "Back Capture")):
            if not hasattr(p, frame_attr):
                continue
            frame = getattr(p, frame_attr)
            tab = QWidget()
            layout = QHBoxLayout(tab)
            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(pixmap.scaled(
                pictureView.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            layout.addWidget(pictureView)
            p.reviewTabWidget.addTab(tab, tab_label)

        # Uploaded files list (passport upload flow)
        for file in p.files.uploaded_files:
            path = file.get("path")
            frame = cv2.imread(path)
            if frame is None:
                continue
            tab = QWidget()
            layout = QHBoxLayout(tab)
            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(pixmap.scaled(
                pictureView.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            layout.addWidget(pictureView)
            p.reviewTabWidget.addTab(tab, file["name"])

        if getattr(p, "debug_mode", False) and getattr(p, "pendingDebugImage", None):
            debug_path = p.pendingDebugImage
            frame = cv2.imread(debug_path)
            if frame is not None:
                tab = QWidget()
                layout = QHBoxLayout(tab)
                pictureView = QLabel()
                pictureView.setFixedSize(512, 384)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                pictureView.setPixmap(pixmap.scaled(
                    pictureView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
                layout.addWidget(pictureView)
                p.reviewTabWidget.addTab(tab, "Debug - Bounding Boxes")
            p.pendingDebugImage = None

    def _add_file_tab(self, file_info, tab_name):
        p = self.parent
        path = file_info.get("path")
        frame = cv2.imread(path)
        if frame is None:
            return

        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Image preview
        pictureView = QLabel()
        pictureView.setFixedSize(512, 384)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pictureView.setPixmap(
            pixmap.scaled(pictureView.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )
        layout.addWidget(pictureView)
        p.reviewTabWidget.addTab(tab, tab_name)

    def download_text(self, text_box, default_name="extracted_text"):
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