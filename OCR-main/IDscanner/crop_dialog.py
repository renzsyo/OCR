import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QWidget, QMainWindow
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


def numpy_to_pixmap(image: np.ndarray, max_width: int = 400, max_height: int = 300) -> QPixmap:
    """Convert a BGR numpy array to a scaled QPixmap for display."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        max_width, max_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

class CropPreviewDialog(QDialog):
    """
    Shows the user a preview of the auto-detected crop(s) from a PDF page.

    Single mode  — one image, user confirms or rejects the crop.
    Double mode  — two images (front + back), user confirms or rejects both.

    Result is stored in self.accepted_crop (single) or
    self.accepted_front / self.accepted_back (double).
    """
    def __init__(
        self,
        parent=None,
        front: np.ndarray | None = None,
        back: np.ndarray | None = None,
        single: np.ndarray | None = None,
        full_page: np.ndarray | None = None,
        full_front: np.ndarray | None = None,
        full_back: np.ndarray | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Detected ID Crop")
        self.setModal(True)

        # Outputs — set after user interaction
        self.accepted_crop: np.ndarray | None = None
        self.accepted_front: np.ndarray | None = None
        self.accepted_back: np.ndarray | None = None

        # Store fallback full pages
        self.full_page = full_page
        self.full_front = full_front
        self.full_back = full_back

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        if single is not None:
            self.build_single_mode(layout, single)
        elif front is not None and back is not None:
            self.build_double_mode(layout, front, back)
        else:
            self.accept_full()
            return

        self.add_buttons(layout, double=(front is not None or back is not None))

    #Layout builders

    def build_single_mode(self, layout: QVBoxLayout, crop: np.ndarray | None) -> None:
        self.crop = crop

        info = QLabel("An ID was detected in your PDF. \nUse this cropped region for scanning?")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        img_label= QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setPixmap(numpy_to_pixmap(crop, 500, 350))
        layout.addWidget(img_label)

    def build_double_mode(self, layout: QVBoxLayout, front: np.ndarray, back: np.ndarray) -> None:
        self.front = front
        self.back = back

        info = QLabel(
            "Two ID sides were detected in your PDF.\n"
            "Left side = Front   |   Right side = Back\n"
            "Use these crops for scanning?")

        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setSpacing(16)

        for label_text, img in (("Front", front), ("Back", back)):
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setSpacing(4)

            title = QLabel(f"<b>{label_text}</b>")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_layout.addWidget(title)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setPixmap(numpy_to_pixmap(img, 360, 260))
            col_layout.addWidget(img_label)

            row_layout.addWidget(col)

        layout.addWidget(row)

    def add_buttons(self, layout: QHBoxLayout, double: bool) -> None:
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setSpacing(12)

        yes_btn = QPushButton("Yes, use crop(s)")
        yes_btn.setFixedHeight(36)
        yes_btn.clicked.connect(self.on_accept)

        no_btn = QPushButton("No, use full page")
        no_btn.setFixedHeight(36)
        no_btn.clicked.connect(self.on_reject)

        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        layout.addWidget(btn_row)

    #Button Handlers

    def on_accept(self) -> None:
        if hasattr(self, "crop"):
            self.accepted_crop = self.crop
        if hasattr(self, "front"):
            self.accepted_front = self.front
        if hasattr(self, "back"):
            self.accepted_back = self.back
        self.accept()

    def on_reject(self) -> None:
        # Fall back to full page image(s)
        self.accepted_crop = self.full_page
        self.accepted_front = self.full_front
        self.accepted_back = self.full_back
        self.accept()

    def accept_full(self) -> None:
        self.accepted_crop = self.full_page
        self.accepted_front = self.full_front
        self.accepted_back = self.full_back
        self.accept()