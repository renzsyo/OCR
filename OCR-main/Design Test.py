"""
IDscanner_design.py
--------------------
Standalone UI preview for ID Scanner.
Loads the .ui file and wires up ONLY navigation (Back / Continue / ComboBox),
so you can freely work on the design without waiting for the full backend.

HOW TO RUN:
    python IDscanner_design.py

REQUIREMENTS:
    pip install PyQt5

PLACE THIS FILE in the same folder as:
    IDscanner_-_Copy.ui
"""

import sys
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QMainWindow, QApplication


# ──────────────────────────────────────────────
#  Page index constants  (matches .ui order)
# ──────────────────────────────────────────────
PAGE_HOME        = 0   # page   – Select method
PAGE_CAMERA      = 1   # page_2 – Capture single image (camera)
PAGE_UPLOAD_IMG  = 2   # page_3 – Upload single image
PAGE_RESULTS     = 3   # page_4 – Extracted text / results
PAGE_CAMERA_DUAL = 4   # page_5 – Capture front + back via camera
PAGE_UPLOAD_DUAL = 5   # page_6 – Upload front + back images
PAGE_PDF         = 6   # page_7 – Upload PDF


class IDScannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ── Load the .ui file ─────────────────────────────────────────────
        uic.loadUi(r"C:\Users\Daniel\Downloads\Personal\WORK\OJT\OCR-main\IDscanner\IDscanner - Copy.ui", self)

        # ── Show the home page on startup ─────────────────────────────────
        self.Form1.setCurrentIndex(PAGE_HOME)

        # ── Wire up every button ──────────────────────────────────────────
        self._connect_navigation()

        # ── Placeholder visuals for image/camera labels ───────────────────
        self._setup_placeholder_labels()

    # ──────────────────────────────────────────────────────────────────────
    #  Navigation wiring
    # ──────────────────────────────────────────────────────────────────────
    def _connect_navigation(self):

        # ── Page 0 (Home) ─────────────────────────────────────────────────
        # "Continue" routes to the correct page based on the combo selection
        self.continuep1.clicked.connect(self._on_home_continue)

        # ── Page 1 (Camera – single) ──────────────────────────────────────
        self.backButtonp1.clicked.connect(lambda: self._go(PAGE_HOME))
        self.captureButtonp1.clicked.connect(self._stub("Capture Image"))
        self.recaptureButtonp1.clicked.connect(self._stub("Recapture Image"))
        self.continuep2.clicked.connect(lambda: self._go(PAGE_RESULTS))

        # ── Page 2 (Upload Image – single) ────────────────────────────────
        self.backButtonp2.clicked.connect(lambda: self._go(PAGE_HOME))
        self.uploadButtonp4.clicked.connect(self._stub("Upload Image"))
        self.uploadButtonp3.clicked.connect(self._stub("Re-Upload Image"))
        self.continuep3.clicked.connect(lambda: self._go(PAGE_RESULTS))

        # ── Page 3 (Results) ──────────────────────────────────────────────
        self.backButtonp3.clicked.connect(lambda: self._go(PAGE_HOME))
        self.continuep4.clicked.connect(lambda: self._go(PAGE_HOME))
        self.downloadp4.clicked.connect(self._stub("Download as Text"))

        # ── Page 4 (Camera – dual) ────────────────────────────────────────
        self.backButtonp4.clicked.connect(lambda: self._go(PAGE_HOME))
        self.captureButtonp2.clicked.connect(self._stub("Capture Front Image"))
        self.captureButtonp3.clicked.connect(self._stub("Capture Back Image"))
        self.continuep5.clicked.connect(lambda: self._go(PAGE_RESULTS))

        # ── Page 5 (Upload – dual) ────────────────────────────────────────
        self.backButtonp5.clicked.connect(lambda: self._go(PAGE_HOME))
        self.uploadFrontButton.clicked.connect(self._stub("Upload Front Image"))
        self.uploadBackButton.clicked.connect(self._stub("Upload Back Image"))
        self.continuep6.clicked.connect(lambda: self._go(PAGE_RESULTS))

        # ── Page 6 (Upload PDF) ───────────────────────────────────────────
        self.backButtonp7.clicked.connect(lambda: self._go(PAGE_HOME))
        self.ReuploadPDF.clicked.connect(self._stub("Re-Upload PDF"))
        self.continuep7.clicked.connect(lambda: self._go(PAGE_RESULTS))

    # ──────────────────────────────────────────────────────────────────────
    #  Home → Continue routing
    # ──────────────────────────────────────────────────────────────────────
    def _on_home_continue(self):
        index = self.idOption.currentIndex()
        routes = {
            0: None,                # "Select Method" – do nothing
            1: PAGE_CAMERA,         # Capture Using Camera
            2: PAGE_UPLOAD_IMG,     # Upload Image
            3: PAGE_PDF,            # Upload PDF
        }
        destination = routes.get(index)
        if destination is None:
            QtWidgets.QMessageBox.information(
                self, "Select Method", "Please choose a method from the dropdown."
            )
            return
        self._go(destination)

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _go(self, page_index: int):
        """Switch the stacked widget to a page by index."""
        self.Form1.setCurrentIndex(page_index)

    @staticmethod
    def _stub(label: str):
        """Return a slot that pops a 'not yet implemented' dialog."""
        def slot():
            QtWidgets.QMessageBox.information(
                None, label, f"[DESIGN PREVIEW]\n'{label}' not wired to backend yet."
            )
        return slot

    def _setup_placeholder_labels(self):
        """
        Fill camera/image QLabels with a light grey placeholder so the
        layout looks correct even without a real image or camera feed.
        """
        from PyQt6.QtGui import QPixmap, QColor, QPainter
        from PyQt6.QtCore import Qt

        placeholder_labels = [
            "cameraView",
            "uploadedImageView",
            "cameraView1",
            "cameraView2",
            "frontImageView",
            "backImageView",
        ]

        for name in placeholder_labels:
            label: QtWidgets.QLabel = getattr(self, name, None)
            if label is None:
                continue
            w, h = label.width() or 300, label.height() or 200
            pix = QPixmap(w, h)
            pix.fill(QColor("#e8e0d8"))
            painter = QPainter(pix)
            painter.setPen(QColor("#999999"))
            painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, f"[ {name} ]")
            painter.end()
            label.setPixmap(pix)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)


# ──────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IDScannerWindow()
    window.show()
    sys.exit(app.exec())