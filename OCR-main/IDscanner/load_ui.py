"""
load_ui.py
----------
CHANGES FROM PREVIOUS VERSION:
  - REMOVED [lines 29-36]:  radio button group (cameraOption, uploadOption,
                             uploadPDFOption) — no longer in the UI
  - REMOVED [lines 29-46]:  idOption dropdown delegate/placeholder setup removed;
                             "Select Method" first item is disabled via HiddenFirstItem
                             delegate instead
  - CHANGED [line 22]:      UI filename changed from IDscanner.ui to
                             IDscanner - Copy.ui
  - CHANGED [lines 75-78]:  captureButtonp1 connects to camera.capture_image
                             (slot now triggers auto-detection internally)
  - ADDED   [lines 113-117]:uploadButtonp4 connection added (second upload button
                             on single upload page)
  - NOTE:                   ReuploadPDF is connected in main.py (after pdf_preview
                             is assigned) — NOT here, as pdf_preview doesn't exist
                             when UiLoader runs
  - REMOVED [lines 137-142]:fileListWidget signal connections removed entirely
                             (widget no longer exists on upload page)
  - KEPT    [lines 55-74]:  all navigation button connections unchanged
  - KEPT    [lines 80-136]: all other upload, download, debug connections unchanged
"""

from PyQt6 import uic
from PyQt6.QtCore import Qt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow


class UiLoader:
    def __init__(self, parent: "MainWindow") -> None:
        uic.loadUi("IDscanner\\IDscanner - Copy.ui", parent)
        try:
            parent.Form1.setCurrentIndex(0)
        except Exception as e:
            print("[UiLoader/__init__] Failed to set initial page index:", e)
        self.connect_signals(parent)

    def connect_signals(self, p: "MainWindow") -> None:

        # ── Method dropdown — disable the placeholder "Select Method" item ──
        try:
            from PyQt6.QtWidgets import QStyledItemDelegate
            from PyQt6.QtCore import QSize

            class HiddenFirstItem(QStyledItemDelegate):
                def sizeHint(self, option, index):
                    if index.row() == 0:
                        return QSize(0, 0)
                    return super().sizeHint(option, index)

            p.idOption.setItemDelegate(HiddenFirstItem(p.idOption))
            p.idOption.model().item(0).setEnabled(False)
        except Exception as e:
            print("[UiLoader] Failed to configure idOption dropdown:", e)

        # ── Navigation buttons ───────────────────────────────────────────────
        nav_buttons = [
            ("continuep1",   p.go_next),
            ("continuep4",   p.go_next),
            ("continuep7",   p.go_next),   # PDF preview Continue
            ("backButtonp1", p.go_back),
            ("backButtonp2", p.go_back),
            ("backButtonp3", p.go_back),
            ("backButtonp4", p.go_back),
            ("backButtonp5", p.go_back),
            ("backButtonp7", p.go_back),   # PDF preview Back
        ]
        for name, slot in nav_buttons:
            try:
                getattr(p, name).clicked.connect(slot)
            except Exception as e:
                print(f"[UiLoader] Failed to connect '{name}':", e)

        # Inference continue buttons
        for name in ("continuep2", "continuep3", "continuep5", "continuep6"):
            try:
                getattr(p, name).clicked.connect(p.go_next)
            except Exception as e:
                print(f"[UiLoader] Failed to connect '{name}':", e)

        # ── Camera buttons ───────────────────────────────────────────────────
        # captureButtonp1: single-cam page — triggers auto-detection
        try:
            p.captureButtonp1.clicked.connect(p.camera.capture_image)
        except Exception as e:
            print("[UiLoader] Failed to connect captureButtonp1:", e)

        try:
            p.recaptureButtonp1.clicked.connect(p.camera.recapture_image)
        except Exception as e:
            print("[UiLoader] Failed to connect recaptureButtonp1:", e)

        # Dual-cam page capture buttons (front/back)
        try:
            p.captureButtonp2.clicked.connect(
                lambda: p.camera.toggle_capture(
                    "captured_front_frame", p.cameraView1, p.captureButtonp2)
            )
        except Exception as e:
            print("[UiLoader] Failed to connect captureButtonp2:", e)

        try:
            p.captureButtonp3.clicked.connect(
                lambda: p.camera.toggle_capture(
                    "captured_back_frame", p.cameraView2, p.captureButtonp3)
            )
        except Exception as e:
            print("[UiLoader] Failed to connect captureButtonp3:", e)

        # ── Upload buttons ───────────────────────────────────────────────────
        # Single upload page
        try:
            p.uploadButtonp3.clicked.connect(
                lambda: p.files.upload_image(p.uploadedImageView, None))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadButtonp3:", e)

        # Dual upload page — re-upload buttons use side="front"/"back"
        # so detected_id_type is already set before these are reachable
        try:
            p.uploadFrontButton.clicked.connect(
                lambda: p.files.upload_image(p.frontImageView, side="front"))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadFrontButton:", e)

        try:
            p.uploadBackButton.clicked.connect(
                lambda: p.files.upload_image(p.backImageView, side="back"))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadBackButton:", e)

        # Also connect uploadButtonp4 (single upload page — second upload button)
        try:
            p.uploadButtonp4.clicked.connect(
                lambda: p.files.upload_image(p.uploadedImageView, None))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadButtonp4:", e)

        # ── Download ─────────────────────────────────────────────────────────
        try:
            p.downloadp4.clicked.connect(
                lambda: p.review.download_text(p.resultbox, "extracted_text"))
        except Exception as e:
            print("[UiLoader] Failed to connect downloadp4:", e)

        # ── Debug checkbox ───────────────────────────────────────────────────
        try:
            p.debugOption.stateChanged.connect(p.on_debug_toggled)
        except Exception as e:
            print("[UiLoader] Failed to connect debugOption:", e)