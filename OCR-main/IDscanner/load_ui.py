"""
load_ui.py
----------
CHANGES FROM PREVIOUS VERSION:
  - ADDED: connect backButtonp7 -> go_back
  - ADDED: connect continuep7  -> go_next  (page 6 / PDF preview)
"""

from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QButtonGroup
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow


class UiLoader:
    def __init__(self, parent: "MainWindow") -> None:
        uic.loadUi("IDscanner\\IDscanner.ui", parent)
        try:
            parent.Form1.setCurrentIndex(0)
        except Exception as e:
            print("[UiLoader/__init__] Failed to set initial page index:", e)
        self.connect_signals(parent)

    def connect_signals(self, p: "MainWindow") -> None:
        # Dropdown — hide placeholder item
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

        # Radio button group
        try:
            p._inputMethodGroup = QButtonGroup(p)
            p._inputMethodGroup.addButton(p.cameraOption)
            p._inputMethodGroup.addButton(p.uploadOption)
            p._inputMethodGroup.addButton(p.uploadPDFOption)
        except Exception as e:
            print("[UiLoader] Failed to group radio buttons:", e)

        # Navigation buttons
        nav_buttons = [
            ("continuep1",    p.go_next),
            ("continuep4",    p.go_next),
            ("continuep7",    p.go_next),   # ADDED: PDF preview Continue
            ("backButtonp1",  p.go_back),
            ("backButtonp2",  p.go_back),
            ("backButtonp3",  p.go_back),
            ("backButtonp4",  p.go_back),
            ("backButtonp5",  p.go_back),
            ("backButtonp7",  p.go_back),   # ADDED: PDF preview Back
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

        # Camera buttons
        for name, slot in [
            ("captureButtonp1",  p.camera.capture_image),
            ("recaptureButtonp1", p.camera.recapture_image),
        ]:
            try:
                getattr(p, name).clicked.connect(slot)
            except Exception as e:
                print(f"[UiLoader] Failed to connect '{name}':", e)

        try:
            p.captureButtonp2.clicked.connect(
                lambda: p.camera.toggle_capture("captured_front_frame", p.cameraView1, p.captureButtonp2)
            )
        except Exception as e:
            print("[UiLoader] Failed to connect captureButtonp2:", e)

        try:
            p.captureButtonp3.clicked.connect(
                lambda: p.camera.toggle_capture("captured_back_frame", p.cameraView2, p.captureButtonp3)
            )
        except Exception as e:
            print("[UiLoader] Failed to connect captureButtonp3:", e)

        # Upload buttons
        try:
            p.uploadButtonp3.clicked.connect(lambda: p.files.upload_image(p.uploadedImageView, None))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadButtonp3:", e)
        try:
            p.uploadFrontButton.clicked.connect(lambda: p.files.upload_image(p.frontImageView, side="front"))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadFrontButton:", e)
        try:
            p.uploadBackButton.clicked.connect(lambda: p.files.upload_image(p.backImageView, side="back"))
        except Exception as e:
            print("[UiLoader] Failed to connect uploadBackButton:", e)

        # Download
        try:
            p.downloadp4.clicked.connect(lambda: p.review.download_text(p.resultbox, "extracted_text"))
        except Exception as e:
            print("[UiLoader] Failed to connect downloadp4:", e)

        # Debug checkbox
        try:
            p.debugOption.stateChanged.connect(p.on_debug_toggled)
        except Exception as e:
            print("[UiLoader] Failed to connect debugOption:", e)

        # File list widget
        try:
            p.fileListWidget.currentRowChanged.connect(p.files.on_current_row_changed)
            p.fileListWidget.itemClicked.connect(p.files.list_item_clicked)
            p.fileListWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            p.fileListWidget.customContextMenuRequested.connect(p.files.show_list_menu)
        except Exception as e:
            print("[UiLoader] Failed to connect fileListWidget signals:", e)