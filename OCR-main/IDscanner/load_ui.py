from PyQt6 import uic
from PyQt6.QtCore import Qt

class UiLoader:
    def __init__(self, parent):
        uic.loadUi("IDscanner\\IDscanner.ui", parent)

        try:
            parent.Form1.setCurrentIndex(0)
        except Exception as e:
            print("INIT ",e)

        self._connect_signals(parent)

    def _connect_signals(self, p):
        try:
            from PyQt6.QtWidgets import QStyledItemDelegate
            from PyQt6.QtCore import QSize

            class HiddenFirstItem(QStyledItemDelegate):
                def sizeHint(self, option, index):
                    if index.row() == 0:
                        return QSize(0, 0)
                    return super().sizeHint(option, index)

            p.idOption.setItemDelegate(HiddenFirstItem(p.idOption))
            model = p.idOption.model()
            model.item(0).setEnabled(False)

        except Exception as e:
            print("[COMBO ERROR]", e)

        for name, slot in [
            ("continuep1",   p.go_next),
            ("continuep4",   p.go_next),
            ("backButtonp1", p.go_back),
            ("backButtonp2", p.go_back),
            ("backButtonp3", p.go_back),
            ("backButtonp4", p.go_back),
            ("backButtonp5", p.go_back),
        ]:
            try:
                getattr(p, name).clicked.connect(slot)
            except Exception:
                pass

        # Inference
        for name, slot in [
            ("continuep2", p.inference.infer_page2_camera_passport),
            ("continuep3", p.inference.infer_page3_upload_passport),
            ("continuep5", p.inference.infer_page5),
            ("continuep6", p.inference.infer_page6),
        ]:
            try:
                getattr(p, name).clicked.connect(slot)
            except Exception:
                pass

        # Camera
        try:
            p.captureButtonp1.clicked.connect(p.camera.capture_image)
        except Exception:
            pass
        try:
            p.recaptureButtonp1.clicked.connect(p.camera.recapture_image)
        except Exception:
            pass
        try:
            p.captureButtonp2.clicked.connect(
                lambda: p.camera.toggle_capture("captured_front_frame", p.cameraView1, p.captureButtonp2)
            )
        except Exception:
            pass
        try:
            p.captureButtonp3.clicked.connect(
                lambda: p.camera.toggle_capture("captured_back_frame", p.cameraView2, p.captureButtonp3)
            )
        except Exception:
            pass

        # Upload
        try:
            p.uploadButtonp3.clicked.connect(lambda: p.files.upload_image(p.uploadedImageView))
        except Exception:
            pass
        try:
            p.uploadFrontButton.clicked.connect(lambda: p.files.upload_image(p.frontImageView, side="front"))
        except Exception:
            pass
        try:
            p.uploadBackButton.clicked.connect(lambda: p.files.upload_image(p.backImageView, side="back"))
        except Exception:
            pass

        # Download
        try:
            p.downloadp4.clicked.connect(lambda: p.review.download_text(p.resultbox, "extracted_text"))
        except Exception:
            pass
        try:
            p.debugOption.stateChanged.connect(p.on_debug_toggled)
        except Exception as e:
            print("[DEBUG CONNECT ERROR]", e)

        # File list widget
        try:
            p.fileListWidget.currentRowChanged.connect(p.files.on_current_row_changed)
            p.fileListWidget.itemClicked.connect(p.files.list_item_clicked)
            p.fileListWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            p.fileListWidget.customContextMenuRequested.connect(p.files.show_list_menu)
        except Exception:
            pass