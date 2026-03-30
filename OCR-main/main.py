import sys, os, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IDscanner"))
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QFileDialog
from IDscanner import CamHandler, FileManager, InferenceHandler, ReviewHandler, UiLoader
from IDscanner.pdf_preview_handler import PdfPreviewHandler
from IDscanner.file_handler import convert_pdf_pages
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Widget refs — populated by UiLoader
        self.uploadedImageView = None
        self.backImageView = None
        self.idOption = None  # now used as METHOD selector
        self.Form1 = None
        self.frontImageView = None

        # Session state
        self.front_file = None
        self.back_file = None
        self.pendingResponse = None
        self.debug_mode = False
        self.pendingDebugImage = None
        self.current_frame = None
        self.lastResult = None
        self.lastIdType = None
        self.detected_id_type = None  # set by InferenceHandler after front detection
        self._last_method = "Unknown"  # set when user picks method on home page

        # Pre-initialize Supabase client on the main thread before any
        # background inference threads start. Lazy init inside a worker
        # thread races with PyTorch/CUDA and causes 0xC0000409 on Windows.
        try:
            from IDscanner.db_handler import init_client_on_main_thread
            init_client_on_main_thread()
        except Exception as _db_e:
            print(f"[MainWindow] DB pre-init failed (non-fatal): {_db_e}")

        self.camera = CamHandler(self)
        self.files = FileManager(self)
        self.inference = InferenceHandler(self)
        self.review = ReviewHandler(self)

        UiLoader(self)

        self.pdf_preview = PdfPreviewHandler(self)

        # Connect ReuploadPDF here — after pdf_preview is assigned —
        # because UiLoader runs before pdf_preview exists
        try:
            self.ReuploadPDF.clicked.connect(self.pdf_preview.reupload_pdf)
        except Exception as e:
            print("[MainWindow] Failed to connect ReuploadPDF:", e)
        # label_21 is the detection status label on page_7 (added in Qt Designer)
        try:
            self.pdfDetectionStatus = self.label_21
            self.pdfDetectionStatus.setMinimumWidth(400)
            self.pdfDetectionStatus.setStyleSheet("font-size: 12px; color: #444;")
        except Exception as e:
            print("[MainWindow] Could not bind pdfDetectionStatus to label_21:", e)
            self.pdfDetectionStatus = QLabel()  # dummy so handler never crashes

        self.page_history = []

    def go_back(self) -> None:
        if not self.page_history:
            return
        current = self.Form1.currentIndex()
        prev_page = self.page_history.pop()

        # ALWAYS stop the camera on every back navigation
        self.camera.stop_camera()

        # Full session reset when going back to home or from review
        if prev_page == 0 or current == 3:
            self.reset_session()
            # Going to home — camera stays off, no restart ever
            self.Form1.setCurrentIndex(prev_page)
            return

        self.Form1.setCurrentIndex(prev_page)

        # Restart camera only when returning to an active camera page
        # AND no capture has been taken on that page yet
        # (only runs when NOT going to page 0 — see early return above)
        if prev_page == 4 and not hasattr(self, "captured_front_frame"):
            self.camera.start_camera()
        elif prev_page == 1 and not hasattr(self, "captured_frame"):
            self.camera.start_camera()

    def go_next(self) -> None:
        current = self.Form1.currentIndex()
        self.page_history.append(current)

        # PDF preview Continue
        if current == 6:
            self.pdf_preview.on_continue()
            self.page_history.pop()
            return

        # Review → restart
        if current == 3:
            self.page_history.pop()
            self.reset_session()
            self.Form1.setCurrentIndex(0)
            return

        # ── Single-cam page (page 1) ──────────────────────────────────
        if current == 1:
            if not hasattr(self, "captured_frame"):
                QMessageBox.warning(self, "No Capture", "Please capture an image first.")
                self.page_history.pop()
                return
            # Validation depends on detected type
            id_type = getattr(self, "detected_id_type", None)
            if id_type and not self.validate_for_type(id_type):
                self.page_history.pop()
                return
            self.go_to_review()
            return

        # ── Single-upload page (page 2) ───────────────────────────────
        if current == 2:
            if not self.front_file:
                QMessageBox.warning(self, "No file", "Please upload a file first.")
                self.page_history.pop()
                return
            id_type = getattr(self, "detected_id_type", None)
            if id_type and not self.validate_for_type(id_type):
                self.page_history.pop()
                return
            self.go_to_review()
            return

        # ── Dual-cam page (page 4) ────────────────────────────────────
        if current == 4:
            if not hasattr(self, "captured_front_frame") or not hasattr(self, "captured_back_frame"):
                QMessageBox.warning(self, "Missing Capture",
                    "Please capture both front and back images first.")
                self.page_history.pop()
                return
            id_type = getattr(self, "detected_id_type", "National ID")
            if id_type == "Driver's License":
                if not self.inference.validate_driver_license_result_sync(
                        getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            else:  # National ID or UMID
                if not self.inference.validate_national_id_result_sync(
                        getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            self.go_to_review()
            return

        # ── Dual-upload page (page 5) ─────────────────────────────────
        if current == 5:
            if not self.front_file or not self.back_file:
                QMessageBox.warning(self, "Missing files",
                    "Please upload both front and back images first.")
                self.page_history.pop()
                return
            id_type = getattr(self, "detected_id_type", "National ID")
            if id_type == "Driver's License":
                if not self.inference.validate_driver_license_result_sync(
                        getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            else:
                if not self.inference.validate_national_id_result_sync(
                        getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            self.go_to_review()
            return

        # ── Home page (page 0) — method selection ────────────────────
        if current == 0:
            self.handle_home_continue()

    def handle_home_continue(self) -> None:
        """Routes from the home page based on the method dropdown."""
        try:
            method = self.idOption.currentText()
        except Exception as e:
            print("[go_next] Failed to read idOption:", e)
            self.page_history.pop()
            return

        if method == "Select Method":
            QMessageBox.information(self, "Selection",
                                    "Please select a scanning method.")
            self.page_history.pop()
            return

        # Reset detection state for new session
        self.inference.reset_detection()
        self.detected_id_type = None

        if method == "Capture Using Camera":
            self._last_method = "Camera"
            self.camera.stop_camera()
            self.Form1.setCurrentIndex(1)
            self.camera.start_camera()

        elif method == "Upload Image":
            self._last_method = "Upload"
            self.camera.stop_camera()
            self.Form1.setCurrentIndex(2)

        elif method == "Upload PDF":
            self._last_method = "PDF"
            self.camera.stop_camera()
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select PDF", "", "PDF Files (*.pdf)")
            if not file_paths:
                self.page_history.pop()
                return
            pdf_path = file_paths[-1]
            if not os.path.exists(pdf_path):
                self.page_history.pop()
                return
            pages = convert_pdf_pages(pdf_path)
            if not pages:
                QMessageBox.warning(self, "PDF Error",
                    "Could not convert the PDF.\n\n"
                    "Make sure pdf2image and poppler are installed.")
                self.page_history.pop()
                return
            self.pdf_preview.load_pdf(pages, pdf_path)

        else:
            self.page_history.pop()

    # Called by InferenceHandler when front detection determines two-sided
    def proceed_to_back_camera(self) -> None:
        """
        Front has been captured and detected as NID/UMID.
        Save the front frame, then navigate to the dual-cam page.
        The left panel (cameraView1) will be frozen with the front image.
        """
        if not hasattr(self, "captured_frame"):
            return


        self.captured_front_frame = self.captured_frame.copy()

        try:
            from PyQt6.QtGui import QImage, QPixmap
            rgb = cv2.cvtColor(self.captured_front_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            self.cameraView1.setPixmap(
                pixmap.scaled(
                    self.cameraView1.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as e:
            print(f"[_proceed_to_back_camera] Could not display front image: {e}")

        self.page_history.append(1)
        self.Form1.setCurrentIndex(4)
        self.camera.start_camera()

    def proceed_to_back_upload(self) -> None:
        """
        Front has been uploaded and detected as NID/UMID.
        Pre-populate the front panel on the dual-upload page.
        """
        # front_file is already set by FileManager.finalise_upload
        if not self.front_file:
            return

        # Display the front image in frontImageView
        try:
            import cv2
            from PyQt6.QtGui import QImage, QPixmap
            frame = cv2.imread(self.front_file["path"])
            if frame is not None:
                from IDscanner.file_handler import display_frame_on_label
                display_frame_on_label(frame, self.frontImageView)
        except Exception as e:
            print(f"[_proceed_to_back_upload] Could not display front image: {e}")

        self.page_history.append(2)
        self.Form1.setCurrentIndex(5)

    def validate_for_type(self, id_type: str) -> bool:
        result = getattr(self, "pendingResponse", {})
        if id_type == "Passport":
            return self.inference.validate_passport_result_sync(result)
        elif id_type == "Driver's License":
            return self.inference.validate_driver_license_result_sync(result)
        elif id_type in ("National ID", "UMID"):
            return self.inference.validate_national_id_result_sync(result)
        elif id_type == "PhilHealth":
            return self.inference.validate_philhealth_result_sync(result)
        elif id_type == "TIN":
            return self.inference.validate_tin_result_sync(result)
        return True

    def go_to_review(self) -> None:
        self.camera.stop_camera()
        self.review.show_review_page()
        self.Form1.setCurrentIndex(3)

    @staticmethod
    def get_output_folder(category: str, subfolder: str) -> str:
        folder = os.path.join("IDscanner/output", category.strip(), subfolder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def reset_session(self) -> None:
        print("[reset_session] called")
        for attr in ("captured_frame", "captured_front_frame", "captured_back_frame"):
            if hasattr(self, attr):
                delattr(self, attr)
        try:
            self.captureButtonp2.setText("Capture Image")
            self.captureButtonp3.setText("Capture Image")
        except Exception:
            pass
        self.front_file = None
        self.back_file = None
        self.detected_id_type = None
        self.pendingResponse = None
        self.pendingDebugImage = None
        self.pendingDebugImageBack = None
        self.inference.reset_detection()
        self.camera.stop_camera()
        for widget_name in [
            "uploadedImageView", "cameraView",
            "cameraView1", "cameraView2", "frontImageView", "backImageView",
        ]:
            try:
                w = getattr(self, widget_name, None)
                if w:
                    w.clear()
            except Exception:
                pass

    def on_debug_toggled(self, state: int) -> None:
        self.debug_mode = (state == 2)
        print("[DEBUG mode] is now:", self.debug_mode)

    def closeEvent(self, event) -> None:
        self.camera.stop_camera()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()