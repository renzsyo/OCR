"""
main.py
-------
CHANGES FROM PREVIOUS VERSION:
  - ADDED: import PdfPreviewHandler
  - ADDED: self.pdf_preview = PdfPreviewHandler(self)
  - ADDED: self.pdfDetectionStatus QLabel (dynamic, injected onto page_7)
  - CHANGED: pdf_checked branch — opens file dialog, converts PDF,
             calls pdf_preview.load_pdf() to show preview page (index 6)
  - ADDED: page 6 case in go_next — delegates to pdf_preview.on_continue()
  - ADDED: self.lastResult / self.lastIdType init (were missing)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IDscanner"))
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QFileDialog
from PyQt6.QtCore import Qt
from IDscanner import CamHandler, FileManager, InferenceHandler, ReviewHandler, UiLoader
from IDscanner.pdf_preview_handler import PdfPreviewHandler
from IDscanner.file_handler import convert_pdf_pages


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.uploadedImageView = None
        self.backImageView = None
        self.cameraOption = None
        self.idOption = None
        self.Form1 = None
        self.uploadOption = None
        self.uploadPDFOption = None
        self.frontImageView = None
        self.front_file = None
        self.back_file = None
        self.pendingResponse = None
        self.debug_mode = False
        self.pendingDebugImage = None
        self.current_frame = None
        self.lastResult = None      # ADDED: was missing, caused download errors
        self.lastIdType = None      # ADDED: was missing, caused download errors

        self.camera    = CamHandler(self)
        self.files     = FileManager(self)
        self.inference = InferenceHandler(self)
        self.review    = ReviewHandler(self)

        UiLoader(self)

        # ADDED: PDF preview handler (must come after UiLoader so widgets exist)
        self.pdf_preview = PdfPreviewHandler(self)

        # label_21 is the detection status label on page_7 (added in Qt Designer)
        try:
            self.pdfDetectionStatus = self.label_21
            self.pdfDetectionStatus.setMinimumWidth(400)
            self.pdfDetectionStatus.setStyleSheet("font-size: 12px; color: #444;")
        except Exception as e:
            print("[MainWindow] Could not bind pdfDetectionStatus to label_21:", e)
            self.pdfDetectionStatus = QLabel()  # dummy so handler never crashes

        self.page_flow = {1: 3, 2: 3, 4: 3, 5: 3, 3: 0}
        self.page_history = []

    def go_back(self) -> None:
        if not self.page_history:
            return
        current = self.Form1.currentIndex()
        prev_page = self.page_history.pop()
        if current == 3 or prev_page == 0:
            self.reset_session()
        self.Form1.setCurrentIndex(prev_page)

    def go_next(self) -> None:
        current = self.Form1.currentIndex()
        self.page_history.append(current)

        # ADDED: PDF preview Continue button
        if current == 6:
            self.pdf_preview.on_continue()
            self.page_history.pop()
            return

        if current in self.page_flow:
            if current == 1 and not hasattr(self, "captured_frame"):
                QMessageBox.warning(self, "No Capture", "Please capture an image first.")
                self.page_history.pop(); return
            if current == 1 and hasattr(self, "captured_frame"):
                if not self.inference.validate_passport_result_sync(getattr(self, "pendingResponse", {})):
                    self.page_history.pop(); return
            if current == 2 and not self.files.uploaded_files:
                QMessageBox.warning(self, "No file", "Please upload a file first.")
                self.page_history.pop(); return
            if current == 2 and self.files.uploaded_files:
                if not self.inference.validate_passport_result_sync(getattr(self, "pendingResponse", {})):
                    self.page_history.pop(); return
            if current == 4 and (not hasattr(self, "captured_front_frame") or not hasattr(self, "captured_back_frame")):
                QMessageBox.warning(self, "Missing Capture", "Please capture both front and back images first.")
                self.page_history.pop(); return
            if current == 4:
                selected_id = self.idOption.currentText()
                if selected_id == "Driver's License":
                    if not self.inference.validate_driver_license_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop(); return
                if selected_id == "National ID":
                    if not self.inference.validate_national_id_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop(); return
            if current == 5 and (not self.front_file or not self.back_file):
                QMessageBox.warning(self, "Missing files", "Please upload both front and back images first.")
                self.page_history.pop(); return
            if current == 5:
                selected_id = self.idOption.currentText()
                if selected_id == "Driver's License":
                    if not self.inference.validate_driver_license_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop(); return
                if selected_id == "National ID":
                    if not self.inference.validate_national_id_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop(); return

            next_page = self.page_flow[current]
            self.Form1.setCurrentIndex(next_page)
            if next_page == 0:
                self.reset_session()
            elif next_page == 3:
                self.camera.stop_camera()
                self.review.show_review_page()
            return

        # Home page (page 0) routing
        try:
            selected_id = self.idOption.currentText()
        except Exception as e:
            print("[go_next] Failed to read idOption:", e)
            self.page_history.pop(); return

        camera_checked = self.cameraOption.isChecked()
        upload_checked = self.uploadOption.isChecked()
        pdf_checked    = self.uploadPDFOption.isChecked()

        if not (camera_checked or upload_checked or pdf_checked):
            QMessageBox.information(self, "Selection", "Please select camera, upload image, or upload PDF.")
            self.page_history.pop(); return

        # CHANGED: PDF — open dialog here, convert, then hand off to pdf_preview
        if pdf_checked:
            self.camera.stop_camera()
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF", "", "PDF Files (*.pdf)")
            if not file_paths:
                self.page_history.pop(); return
            pdf_path = file_paths[-1]
            if not os.path.exists(pdf_path):
                self.page_history.pop(); return
            pages = convert_pdf_pages(pdf_path)
            if not pages:
                QMessageBox.warning(self, "PDF Error",
                    "Could not convert the PDF.\n\nMake sure pdf2image and poppler are installed.")
                self.page_history.pop(); return
            self.pdf_preview.load_pdf(pages, pdf_path)
            return

        if selected_id not in ("Passport", "National ID", "Driver's License"):
            QMessageBox.information(self, "Selection", "Please select an ID type.")
            self.page_history.pop(); return

        if selected_id == "Passport":
            if camera_checked:
                self.Form1.setCurrentIndex(1)
                self.camera.start_camera()
            else:
                self.camera.stop_camera()
                self.Form1.setCurrentIndex(2)
                self.files.upload_image(self.uploadedImageView, None)
        elif selected_id in ["National ID", "Driver's License"]:
            if camera_checked:
                self.Form1.setCurrentIndex(4)
                self.camera.start_camera()
            else:
                self.camera.stop_camera()
                try:
                    self.frontImageView.clear()
                    self.backImageView.clear()
                except Exception:
                    pass
                self.Form1.setCurrentIndex(5)

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
        self.files.uploaded_files.clear()
        self.files.current_index = -1
        self.pendingResponse = None
        self.pendingDebugImage = None
        self.camera.stop_camera()
        for widget_name in [
            "uploadedImageView", "fileListWidget", "fileNameLabel",
            "fileSizeLabel", "fileStatusLabel", "cameraView",
            "cameraView1", "cameraView2", "frontImageView", "backImageView"
        ]:
            try:
                getattr(self, widget_name).clear()
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