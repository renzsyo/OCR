import sys, os
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import Qt
from IDscanner import CamHandler, FileManager, InferenceHandler, ReviewHandler, UiLoader

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.uploadedImageView = None
        self.backImageView = None
        self.cameraOption = None
        self.idOption = None
        self.Form1 = None
        self.uploadOption = None
        self.frontImageView = None
        self.front_file: dict | None = None
        self.back_file: dict | None = None
        self.pendingResponse: dict | None = None
        self.debug_mode: bool = False
        self.pendingDebugImage: str | None = None
        self.current_frame = None

        self.camera = CamHandler(self)
        self.files = FileManager(self)
        self.inference = InferenceHandler(self)
        self.review = ReviewHandler(self)

        UiLoader(self)

        self.page_flow: dict[int, int] = {
            1: 3,
            2: 3,
            4: 3,
            5: 3,
            3: 0,
            6: 0,
        }
        self.page_history: list[int] = []

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

        if current in self.page_flow:
            # Validate before advancing
            if current == 1 and not hasattr(self, "captured_frame"):
                QMessageBox.warning(self, "No Capture", "Please capture an image first.")
                self.page_history.pop()
                return
            if current == 1 and hasattr(self, "captured_frame"):
                if not self.inference.validate_passport_result_sync(getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            if current == 2 and not self.files.uploaded_files:
                QMessageBox.warning(self, "No file", "Please upload a file first.")
                self.page_history.pop()
                return
            if current == 2 and self.files.uploaded_files:
                if not self.inference.validate_passport_result_sync(getattr(self, "pendingResponse", {})):
                    self.page_history.pop()
                    return
            if current == 4 and (
                not hasattr(self, "captured_front_frame") or
                not hasattr(self, "captured_back_frame")
            ):
                QMessageBox.warning(self, "Missing Capture", "Please capture both front and back images first.")
                self.page_history.pop()
                return
            if current == 4:
                selected_id = self.idOption.currentText()
                if selected_id == "Driver's License":
                    if not self.inference.validate_driver_license_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop()
                        return
                if selected_id == "National ID":
                    if not self.inference.validate_national_id_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop()
                        return
            if current == 5 and (not self.front_file or not self.back_file):
                QMessageBox.warning(self, "Missing files", "Please upload both front and back images first.")
                self.page_history.pop()
                return
            if current == 5:
                selected_id = self.idOption.currentText()
                if selected_id == "Driver's License":
                    if not self.inference.validate_driver_license_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop()
                        return
                if selected_id == "National ID":
                    if not self.inference.validate_national_id_result_sync(getattr(self, "pendingResponse", {})):
                        self.page_history.pop()
                        return

            next_page = self.page_flow[current]
            self.Form1.setCurrentIndex(next_page)

            if next_page == 0:
                self.reset_session()
            elif next_page == 3:
                self.camera.stop_camera()
                self.review.show_review_page()
            return

        # --- Home page (page 0) routing ---
        try:
            selected_id = self.idOption.currentText()
        except Exception as e:
            print("[MainWindow/go_next] Failed to read idOption:", e)
            self.page_history.pop()
            return

        if not (self.uploadOption.isChecked() or self.cameraOption.isChecked()):
            QMessageBox.information(self, "Selection", "Select camera or upload.")
            self.page_history.pop()
            return

        if selected_id == "Passport":
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(1)
                self.camera.start_camera()
            else:
                self.camera.stop_camera()
                self.Form1.setCurrentIndex(2)
                self.files.upload_image(self.uploadedImageView, None)

        elif selected_id in ["National ID", "Driver's License"]:
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(4)
                self.camera.start_camera()
            else:
                self.camera.stop_camera()
                try:
                    self.frontImageView.clear()
                    self.backImageView.clear()
                except Exception as e:
                    print("[MainWindow/go_next] Failed to clear front/back image views:", e)

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
        except Exception as e :
            print("[MainWindow/reset_session] Failed to reset capture buttons:", e)

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
            except Exception as e:
                print(f"[reset_session] Could not clear {widget_name}: {e}")

    def on_debug_toggled(self, state: int) -> None:
        self.debug_mode = (state == 2)
        print ("[DEBUG mode] is now:", self.debug_mode)

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



