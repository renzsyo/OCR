import sys, os, time, threading, cv2, shutil
from PyQt6 import uic
from inference import scan_passport, scan_national_id, scan_driver_license
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMenu,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QPushButton,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("IDscanner.ui", self)

        try:
            self.Form1.setCurrentIndex(0)
        except Exception:
            # If your UI doesn't have Form1 or index 0, ignore here
            pass

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Timer for camera frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.pendingResponse = None
        self.reviewTextBox = None
        # Data storage
        self.uploaded_files = []  # list of dicts: {path, name, size, status}
        self.current_index = -1

        try:
            self.continuep1.clicked.connect(self.go_next)
            self.continuep2.clicked.connect(self.infer_page2_camera_passport)
            self.continuep3.clicked.connect(self.infer_page3_upload_passport)
            self.continuep4.clicked.connect(self.go_next)
            self.continuep5.clicked.connect(self.infer_page5)
            self.continuep6.clicked.connect(self.infer_page6)
            self.backButtonp1.clicked.connect(self.go_back)
            self.backButtonp2.clicked.connect(self.go_back)
            self.backButtonp3.clicked.connect(self.go_back)
            self.backButtonp4.clicked.connect(self.go_back)
            self.backButtonp5.clicked.connect(self.go_back)
            self.debugOption.stateChanged.connect(self.on_debug_toggled)
            self.captureButtonp1.clicked.connect(self.capture_image)
            self.recaptureButtonp1.clicked.connect(self.recapture_image)
            self.captureButtonp2.clicked.connect(
                lambda : self.toggle_capture("captured_front_frame", self.cameraView1, self.captureButtonp2)
            )
            self.captureButtonp3.clicked.connect(
                lambda: self.toggle_capture("captured_back_frame", self.cameraView2, self.captureButtonp3)
            )
            self.uploadButtonp3.clicked.connect(
                lambda: self.upload_image(self.uploadedImageView)
            )
            self.downloadp4.clicked.connect(
                lambda: self.download_text(self.resultbox, "extracted_text")
            )
        except Exception as e:
            print("[INIT ERROR]", e)

        try:
            # Use currentRowChanged so keyboard navigation also updates preview
            self.fileListWidget.currentRowChanged.connect(self.on_current_row_changed)
            # Also keep itemClicked for mouse clicks (optional)
            self.fileListWidget.itemClicked.connect(self.list_item_clicked)

            # Right click context menu
            self.fileListWidget.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            self.fileListWidget.customContextMenuRequested.connect(self.show_list_menu)
        except Exception:
            pass

        self.page_flow = {
            1: 3,
            2: 3,
            4: 3,
            5: 3,
            3: 0,
            6: 0,
        }
        self.page_history = []

        self.front_file = None
        self.back_file = None
        try:
            self.uploadFrontButton.clicked.connect(lambda: self.upload_image(self.frontImageView, side="front"))
            self.uploadBackButton.clicked.connect(lambda: self.upload_image(self.backImageView, side="back"))
        except Exception:
            pass

    def on_debug_toggled(self, state):
        print("[DEBUG TOGGLE] called with state:", state)
        self.debug_mode = (state == 2)  # 2 = Checked in Qt
        print("[DEBUG MODE] is now:", self.debug_mode)
    def _run_inference_national_id(self, image):
        result = scan_national_id(image)
        self.pendingResponse = result
        formatted = self._format_pending_response(result, "National ID")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))
        print(result)

    def _run_inference_passport(self, path):
        debug = getattr(self, "debug_mode", False)
        print("[INFERENCE] debug mode:", debug)
        result = scan_passport(path, debug=debug)
        self.pendingResponse = result
        self.pendingDebugImage = result.get("debug_image")
        print("[INFERENCE] pendingDebugImage:", self.pendingDebugImage)
        formatted = self._format_pending_response(result, "Passport")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))

    def _run_inference_driver_license(self, path):
        debug = getattr(self, "debug_mode", False)
        result = scan_driver_license(path, debug=debug)
        self.pendingResponse = result
        self.pendingDebugImage = result.get("debug_image")
        formatted = self._format_pending_response(result, "Driver's License")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))
        print(result)

    def infer_page2_camera_passport(self):
        print("[DEBUG] infer_page2_camera_passport called")
        if not hasattr(self, "captured_frame"):
            QMessageBox.warning(self, "No Capture", "Please capture an image first.")
            return

        path = "temp_passport_camera.jpg"
        cv2.imwrite(path, self.captured_frame)

        def task():
            self._run_inference_passport(path)

            # after inference finishes, navigate on UI thread
            QTimer.singleShot(0, self.go_next)

        threading.Thread(target=task, daemon=True).start()

    def infer_page3_upload_passport(self):
        if not self.uploaded_files:
            QMessageBox.warning(self, "No file", "Please upload an image first.")
            return

        if self.current_index < 0:
            self.current_index = len(self.uploaded_files) - 1

        if self.current_index < 0:
            QMessageBox.warning(self, "No file", "Please upload an image first.")
            return

        path = self.uploaded_files[self.current_index].get("path")
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Selected file not found.")
            return

        def task():
            self._run_inference_passport(path)
            QTimer.singleShot(0, self.go_next)

        threading.Thread(target=task, daemon=True).start()

    def infer_page5(self):
        selected_id = self.idOption.currentText()

        if selected_id == "National ID":
            if not hasattr(self, "captured_back_frame") or self.captured_back_frame is None:
                QMessageBox.warning(self, "Missing Capture", "Please capture the back image.")
                return

            def task():
                self._run_inference_national_id(self.captured_back_frame)
                QTimer.singleShot(0, self.go_next)  #

            threading.Thread(target=task, daemon=True).start()

        elif selected_id == "Driver's License":
            if not hasattr(self, "captured_front_frame") or self.captured_front_frame is None:
                QMessageBox.warning(self, "Missing Capture", "Please capture the front image.")
                return

            temp_path = f"front_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, self.captured_front_frame)

            def task():
                self._run_inference_driver_license(temp_path)
                QTimer.singleShot(0, self.go_next)  # ✅ after inference

            threading.Thread(target=task, daemon=True).start()

        else:
            QMessageBox.warning(self, "Unknown ID", "Please select a valid ID type.")

    def infer_page6(self):
        selected_id = self.idOption.currentText()

        if not hasattr(self, "front_file") or not self.front_file:
            QMessageBox.warning(self, "Missing Front Image", "Please upload the front image.")
            return

        if not hasattr(self, "back_file") or not self.back_file:
            QMessageBox.warning(self, "Missing Back Image", "Please upload the back image.")
            return

        if selected_id == "National ID":
            image = cv2.imread(self.back_file["path"])

            def task():
                self._run_inference_national_id(image)
                QTimer.singleShot(0, self.go_next)  # ✅ after inference

            threading.Thread(target=task, daemon=True).start()

        elif selected_id == "Driver's License":
            image = cv2.imread(self.front_file["path"])

            def task():
                self._run_inference_driver_license(image)
                QTimer.singleShot(0, self.go_next)  # ✅ after inference

            threading.Thread(target=task, daemon=True).start()

        else:
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid ID type.")

    def _format_pending_response(self, result, id_type):
        print(f"[FORMAT DEBUG] id_type='{id_type}', result type={type(result)}, result={result}")
        try:
            if id_type == "National ID":
                data = result.get("NationalID/QR", {})
                subject = data.get("subject", {})
                return (

                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Last Name   : {subject.get('lName', 'N/A')}\n"
                    f"  First Name  : {subject.get('fName', 'N/A')}\n"
                    f"  Middle Name : {subject.get('mName', 'N/A')}\n"
                    f"  Suffix      : {subject.get('Suffix', 'N/A') or 'None'}\n"
                    f"  Sex         : {subject.get('sex', 'N/A')}\n"
                    f"  Birthday    : {subject.get('DOB', 'N/A')}\n"
                    f"  Birthplace  : {subject.get('POB', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  PCN         : {subject.get('PCN', 'N/A')}\n"
                    f"  Issuer      : {data.get('Issuer', 'N/A')}\n"
                    f"  Date Issued : {data.get('DateIssued', 'N/A')}\n\n"

                )

            elif id_type == "Driver's License":
                data = result.get("parsed", {}).get("Driverslicense/OCR", {})
                return (
                    f"PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Name        : {data.get('Name', 'N/A')}\n"
                    f"  Sex         : {data.get('Sex', 'N/A')}\n"
                    f"  Birthday    : {data.get('Birthdate', 'N/A')}\n"
                    f"  Address     : {data.get('Address', 'N/A')}\n\n"
                    f"  LICENSE DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  License No  : {data.get('License No', 'N/A')}\n"
                    f"  Expiration  : {data.get('Expiration Date', 'N/A')}\n\n"

                )

            elif id_type == "Passport":
                data = result.get("parsed", {}).get("Passport/MRZ", {})
                return (

                    f" PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Last Name   : {data.get('Surname', 'N/A')}\n"
                    f"  First Name  : {data.get('Given_names', 'N/A')}\n"
                    f"  Sex         : {data.get('Sex', 'N/A')}\n"
                    f"  Birthday    : {data.get('Birth_date', 'N/A')}\n\n"
                    f"  PASSPORT DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  Document No : {data.get('Document_number', 'N/A')}\n"
                    f"  Nationality : {data.get('Nationality', 'N/A')}\n"
                    f"  Country     : {data.get('Country', 'N/A')}\n"
                    f"  Expiry Date : {data.get('Expiry_date', 'N/A')}\n\n"
                )

        except Exception as e:
            return f"⚠️ Could not format result: {e}\n\nRaw output:\n{result}"

    def _update_extracted_text(self, text):
        self.extractedText.setText(text) # or whatever your text widget is

    def _get_output_folder(self, category, subfolder):
        base = "output"
        folder = os.path.join(base, category.strip(), subfolder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def start_camera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(30)

    def stop_camera(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        try:
            self.current_frame = frame.copy()

            # Draw guide rectangle on display frame (not on captured frame)
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # ID card guide box — centered, proportional to ID card aspect ratio
            box_w = int(w * 0.75)
            box_h = int(box_w / 1.586)  # standard ID aspect ratio 85.6mm x 53.98mm
            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Align ID within the box", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            if qimg.isNull():
                return
            pixmap = QPixmap.fromImage(qimg)

            view_to_attr = {
                "cameraView": None,
                "cameraView1": "captured_front_frame",
                "cameraView2": "captured_back_frame",
            }
            for view_name, frozen_attr in view_to_attr.items():
                view = getattr(self, view_name, None)
                if view is None:
                    continue
                if frozen_attr and hasattr(self, frozen_attr):
                    continue
                scaled = pixmap.scaled(
                    view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                view.setPixmap(scaled)
        except Exception as e:
            print("update_frame error:", e)

    def capture_image(self):
        print("[DEBUG] capture_image called")
        print("[DEBUG] has current_frame:", hasattr(self, "current_frame"))
        # Capture current frame and send to OCR in background
        if not hasattr(self, "current_frame"):
            return

        # Stop timer so preview shows captured frame
        try:
            self.timer.stop()
        except Exception:
            pass

        self.captured_frame = self.current_frame.copy()

        # Save with unique filename to avoid overwriting
        selected_id = self.idOption.currentText()
        folder = self._get_output_folder(selected_id, "Capture")
        filename = f"{int(time.time())}.jpg"
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, self.captured_frame)
        print("Saved capture to:", save_path)

        # Start background thread to send OCR request (non-blocking)

        # Show captured image in cameraView
        try:
            rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            if not qimg.isNull() and hasattr(self, "cameraView"):
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self.cameraView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.cameraView.setPixmap(scaled)
        except Exception as e:
            print("capture_image display error:", e)

    def toggle_capture(self, frame_attr, display_label, button):
        print("[DEBUG] toggle_capture called, frame_attr:", frame_attr)
        print("[DEBUG] has current_frame:", hasattr(self, "current_frame"))
        selected_id = self.idOption.currentText()
        if hasattr(self,frame_attr):
            delattr(self, frame_attr)
            button.setText("Capture Image")
            display_label.clear()
            self.start_camera()
        else:
            if not hasattr(self, "current_frame"):
                return
            frame = self.current_frame.copy()
            setattr(self, frame_attr, frame)
            button.setText("Recapture Image")

            folder = self._get_output_folder(selected_id, "Capture")
            filename = f"{frame_attr}_{int(time.time())}.jpg"
            save_path = os.path.join(folder, filename)

            cv2.imwrite(save_path, frame)
            print("Saved to:", save_path)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))



    def show_response_in_listwidget(self, data):
        currentTab = self.reviewTabWidget.currentWidget()

        if not hasattr(currentTab, "extractedTextBox"):
            print("[DEBUG] extractedTextBox not found on current tab")
            return

        box = currentTab.extractedTextBox

        if isinstance(data, dict):
            text = "\n".join(f"{k}: {v}" for k, v in data.items())
        else:
            text = str(data)

        box.setPlainText(text)
        print("[DEBUG] Response written to extractedTextBox")

    def recapture_image(self):
        if hasattr(self, "captured_frame"):
            del self.captured_frame

        self.start_camera()

    def upload_image(self, target_label, side=None):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ID Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_paths:
            return

        try:
            file_path = file_paths[-1]
            if not os.path.exists(file_path):
                return

            selected_id = self.idOption.currentText()
            folder = self._get_output_folder(selected_id, "Upload")

            filename = os.path.basename(file_path)
            dest = os.path.join(folder, filename)

            try:
                shutil.copyfile(file_path, dest)
                print("Upload copied to:", dest)
            except Exception as e:
                print("Copy failed:", e)
                QMessageBox.warning(self, "Copy Error", str(e))
                return

            file_info = {
                "path": dest,
                "name": filename,
                "size": f"{os.path.getsize(dest) / (1024 * 1024):.2f} MB",
                "status": "Completed",
                "side": side,
            }

            if side == "front":
                self.front_file = file_info
            elif side == "back":
                self.back_file = file_info
            else:
                self.uploaded_files.append(file_info)
                self.current_index = len(self.uploaded_files) - 1
                self.refresh_file_list()
                if self.current_index >= 0:
                    self.fileListWidget.setCurrentRow(self.current_index)

            frame = cv2.imread(dest)
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                target_label.setPixmap(
                    pixmap.scaled(
                        target_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

        except Exception as e:
            print("Upload failed:", e)
            QMessageBox.warning(self, "Upload Error", f"Upload failed: {e}")
    def refresh_file_list(self):
        try:
            self.fileListWidget.clear()
            for file in self.uploaded_files:
                self.fileListWidget.addItem(
                    f"{file['name']} | {file['size']} | {file['status']}"
                )
            if 0 <= self.current_index < len(self.uploaded_files):
                self.fileListWidget.setCurrentRow(self.current_index)
            else:
                # no valid selection
                self.fileListWidget.setCurrentRow(-1)
        except Exception as e:
            print("refresh_file_list error:", e)

    def list_item_clicked(self, item):
        # Mouse click handler
        row = self.fileListWidget.row(item)
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(self.uploadedImageView)

    def on_current_row_changed(self, row):
        # Keyboard or programmatic selection change
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(self.uploadedImageView)
        else:
            # clear preview if selection invalid
            self.current_index = -1
            self.uploadedImageView.clear()
            try:
                self.fileNameLabel.clear()
                self.fileSizeLabel.clear()
                self.fileStatusLabel.clear()
            except Exception:
                pass

    def show_list_menu(self, position):
        # position is widget-local coordinates
        item = self.fileListWidget.itemAt(position)
        menu = QMenu(self)

        if item is None:
            # Clicked empty area: optionally show actions like "Add files"
            add_action = menu.addAction("Add files")
            action = menu.exec(self.fileListWidget.mapToGlobal(position))
            if action == add_action:
                self.upload_image(self.uploadedImageView)
            return

        # If an item exists under cursor, select it and show delete
        row = self.fileListWidget.row(item)
        if row < 0 or row >= len(self.uploaded_files):
            return

        # Select the item under cursor so delete uses correct index
        self.fileListWidget.setCurrentRow(row)
        self.current_index = row

        delete_action = menu.addAction("Delete")
        action = menu.exec(self.fileListWidget.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected_file()

    def delete_selected_file(self):
        row = self.fileListWidget.currentRow()
        if row < 0 or row >= len(self.uploaded_files):
            return

        try:
            # Remove the file entry
            removed = self.uploaded_files.pop(row)
            print("Removed:", removed["name"])
        except Exception as e:
            print("delete_selected_file error:", e)
            return

        # Compute new current_index safely
        if len(self.uploaded_files) == 0:
            self.current_index = -1
        else:
            self.current_index = min(row, len(self.uploaded_files) - 1)

        self.refresh_file_list()

        if self.current_index >= 0:
            self.fileListWidget.setCurrentRow(self.current_index)
            self.display_file_details(self.uploadedImageView)
        else:
            try:
                self.uploadedImageView.clear()
                self.fileNameLabel.clear()
                self.fileSizeLabel.clear()
                self.fileStatusLabel.clear()
            except Exception:
                pass

    def display_file_details(self, target_label):
        # Validate index
        if self.current_index < 0 or self.current_index >= len(self.uploaded_files):
            # Clear UI to avoid stale content
            try:
                target_label.clear()
                self.fileNameLabel.clear()
                self.fileSizeLabel.clear()
                self.fileStatusLabel.clear()
            except Exception:
                pass
            return

        file_info = self.uploaded_files[self.current_index]
        path = file_info.get("path")
        if not path or not os.path.exists(path):
            print("display_file_details: missing file", path)
            try:
                target_label.clear()
                self.fileNameLabel.clear()
                self.fileSizeLabel.clear()
                self.fileStatusLabel.clear()
            except Exception:
                pass
            return

        frame = cv2.imread(path)
        if frame is None:
            print("display_file_details: cv2.imread returned None for", path)
            try:
                target_label.clear()
                self.fileNameLabel.clear()
                self.fileSizeLabel.clear()
                self.fileStatusLabel.clear()
            except Exception:
                pass
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            if qimg.isNull():
                print("display_file_details: QImage is null for", path)
                target_label.clear()
                return
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                target_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            target_label.setPixmap(scaled)

            try:
                self.fileNameLabel.setText(file_info["name"])
                self.fileSizeLabel.setText(file_info["size"])
                self.fileStatusLabel.setText(file_info["status"])
            except Exception:
                pass
        except Exception as e:
            print("display_file_details error:", e)
            try:
                target_label.clear()
            except Exception:
                pass

    def download_text(self, text_box, default_name="extracted_text"):
        text = text_box.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "No text", "There is no text to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Extracted Text",
            default_name + ".txt",
            "Text Files (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Saved", f"File saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save file: {e}")

    def show_review_page(self):
        self.reviewTabWidget.clear()

        if self.front_file:
            self._add_file_tab(self.front_file, "Front Side")
        if self.back_file:
            self._add_file_tab(self.back_file, "Back Side")
        if hasattr(self, "pendingResponse") and self.pendingResponse is not None:
            selected_id = self.idOption.currentText()
            formatted = self._format_pending_response(self.pendingResponse, selected_id)
            self.resultbox.setPlainText(formatted)
            self.pendingResponse = None
            print("[DEBUG] resultbox populated from pendingResponse")
        # --- Add captured frame as a tab if available ---
        if hasattr(self, "captured_frame"):
            tab = QWidget()
            layout = QHBoxLayout(tab)

            # Image preview only
            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(
                pixmap.scaled(
                    pictureView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            layout.addWidget(pictureView)
            self.reviewTabWidget.addTab(tab, "Captured Image")

        # --- Add captured front/back tabs (image only) ---
        for frame_attr, tab_label in (("captured_front_frame", "Front Capture"),
                                      ("captured_back_frame", "Back Capture")):
            if not hasattr(self, frame_attr):
                continue

            frame = getattr(self, frame_attr)
            tab = QWidget()
            layout = QHBoxLayout(tab)

            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(
                pixmap.scaled(
                    pictureView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            layout.addWidget(pictureView)
            self.reviewTabWidget.addTab(tab, tab_label)

        # --- Add uploaded files as image-only tabs ---
        # --- Add uploaded files as image-only tabs ---
        for file in self.uploaded_files:
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
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pictureView.setPixmap(
                pixmap.scaled(
                    pictureView.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            layout.addWidget(pictureView)
            self.reviewTabWidget.addTab(tab, file["name"])
        # --- Add debug bounding box image if debug mode is on ---
        if getattr(self, "debug_mode", False) and getattr(self, "pendingDebugImage", None):
            debug_path = self.pendingDebugImage
            frame = cv2.imread(debug_path)
            print("[REVIEW] debug_mode:", getattr(self, "debug_mode", False))
            print("[REVIEW] pendingDebugImage:", getattr(self, "pendingDebugImage", None))
            if frame is not None:
                tab = QWidget()
                layout = QHBoxLayout(tab)

                pictureView = QLabel()
                pictureView.setFixedSize(512, 384)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                pictureView.setPixmap(
                    pixmap.scaled(
                        pictureView.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
                layout.addWidget(pictureView)
                self.reviewTabWidget.addTab(tab, "Debug - Bounding Boxes")
            self.pendingDebugImage = None
    def _add_file_tab(self, file_info, tab_name):
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

        # Right side layout (text + button stacked vertically)
        right_layout = QVBoxLayout()
        layout.addWidget(pictureView)
        layout.addLayout(right_layout)

        self.reviewTabWidget.addTab(tab, tab_name)

    def reset_session(self):

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
        self.uploaded_files.clear()
        self.current_index = -1

        self.stop_camera()
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
        except Exception:
            pass

        try:
            self.pictureView1.clear()
            self.extractedTextBox.clear()
            self.uploadedImageView.clear()
            self.fileListWidget.clear()
            self.fileNameLabel.clear()
            self.fileSizeLabel.clear()
            self.fileStatusLabel.clear()
        except Exception:
            pass

    def go_back(self):
        if not self.page_history:
            if not self.page_history:
                return

        prev_page = self.page_history.pop()
        self.Form1.setCurrentIndex(prev_page)

    # Navigation Between Pages
    def go_next(self):
        self.page_history.append(self.Form1.currentIndex())
        current = self.Form1.currentIndex()

        if current in self.page_flow:
            if current == 1 and not hasattr(self, "captured_frame"):
                QMessageBox.warning(self, "No Capture", "Please capture an image first")
                return
            if current == 2 and not self.uploaded_files:
                QMessageBox.warning(self, "No file", "Please upload an file first")
                return
            if current == 4 and (not hasattr(self, "captured_front_frame") or not hasattr(self, "captured_back_frame")):
                QMessageBox.warning(self, "Missing Capture", "Please capture both front and back images first.")
                return
            if current == 5 and (not self.front_file or not self.back_file):
                QMessageBox.warning(self, "Missing files", "Please upload both front and back images first")
                return
            self.Form1.setCurrentIndex(self.page_flow[current])

            if self.page_flow[current] == 0:
                self.reset_session()

            if self.page_flow[current] == 3:
                self.stop_camera()
                self.show_review_page()
                self.pending_file = None
            return
        try:
            selected_id = self.idOption.currentText()
        except Exception:
            selected_id = None

        if not (getattr(self, "uploadOption", None) and getattr(self, "cameraOption", None)):
            return

        if not (self.uploadOption.isChecked() or self.cameraOption.isChecked()):
            QMessageBox.information(self, "Selection", "Select camera or upload")
            return

        if selected_id == "Passport":
            if self.cameraOption.isChecked():
                self.Form1.setCurrentIndex(1)
                self.start_camera()
            elif self.uploadOption.isChecked():
                self.stop_camera()
                self.Form1.setCurrentIndex(2)
                self.upload_image(self.uploadedImageView)

        elif selected_id in ["National ID", "Driver's License"]:
            if self.cameraOption.isChecked():
                try:
                    self.Form1.setCurrentIndex(4)
                except Exception:
                    pass
                self.start_camera()
            elif self.uploadOption.isChecked():
                self.stop_camera()
                try:
                    self.frontImageView.clear()
                    self.backImageView.clear()
                    self.Form1.setCurrentIndex(5)
                except Exception:
                    pass

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()