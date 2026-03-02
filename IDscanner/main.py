import sys, os, time, threading, cv2, requests
from PyQt6 import uic
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
            self.continuep2.clicked.connect(self.go_next)
            self.continuep3.clicked.connect(self.infer_and_continue)
            self.continuep4.clicked.connect(self.go_next)
            self.continuep5.clicked.connect(self.go_next)
            self.continuep6.clicked.connect(self.go_next)
            self.backButtonp1.clicked.connect(self.go_back)
            self.backButtonp2.clicked.connect(self.go_back)
            self.backButtonp3.clicked.connect(self.go_back)
            self.backButtonp4.clicked.connect(self.go_back)
            self.backButtonp5.clicked.connect(self.go_back)
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
        except Exception:
            pass

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
        # Safe camera read and QImage creation
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        try:
            # Keep a copy for capture
            self.current_frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        save_path = f"captured_id_{int(time.time())}.jpg"
        try:
            cv2.imwrite(save_path, self.captured_frame)
        except Exception as e:
            print("Failed to save captured image:", e)
            return

        # Start background thread to send OCR request (non-blocking)
        threading.Thread(target=self._send_ocr_request, args=(save_path,), daemon=True).start()

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

            save_path = f"{frame_attr}_{int(time.time())}.jpg"
            cv2.imwrite(save_path, frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            threading.Thread(target=self._send_ocr_request, args=(save_path,), daemon=True).start()

    def infer_and_continue(self):
        """
        Called when continuep3 is clicked.
        1) Send inference request (upload page)
        2) Then go to next page
        """
        # Ensure there is at least one uploaded file
        if not self.uploaded_files:
            QMessageBox.warning(self, "No file", "Please upload an image first.")
            return

        # Use current selection or last file
        if self.current_index < 0:
            self.current_index = len(self.uploaded_files) - 1

        if self.current_index < 0:
            QMessageBox.warning(self, "No file", "Please upload an image first.")
            return

        file_info = self.uploaded_files[self.current_index]
        path = file_info.get("path")

        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Selected file not found.")
            return

        # Debug
        print(f"[DEBUG] Sending inference for: {path}")

        # Start inference in background (non-blocking)
        threading.Thread(target=self._send_ocr_request, args=(path,), daemon=True).start()

        # After triggering inference, go to next page
        self.go_next()

    def _send_ocr_request(self, image_path):
        try:
            # Determine route (same as before)
            selected_id = self.idOption.currentText()
            print(f"[DEBUG] Selected ID option: '{selected_id}'")

            if selected_id == "Passport":
                route = "http://127.0.0.1:5000/pP"
            elif selected_id == "National ID":
                route = "http://127.0.0.1:5000/nID"
            elif selected_id == "Driver's License":
                route = "http://127.0.0.1:5000/dL"
            else:
                print("[DEBUG] Unknown ID option")
                return

            print(f"[DEBUG] Using route: {route}")

            with open(image_path, "rb") as f:
                files = {"file": f}
                resp = requests.post(route, files=files, timeout=15)

            print(f"[DEBUG] HTTP Status: {resp.status_code}")

            try:
                data = resp.json()
            except Exception:
                data = {"response": resp.text}

            print("[DEBUG] Server response:", data)

            # Update UI from main thread (safe)
            QTimer.singleShot(0, lambda: print("[DEBUG] UI update triggered"))
            print("[DEBUG] Setting pendingResponse")
            textbox = self.findChild(QTextEdit, "extractedTextBox")
            if textbox:
                formatted_text = self.format_mrz_response(data)
                textbox.setPlainText(formatted_text)
                print("[DEBUG] Textbox updated successfully")
            else:
                print("[DEBUG] extractedTextBox not found in UI")
            print("[DEBUG] pendingResponse set:", self.pendingResponse)

        except Exception as e:
            print("[DEBUG] OCR request failed:", e)

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

            file_info = {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": f"{os.path.getsize(file_path) / (1024 * 1024):.2f} MB",
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

            frame = cv2.imread(file_path)
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

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Extracted Text",
            f"{default_name}.txt",
            "Text Files (*.txt)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Saved", "Text saved to device.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save text file: {e}")

    def show_review_page(self):
        self.reviewTabWidget.clear()

        if self.front_file:
            self._add_file_tab(self.front_file, "Front Side")
        if self.back_file:
            self._add_file_tab(self.back_file, "Back Side")

        # --- Add captured frame as a tab if available ---
        if hasattr(self, "captured_frame"):
            tab = QWidget()
            layout = QHBoxLayout(tab)

            # Image preview
            pictureView = QLabel()
            pictureView.setFixedSize(512, 384)
            rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
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

            extractedTextBox = QTextEdit()
            extractedTextBox.setFixedSize(191, 271)
            extractedTextBox.setReadOnly(True)
            print("[DEBUG] extractedTextBox created for tab")
            # Store reference on tab
            tab.extractedTextBox = extractedTextBox

            # If pending response exists, display it
            if hasattr(self, "pendingResponse") and self.pendingResponse is not None:
                if isinstance(self.pendingResponse, dict):
                    text = "\n".join(f"{k}: {v}" for k, v in self.pendingResponse.items())
                else:
                    text = str(self.pendingResponse)

                extractedTextBox.setPlainText(text)

                self.pendingResponse = None
            else:
                extractedTextBox.setPlainText("Sample extracted text will appear here.")
            # Download button (centered)
            downloadBtn = QPushButton("Download as File")
            downloadBtn.setFixedSize(121, 41)
            downloadBtn.clicked.connect(
                lambda _, tb=extractedTextBox: self.download_text(tb, "captured_image")
            )
            right_layout.addWidget(extractedTextBox)
            right_layout.addWidget(downloadBtn, alignment=Qt.AlignmentFlag.AlignHCenter)

            layout.addWidget(pictureView)
            layout.addLayout(right_layout)

            self.reviewTabWidget.addTab(tab, "Captured Image")

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
                pixmap.scaled(pictureView.size(),
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )

            right_layout = QVBoxLayout()
            extractedTextBox = QTextEdit()
            extractedTextBox.setFixedSize(191, 271)
            extractedTextBox.setReadOnly(True)
            extractedTextBox.setPlainText("Sample extracted text will appear here.")
            tab.extractedTextBox = extractedTextBox

            downloadBtn = QPushButton("Download as File")
            downloadBtn.setFixedSize(121, 41)
            downloadBtn.clicked.connect(
                lambda _, tb=extractedTextBox, name=tab_label: self.download_text(tb, name)
            )
            right_layout.addWidget(extractedTextBox)
            right_layout.addWidget(downloadBtn, alignment=Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(pictureView)
            layout.addLayout(right_layout)
            self.reviewTabWidget.addTab(tab, tab_label)

        # --- Add uploaded files as tabs ---
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
                pixmap.scaled(pictureView.size(),
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )

            right_layout = QVBoxLayout()
            extractedTextBox = QTextEdit()
            extractedTextBox.setFixedSize(191, 271)
            extractedTextBox.setReadOnly(True)
            print("[DEBUG] extractedTextBox created for tab2")
            # Store reference on tab
            tab.extractedTextBox = extractedTextBox

            # If pending response exists, display it
            if hasattr(self, "pendingResponse") and self.pendingResponse is not None:
                if isinstance(self.pendingResponse, dict):
                    text = "\n".join(f"{k}: {v}" for k, v in self.pendingResponse.items())
                else:
                    text = str(self.pendingResponse)

                extractedTextBox.setPlainText(text)

                # Clear pending (so it doesn't repeat)
                self.pendingResponse = None
            else:
                extractedTextBox.setPlainText("Sample extracted text will appear here.")

            downloadBtn = QPushButton("Download as File")
            downloadBtn.setFixedSize(121, 41)
            downloadBtn.clicked.connect(
                lambda _, tb=extractedTextBox, fname=file["name"]: self.download_text(tb, fname)
            )
            right_layout.addWidget(extractedTextBox)
            right_layout.addWidget(downloadBtn, alignment=Qt.AlignmentFlag.AlignHCenter)

            layout.addWidget(pictureView)
            layout.addLayout(right_layout)

            self.reviewTabWidget.addTab(tab, file["name"])

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

        extractedTextBox = QTextEdit()
        extractedTextBox.setFixedSize(191, 271)
        extractedTextBox.setReadOnly(True)
        extractedTextBox.setPlainText("Sample extracted text will appear here.")
        tab.extractedTextBox = extractedTextBox  # store reference

        downloadBtn = QPushButton("Download as File")
        downloadBtn.setFixedSize(121, 41)
        downloadBtn.clicked.connect(
            lambda _, tb=extractedTextBox, fname=file_info["name"]: self.download_text(tb, fname)
        )

        right_layout.addWidget(extractedTextBox)
        right_layout.addWidget(downloadBtn, alignment=Qt.AlignmentFlag.AlignHCenter)

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