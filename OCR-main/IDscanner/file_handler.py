import os, shutil, cv2
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class FileManager:
    def __init__(self, parent):
        self.parent = parent
        self.uploaded_files = []
        self.current_index = -1

    def upload_image(self, target_label, side=None):
        p = self.parent
        file_paths, _ = QFileDialog.getOpenFileNames(
            p,
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

            selected_id = p.idOption.currentText()
            folder = p.get_output_folder(selected_id, "Upload")

            filename = os.path.basename(file_path)
            dest = os.path.join(folder, filename)

            try:
                shutil.copyfile(file_path, dest)
                print("Upload copied to:", dest)
            except Exception as e:
                print("Copy failed:", e)
                QMessageBox.warning(p, "Copy Error", str(e))
                return

            file_info = {
                "path": dest,
                "name": filename,
                "size": f"{os.path.getsize(dest) / (1024 * 1024):.2f} MB",
                "status": "Completed",
                "side": side,
            }

            if side == "front":
                p.front_file = file_info
            elif side == "back":
                p.back_file = file_info
            else:
                self.uploaded_files.append(file_info)
                self.current_index = len(self.uploaded_files) - 1
                self.refresh_file_list()
                if self.current_index >= 0:
                    p.fileListWidget.setCurrentRow(self.current_index)

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
            QMessageBox.warning(p, "Upload Error", f"Upload failed: {e}")

    def refresh_file_list(self):
        p = self.parent
        try:
            p.fileListWidget.clear()
            for file in self.uploaded_files:
                p.fileListWidget.addItem(
                    f"{file['name']} | {file['size']} | {file['status']}"
                )
            if 0 <= self.current_index < len(self.uploaded_files):
                p.fileListWidget.setCurrentRow(self.current_index)
            else:
                # no valid selection
                p.fileListWidget.setCurrentRow(-1)
        except Exception as e:
            print("refresh_file_list error:", e)

    def list_item_clicked(self, item):
        p = self.parent
        row = p.fileListWidget.row(item)
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(p.uploadedImageView)

    def on_current_row_changed(self, row):
        p = self.parent
        if 0 <= row < len(self.uploaded_files):
            self.current_index = row
            self.display_file_details(p.uploadedImageView)
        else:
            # clear preview if selection invalid
            self.current_index = -1
            p.uploadedImageView.clear()
            try:
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[FileManager/on_current_row_changed] Failed to clear file detail labels:", e)

    def show_list_menu(self, position):
        p = self.parent
        item = p.fileListWidget.itemAt(position)
        menu = QMenu(p)

        if item is None:
            # Clicked empty area: optionally show actions like "Add files"
            add_action = menu.addAction("Add files")
            action = menu.exec(p.fileListWidget.mapToGlobal(position))
            if action == add_action:
                self.upload_image(p.uploadedImageView)
            return

        # If an item exists under cursor, select it and show delete
        row = p.fileListWidget.row(item)
        if row < 0 or row >= len(self.uploaded_files):
            return

        # Select the item under cursor so delete uses correct index
        p.fileListWidget.setCurrentRow(row)
        self.current_index = row

        delete_action = menu.addAction("Delete")
        action = menu.exec(p.fileListWidget.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected_file()

    def delete_selected_file(self):
        p = self.parent
        row = p.fileListWidget.currentRow()
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
            p.fileListWidget.setCurrentRow(self.current_index)
            self.display_file_details(p.uploadedImageView)
        else:
            try:
                p.uploadedImageView.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[FileManager/delete_selected_file] Failed to clear UI after deletion:", e)

    def display_file_details(self, target_label):
        p = self.parent

        if self.current_index < 0 or self.current_index >= len(self.uploaded_files):
            # Clear UI to avoid stale content
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[FileManager/display_file_details] Failed to clear stale UI content:", e)
            return

        file_info = self.uploaded_files[self.current_index]
        path = file_info.get("path")
        if not path or not os.path.exists(path):
            print("display_file_details: missing file", path)
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[FileManager/display_file_details] Failed to clear UI for missing file:", e)

            return

        frame = cv2.imread(path)
        if frame is None:
            print("display_file_details: cv2.imread returned None for", path)
            try:
                target_label.clear()
                p.fileNameLabel.clear()
                p.fileSizeLabel.clear()
                p.fileStatusLabel.clear()
            except Exception as e:
                print("[FileManager/display_file_details] Failed to clear UI after imread failure:", e)
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
                p.fileNameLabel.setText(file_info["name"])
                p.fileSizeLabel.setText(file_info["size"])
                p.fileStatusLabel.setText(file_info["status"])
            except Exception as e:
                print("[FileManager/display_file_details] Failed to update file detail labels:", e)
        except Exception as e:
            print("[FileManager/display_file_details] Failed to render image preview:", e)
            try:
                target_label.clear()
            except Exception as e:
                print("[FileManager/display_file_details] Failed to clear target label after render error:", e)
