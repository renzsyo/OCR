import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

class ReviewHandler:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self._last_result: dict | None = None
        self._last_id_type: str | None = None
    @staticmethod
    def frame_to_tab(frame) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        pictureView = QLabel()
        pictureView.setFixedSize(512, 384)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        pictureView.setPixmap(pixmap.scaled(
            pictureView.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        layout.addWidget(pictureView)
        return tab

    def show_review_page(self) -> None:
        p = self.parent
        p.reviewTabWidget.clear()

        print("[REVIEW DEBUG] debug_mode:", getattr(p, "debug_mode", False))
        print("[REVIEW DEBUG] pendingDebugImage:", getattr(p, "pendingDebugImage", None))
        print("[REVIEW DEBUG] pendingResponse:", p.pendingResponse)

        # Front/back uploaded files (NID or DL upload flow)
        if p.front_file:
            self.add_file_tab(p.front_file, "Front Side")
        if p.back_file:
            self.add_file_tab(p.back_file, "Back Side")

        # Populate the shared resultbox from pendingResponse
        selected_id = p.idOption.currentText()  # ← always defined first
        if p.pendingResponse is not None:
            formatted = p.inference.format_pending_response(p.pendingResponse, selected_id)
            p.resultbox.setPlainText(formatted)
            p.lastResult = p.pendingResponse  # ← saved before clearing
            p.lastIdType = selected_id
            p.pendingResponse = None
            print("[DEBUG] resultbox populated from pendingResponse")

        # Single captured frame (passport camera flow)
        if hasattr(p, "captured_frame"):
            tab = ReviewHandler.frame_to_tab(p.captured_frame)
            p.reviewTabWidget.addTab(tab, "Captured Image")

        # Front/back captured frames (NID or DL camera flow)
        for frame_attr, tab_label in (("captured_front_frame", "Front Capture"),
                                      ("captured_back_frame", "Back Capture")):
            if not hasattr(p, frame_attr):
                continue
            tab = ReviewHandler.frame_to_tab(getattr(p, frame_attr))
            p.reviewTabWidget.addTab(tab, tab_label)

        # Uploaded files list (passport upload flow)
        for file in p.files.uploaded_files:
            path = file.get("path")
            frame = cv2.imread(path)
            if frame is None:
                continue
            tab = ReviewHandler.frame_to_tab(frame)
            p.reviewTabWidget.addTab(tab, file["name"])

        if getattr(p, "debug_mode", False) and getattr(p, "pendingDebugImage", None):
            debug_path = p.pendingDebugImage
            frame = cv2.imread(debug_path)
            if frame is not None:
                tab = ReviewHandler.frame_to_tab(frame)
                p.reviewTabWidget.addTab(tab, "Debug - Bounding Boxes")
            p.pendingDebugImage = None

    def add_file_tab(self, file_info: dict, tab_name: str) -> None:
        p = self.parent
        path = file_info.get("path")
        frame = cv2.imread(path)
        if frame is None:
            return

        tab = ReviewHandler.frame_to_tab(frame)
        p.reviewTabWidget.addTab(tab, tab_name)

    def download_text(self, text_box, default_name: str = "extracted_text") -> None:
        p = self.parent
        print("[DOWNLOAD DEBUG] lastResult:", getattr(p, "lastResult", "ATTRIBUTE MISSING"))
        print("[DOWNLOAD DEBUG] lastIdType:", getattr(p, "lastIdType", "ATTRIBUTE MISSING"))
        text = text_box.toPlainText()
        if not text.strip():
            QMessageBox.warning(p, "No text", "There is no text to save.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            p,
            "Save Extracted Data",
            default_name,
            "Text Files (*.txt);;XML Files (*.xml)"
        )
        if not path:
            return

        try:
            if "xml" in selected_filter.lower():
                # Ensure correct extension
                if not path.lower().endswith(".xml"):
                    path += ".xml"

                if p.lastResult and p.lastIdType:
                    xml_content = self.format_as_xml(p.lastResult, p.lastIdType)
                else:
                    # Fallback: wrap plain text in basic XML if result was lost
                    xml_content = (
                        '<?xml version="1.0" ?>\n'
                        '<ScanResult>\n'
                        f'  <RawText>{text}</RawText>\n'
                        '</ScanResult>\n'
                    )

                with open(path, "w", encoding="utf-8") as f:
                    f.write(xml_content)

            else:
                # Plain text — original behavior
                if not path.lower().endswith(".txt"):
                    path += ".txt"

                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

            QMessageBox.information(p, "Saved", f"File saved to:\n{path}")

        except Exception as e:
            QMessageBox.warning(p, "Error", f"Could not save file: {e}")

    def format_as_xml(self, result: dict, id_type: str) -> str:
        root = ET.Element("ScanResult")
        id_type_el = ET.SubElement(root, "IDType")
        id_type_el.text = id_type

        try:
            if id_type == "Passport":
                data = result.get("parsed", {}).get("Passport/MRZ", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "LastName").text = data.get("Surname", "N/A")
                ET.SubElement(personal, "FirstName").text = data.get("Given_names", "N/A")
                ET.SubElement(personal, "Sex").text = data.get("Sex", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("Birth_date", "N/A")
                ET.SubElement(personal, "Nationality").text = data.get("Nationality", "N/A")
                details = ET.SubElement(root, "PassportDetails")
                ET.SubElement(details, "DocumentNumber").text = data.get("Document_number", "N/A")
                ET.SubElement(details, "Country").text = data.get("Country", "N/A")
                ET.SubElement(details, "ExpiryDate").text = data.get("Expiry_date", "N/A")

            elif id_type == "Driver's License":
                data = result.get("parsed", {}).get("Driverslicense/OCR", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "Name").text = data.get("Name", "N/A")
                ET.SubElement(personal, "Sex").text = data.get("Sex", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("Birthdate", "N/A")
                ET.SubElement(personal, "Address").text = data.get("Address", "N/A")
                details = ET.SubElement(root, "LicenseDetails")
                ET.SubElement(details, "LicenseNo").text = data.get("License No", "N/A")
                ET.SubElement(details, "ExpirationDate").text = data.get("Expiration Date", "N/A")

            elif id_type == "National ID":
                subject = result.get("qr", {}).get("NationalID/QR", {}).get("subject", {}) or {}
                qr_data = result.get("qr", {}).get("NationalID/QR", {}) or {}
                front = result.get("front", {}).get("parsed", {}).get("NationalID/Front", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "LastName").text = subject.get("lName", "N/A")
                ET.SubElement(personal, "FirstName").text = subject.get("fName", "N/A")
                ET.SubElement(personal, "MiddleName").text = subject.get("mName", "N/A")
                ET.SubElement(personal, "Suffix").text = subject.get("Suffix", "N/A") or "None"
                ET.SubElement(personal, "Sex").text = subject.get("sex", "N/A")
                ET.SubElement(personal, "Birthday").text = subject.get("DOB", "N/A")
                ET.SubElement(personal, "Birthplace").text = subject.get("POB", "N/A")
                ET.SubElement(personal, "Address").text = front.get("Address", "N/A")
                details = ET.SubElement(root, "IDDetails")
                ET.SubElement(details, "PCN").text = subject.get("PCN", "N/A")
                ET.SubElement(details, "Issuer").text = qr_data.get("Issuer", "N/A")
                ET.SubElement(details, "DateIssued").text = qr_data.get("DateIssued", "N/A")

        except Exception as e:
            print(f"[ReviewHandler/format_as_xml] Error building XML: {e}")
            error_el = ET.SubElement(root, "Error")
            error_el.text = str(e)

        # Pretty print with indentation
        raw = ET.tostring(root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ")

        # minidom adds an <?xml?> header line, keep it
        return pretty
