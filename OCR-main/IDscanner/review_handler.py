import os,cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel,
    QFileDialog, QMessageBox, QVBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

# Pre-import at module load time (happens on main thread before CUDA loads)
# so the supabase/httpx/asyncio init never races with CUDA on a worker thread.
try:
    from .db_handler import save_scan as _save_scan
    _SUPABASE_AVAILABLE = True
except Exception as _e:
    print(f"[ReviewHandler] Supabase import failed: {_e}")
    _SUPABASE_AVAILABLE = False

class ReviewHandler:
    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
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

        # Front/back files (all flows — passport upload, NID/DL upload, PDF)
        if p.front_file:
            self.add_file_tab(p.front_file, "Front Side")
        if p.back_file:
            self.add_file_tab(p.back_file, "Back Side")

        # Populate the shared resultbox from pendingResponse
        # CHANGED: use detected_id_type (auto-detected) instead of idOption
        selected_id = getattr(p, "detected_id_type", None) or getattr(p, "lastIdType", "Unknown")
        if p.pendingResponse is not None:
            formatted = p.inference.format_pending_response(p.pendingResponse, selected_id)
            p.resultbox.setPlainText(formatted)
            p.lastResult = p.pendingResponse  # ← saved before clearing
            p.lastIdType = selected_id
            p.pendingResponse = None
            print("[DEBUG] resultbox populated from pendingResponse")

            # ── Save to Supabase ──────────────────────────────────────
            print("[DEBUG] entering supabase block")
            if _SUPABASE_AVAILABLE:
                try:
                    method = getattr(p, "_last_method", "Unknown")
                    front_path = p.front_file.get("path") if p.front_file else None
                    back_path  = p.back_file.get("path")  if p.back_file  else None
                    if not front_path and hasattr(p, "captured_frame"):
                        front_path = getattr(p, "_captured_front_save_path", None)
                    if not front_path and hasattr(p, "captured_front_frame"):
                        front_path = getattr(p, "_captured_front_save_path", None)
                    if not back_path and hasattr(p, "captured_back_frame"):
                        back_path = getattr(p, "_captured_back_save_path", None)

                    debug_path = getattr(p, "pendingDebugImage", None)
                    if not debug_path:
                        for fname in ("debug_passport.png", "debug_license.png",
                                      "debug_national_id_front.png"):
                            if os.path.exists(fname):
                                debug_path = fname
                                break
                    back_debug_path = getattr(p, "pendingDebugImageBack", None)
                    if not back_debug_path and os.path.exists("debug_national_id_back.png"):
                        back_debug_path = "debug_national_id_back.png"
                    print(f"[DEBUG] calling _save_scan front={front_path} debug={debug_path} back_debug={back_debug_path}")
                    gradcam_path = getattr(p, "_gradcam_path", None)
                    gradcam_back_path = getattr(p, "_gradcam_path_back", None)
                    _save_scan(
                        id_type     = selected_id,
                        method      = method,
                        result_text = formatted,
                        front_path  = front_path,
                        back_path   = back_path,
                        debug_path  = debug_path,
                        back_debug_path=back_debug_path,
                        gradcam_path = gradcam_path,
                        gradcam_back_path = gradcam_back_path,
                    )
                    print("[DEBUG] _save_scan returned")
                except Exception as e:
                    print(f"[DB] save_scan error: {e}")
            else:
                print("[DEBUG] supabase not available, skipping")


        selected_id_for_tab = getattr(p, "lastIdType", None) or getattr(p, "detected_id_type", None)
        if hasattr(p, "captured_frame") and selected_id_for_tab not in ("National ID", "UMID"):
            tab = ReviewHandler.frame_to_tab(p.captured_frame)
            p.reviewTabWidget.addTab(tab, "Captured Image")

        # Front/back captured frames (NID or DL camera flow)
        for frame_attr, tab_label in (("captured_front_frame", "Front Capture"),
                                      ("captured_back_frame", "Back Capture")):
            if not hasattr(p, frame_attr):
                continue
            tab = ReviewHandler.frame_to_tab(getattr(p, frame_attr))
            p.reviewTabWidget.addTab(tab, tab_label)

        # Load debug and Grad-CAM images into memory NOW — before the Supabase
        # background thread (delete_after=True) can delete the files from disk.
        # Tab widgets are built from the in-memory arrays, so file deletion
        # timing no longer causes a crash.
            # ── PDF Debug tab (only when debug_mode is on and method was PDF) ──
        pdf_debug = getattr(p, "_pdf_debug_info", None)
        if getattr(p, "debug_mode", False) and pdf_debug:
            p._pdf_debug_info = None  # clear so it doesn't bleed into next session
            try:
                lines = [
                    f"PDF DEBUG INFORMATION",
                    f"{'─' * 35}",
                    f"  ID Type Detected : {pdf_debug.get('id_type', 'N/A')}",
                    f"  Total Pages      : {pdf_debug.get('page_count', 'N/A')}",
                    f"  Detected on Page : {pdf_debug.get('detected_page', 'N/A')}",
                    f"  Front Assigned   : Page {pdf_debug.get('front_page', 'N/A')}",
                    f"  Back Assigned    : Page {pdf_debug.get('back_page', 'N/A')}",
                    f"",
                    f"RAW SCAN RESULT",
                    f"{'─' * 35}",
                ]
                import json
                raw = pdf_debug.get("raw_result", {})
                lines.append(json.dumps(raw, indent=2, default=str))

                debug_text = "\n".join(lines)

                from PyQt6.QtWidgets import QTextEdit
                debug_widget = QWidget()
                layout = QVBoxLayout(debug_widget)
                text_box = QTextEdit()
                text_box.setReadOnly(True)
                text_box.setPlainText(debug_text)
                text_box.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
                layout.addWidget(text_box)
                p.reviewTabWidget.addTab(debug_widget, "PDF Debug Info")
            except Exception as e:
                print(f"[ReviewHandler] PDF debug tab error: {e}")

        debug_frame   = None
        debug_frame_back = None
        gradcam_frame = None
        gradcam_frame_back = None
        clf_result_tab = None

        if getattr(p, "debug_mode", False) and getattr(p, "pendingDebugImage", None):
            _dbg_path = p.pendingDebugImage
            p.pendingDebugImage = None          # clear immediately
            debug_frame = cv2.imread(_dbg_path) # None if already deleted — handled below

        if getattr(p, "debug_mode", False) and getattr(p, "pendingDebugImageBack", None):
            _dbg_back_path = p.pendingDebugImageBack
            p.pendingDebugImageBack = None
            debug_frame_back = cv2.imread(_dbg_back_path)

        _gradcam_path = getattr(p, "_gradcam_path", None)
        if getattr(p, "debug_mode", False) and _gradcam_path and os.path.exists(_gradcam_path):
            gradcam_frame  = cv2.imread(_gradcam_path)
            clf_result_tab = getattr(p, "_classifier_result", None)
            p._gradcam_path      = None
            p._classifier_result = None

        _gradcam_path_back = getattr(p, "_gradcam_path_back", None)
        if getattr(p, "debug_mode", False) and _gradcam_path_back and os.path.exists(_gradcam_path_back):
            gradcam_frame_back = cv2.imread(_gradcam_path_back)
        p._gradcam_path_back = None

        # Always clear to prevent stale data bleeding into next session
        p._gradcam_path      = None
        p._classifier_result = None

        # Build tabs from already-loaded frames
        if debug_frame is not None:
            tab = ReviewHandler.frame_to_tab(debug_frame)
            p.reviewTabWidget.addTab(tab, "Debug - Front")

        if debug_frame_back is not None:
            tab = ReviewHandler.frame_to_tab(debug_frame_back)
            p.reviewTabWidget.addTab(tab, "Debug - Back")

        if gradcam_frame is not None:
            if clf_result_tab is not None:
                label = f"{clf_result_tab.class_name.upper()}  {clf_result_tab.confidence:.1%}"
                cv2.putText(gradcam_frame, label, (10, gradcam_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(gradcam_frame, label, (10, gradcam_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            tab = ReviewHandler.frame_to_tab(gradcam_frame)
            p.reviewTabWidget.addTab(tab, "Grad-CAM - Front")

        if gradcam_frame_back is not None:
            tab = ReviewHandler.frame_to_tab(gradcam_frame_back)
            p.reviewTabWidget.addTab(tab, "Grad-CAM - Back")

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

            elif id_type == "PhilHealth":
                data = result.get("parsed", {}).get("PhilHealth/OCR", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "Name").text = data.get("name", "N/A")
                ET.SubElement(personal, "Sex").text = data.get("sex", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("date_of_birth", "N/A")
                ET.SubElement(personal, "Address").text = data.get("address", "N/A")
                details = ET.SubElement(root, "IDDetails")
                ET.SubElement(details, "PhilHealthNo").text = data.get("philhealth_id_number", "N/A")

            elif id_type == "TIN":
                data = result.get("parsed", {}).get("TIN/OCR", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "Name").text = data.get("name", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("date_of_birth", "N/A")
                ET.SubElement(personal, "Address").text = data.get("address", "N/A")
                details = ET.SubElement(root, "IDDetails")
                ET.SubElement(details, "TIN").text = data.get("tin", "N/A")
                ET.SubElement(details, "DateIssued").text = data.get("date_issued", "N/A")

            elif id_type == "SSS":
                data = result.get("parsed", {}).get("SSS/OCR", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "Name").text = data.get("name", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("date_of_birth", "N/A")
                details = ET.SubElement(root, "IDDetails")
                ET.SubElement(details, "SSSNumber").text = data.get("sss_number", "N/A")

            elif id_type == "Senior Citizen":
                data = result.get("parsed", {}).get("SeniorCitizen/OCR", {}) or {}
                personal = ET.SubElement(root, "PersonalInformation")
                ET.SubElement(personal, "Name").text = data.get("name", "N/A")
                ET.SubElement(personal, "Birthday").text = data.get("date_of_birth", "N/A")
                ET.SubElement(personal, "Age").text = data.get("age", "N/A")
                ET.SubElement(personal, "Address").text = data.get("address", "N/A")
                details = ET.SubElement(root, "IDDetails")
                ET.SubElement(details, "IDNumber").text = data.get("id_number", "N/A")
                ET.SubElement(details, "DateIssued").text = data.get("date_of_issue", "N/A")
                ET.SubElement(details, "IssuingOffice").text = data.get("issuing_office", "N/A")

        except Exception as e:
            print(f"[ReviewHandler/format_as_xml] Error building XML: {e}")
            error_el = ET.SubElement(root, "Error")
            error_el.text = str(e)

        # Pretty print with indentation
        raw = ET.tostring(root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ")

        # minidom adds an <?xml?> header line, keep it
        return pretty