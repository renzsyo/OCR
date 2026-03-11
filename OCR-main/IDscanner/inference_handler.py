import cv2, time, threading, os
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from .inference import (
    scan_passport, scan_national_id, scan_driver_license,
    scan_national_id_front, classify_id_type,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

class InferenceHandler:

    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self.inference_complete = False
        self.dl_inference_complete = False
        self.ni_inference_complete = False  # ✅

        self.watch_timer = QTimer()
        self.watch_timer.setInterval(100)
        self.watch_timer.timeout.connect(self._check_inference_done)
        self.watch_timer.start()

    def _check_inference_done(self) -> None:
        p = self.parent

        if self.inference_complete:
            self.inference_complete = False
            p.continuep2.setEnabled(True)
            p.continuep3.setEnabled(True)

        if self.dl_inference_complete:
            self.dl_inference_complete = False
            p.continuep5.setEnabled(True)
            p.continuep6.setEnabled(True)

        if self.ni_inference_complete:
            self.ni_inference_complete = False
            p.continuep5.setEnabled(True)
            p.continuep6.setEnabled(True)

    def _classify_check(self, image: np.ndarray, expected_type: str) -> str:
        """
        Run the classifier on an image and return the ID type to use.
        - If the classifier is confident and agrees with the user selection → use it.
        - If the classifier is confident but DISAGREES → log a warning, auto-correct
          the dropdown, and return the classifier's answer.
        - If the classifier is not confident (or unavailable) → trust the user selection.
        """
        predicted, confidence = classify_id_type(image)
        if predicted is None:
            # Not confident enough or model unavailable — trust the user
            return expected_type
        if predicted != expected_type:
            print(f"[Classifier] WARNING: selected='{expected_type}' but "
                  f"model says '{predicted}' ({confidence:.2%}) — auto-correcting.")
            try:
                idx = self.parent.idOption.findText(predicted)
                if idx >= 0:
                    self.parent.idOption.setCurrentIndex(idx)
            except Exception:
                pass
            return predicted
        print(f"[Classifier] Confirmed: '{predicted}' ({confidence:.2%})")
        return predicted

    def run_inference_passport(self, path: np.ndarray | str) -> None:
        p = self.parent
        debug = getattr(p, "debug_mode", False)
        # Run classifier on the image to confirm it's a passport
        image = cv2.imread(path) if isinstance(path, str) else path
        if image is not None:
            self._classify_check(image, "Passport")
        result = scan_passport(path, debug=debug)
        p.pendingResponse = result
        p.pendingDebugImage = result.get("debug_image")
        print(result)
        self.inference_complete = True  # ✅ signal completion

    def run_inference_national_id(self, front_image: np.ndarray | str, back_image: np.ndarray | str) -> None:
        p = self.parent
        debug = getattr(p, "debug_mode", False)
        # Run classifier on the front image (face/text side) to confirm ID type
        front_arr = cv2.imread(front_image) if isinstance(front_image, str) else front_image
        if front_arr is not None:
            detected = self._classify_check(front_arr, "National ID")
            # If classifier says Driver's License, reroute automatically
            if detected == "Driver's License":
                print("[Classifier] Rerouting to Driver's License inference.")
                self.run_inference_driver_license(front_image)
                return

        qr_result = scan_national_id(back_image)
        print("[NationalID] QR result:", qr_result)

        front_result = scan_national_id_front(front_image, debug=debug)
        print("[NationalID] Front OCR result:", front_result)

        match_result = self.match_national_id(qr_result, front_result)
        print("[NationalID] Match result:", match_result)

        p.pendingResponse = {
            "qr": qr_result,
            "front": front_result,
            "match": match_result,
            "valid": qr_result.get("valid", False) and match_result.get("passed", False)
        }
        p.pendingDebugImage = front_result.get("debug_image")

        formatted = self.format_pending_response(p.pendingResponse, "National ID")
        print(p.pendingResponse)
        self.ni_inference_complete = True

    def run_inference_driver_license(self, path: np.ndarray | str) -> None:
        p = self.parent
        debug = getattr(p, "debug_mode", False)
        # Run classifier to confirm ID type
        image = cv2.imread(path) if isinstance(path, str) else path
        if image is not None:
            detected = self._classify_check(image, "Driver's License")
            # If classifier says National ID, reroute automatically
            if detected == "National ID":
                print("[Classifier] Rerouting to National ID inference.")
                self.run_inference_national_id(path, path)
                return
        result = scan_driver_license(path, debug=debug)
        p.pendingResponse = result
        p.pendingDebugImage = result.get("debug_image")
        formatted = self.format_pending_response(result, "Driver's License")
        print(result)
        self.dl_inference_complete = True  #

    def infer_page2_camera_passport(self) -> None:
        p = self.parent
        print("[DEBUG] infer_page2_camera_passport called")
        if not hasattr(p, "captured_frame"):
            QMessageBox.warning(p, "No Capture", "Please capture an image first.")
            return
        frame = p.captured_frame.copy()
        def task():
            self.run_inference_passport(frame)
        threading.Thread(target=task, daemon=True).start()

    def infer_page3_upload_passport(self) -> None:
        print("UPLOAD PASSPORT start")
        p = self.parent
        files = p.files.uploaded_files
        index = p.files.current_index
        print("UPLOAD PASSPORT file")
        if not files:
            QMessageBox.warning(p, "No file", "Please upload an image first.")
            return
        if index < 0:
            index = len(files) - 1

        path = files[index].get("path")
        if not path or not __import__("os").path.exists(path):
            QMessageBox.warning(p, "Error", "Selected file not found.")
            return
        print("UPLOAD PASSPORT path")
        def task():
            self.run_inference_passport(path)


        threading.Thread(target=task, daemon=True).start()

        print("UPLOAD PASSPORT end")

    def format_pending_response(self, result: dict, id_type: str) -> str:
        print(f"[FORMAT DEBUG] id_type='{id_type}', result type={type(result)}, result={result}")
        try:
            if id_type == "National ID":
                data = result.get("qr", {}).get("NationalID/QR") or {}
                subject = data.get("subject") or {}
                front_fields = result.get("front", {}).get("parsed", {}).get("NationalID/Front", {}) or {}
                # Fall back to front OCR fields when QR subject is empty
                return (
                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Last Name   : {subject.get('lName') or front_fields.get('Last Name', 'N/A')}\n"
                    f"  First Name  : {subject.get('fName') or front_fields.get('First Name', 'N/A')}\n"
                    f"  Middle Name : {subject.get('mName') or front_fields.get('Middle Name', 'N/A')}\n"
                    f"  Suffix      : {subject.get('Suffix') or 'None'}\n"
                    f"  Sex         : {subject.get('sex', 'N/A')}\n"
                    f"  Birthday    : {subject.get('DOB') or front_fields.get('DOB', 'N/A')}\n"
                    f"  Birthplace  : {subject.get('POB', 'N/A')}\n\n"
                    f"  Address     : {front_fields.get('Address', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  PCN         : {subject.get('PCN') or front_fields.get('PCN', 'N/A')}\n"
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



    def infer_only_driver_license_upload(self) -> None:
        p = self.parent
        if not p.front_file:
            return
        path = p.front_file["path"]
        if not os.path.exists(path):
            return
        threading.Thread(target=lambda: self.run_inference_driver_license(path), daemon=True).start()

    def infer_only_driver_license_camera(self) -> None:
        p = self.parent
        if not hasattr(p, "captured_front_frame"):
            return
        frame = p.captured_front_frame.copy()
        threading.Thread(target=lambda: self.run_inference_driver_license(frame), daemon=True).start()

    def infer_only_national_id_camera(self) -> None:
        p = self.parent
        if not hasattr(p, "captured_front_frame") or not hasattr(p, "captured_back_frame"):
            return
        front = p.captured_front_frame.copy()
        back = p.captured_back_frame.copy()
        threading.Thread(target=lambda: self.run_inference_national_id(front, back), daemon=True).start()

    def infer_only_national_id_upload(self) -> None:
        p = self.parent
        if not p.front_file or not p.back_file:
            return
        front_path = p.front_file["path"]
        back_path = p.back_file["path"]
        if not os.path.exists(front_path) or not os.path.exists(back_path):
            return
        threading.Thread(target=lambda: self.run_inference_national_id(front_path, back_path), daemon=True).start()

    def validate_passport_result_sync(self, result: dict) -> bool:
        p = self.parent
        if not result:  # handles None or {}
            QMessageBox.warning(p, "Scan Failed",
                                "No data was detected.\n\nPlease upload a clearer image or recapture.")
            return False
        try:
            parsed = result.get("parsed", {})
            mrz = parsed.get("Passport/MRZ")

            if not mrz:
                QMessageBox.warning(p, "Scan Failed",
                                    "No MRZ data was detected.\n\nPlease upload a clearer image or recapture.")
                return False

            surname = mrz.get("Surname", "").strip()
            given_names = mrz.get("Given_names", "").strip()
            doc_number = mrz.get("Document_number", "").strip()

            missing = []
            if not surname:
                missing.append("Surname")
            if not given_names:
                missing.append("Given Names")
            if not doc_number:
                missing.append("Passport Number")

            if missing:
                fields = ", ".join(missing)
                QMessageBox.warning(p, "Incomplete Scan",
                                    f"The following required fields were not detected:\n\n{fields}\n\nPlease upload a clearer image or recapture.")
                return False

            return True

        except Exception as e:
            print(f"[validate_passport_result_sync] Error: {e}")
            return False

    def validate_driver_license_result_sync(self, result: dict) -> bool:
        p = self.parent
        try:
            data = result.get("parsed", {}).get("Driverslicense/OCR", {})

            if not data:
                QMessageBox.warning(p, "Scan Failed",
                                    "No data was detected.\n\nPlease upload a clearer image or recapture.")
                return False

            name = data.get("Name", "").strip() if data.get("Name") else ""
            license_no = data.get("License No", "").strip() if data.get("License No") else ""
            expiration = data.get("Expiration Date", "").strip() if data.get("Expiration Date") else ""
            birthday = data.get("Birthdate", "").strip() if data.get("Birthdate") else ""

            missing = []
            if not name or name == "N/A":
                missing.append("Name")
            if not license_no or license_no == "N/A":
                missing.append("License No")
            if not expiration or expiration == "N/A":
                missing.append("Expiration Date")
            if not birthday or birthday == "N/A":
                missing.append("Birthdate")

            if missing:
                fields = ", ".join(missing)
                QMessageBox.warning(
                    p,
                    "Incomplete Scan",
                    f"The following required fields were not detected:\n\n{fields}\n\nPlease upload a clearer image or recapture."
                )
                return False

            return True

        except Exception as e:
            print(f"[validate_driver_license_result_sync] Error: {e}")
            return False

    def validate_national_id_result_sync(self, result: dict) -> bool:
        p = self.parent
        try:
            qr_valid = result.get("qr", {}).get("valid", False)
            front_fields = result.get("front", {}).get("parsed", {}).get("NationalID/Front", {})
            front_valid = bool(front_fields and front_fields.get("PCN"))

            # Front OCR must have worked — that is always required
            if not front_valid:
                QMessageBox.warning(p, "Front Scan Failed",
                                    "Could not extract data from the front.\n\nPlease recapture or re-upload.")
                return False

            # If QR also succeeded, check that front and back data match
            if qr_valid:
                match = result.get("match", {})
                if not match.get("passed", False):
                    mismatches = match.get("mismatches", [])
                    mismatch_text = "\n".join(mismatches)
                    QMessageBox.warning(p, "ID Verification Failed",
                                        f"Front and back data do not match:\n\n{mismatch_text}\n\nPlease recapture or re-upload.")
                    return False
            else:
                # QR failed but front OCR worked — warn the user but allow proceed
                reply = QMessageBox.warning(p, "QR Not Detected",
                    "The QR code on the back could not be read.\n\n"
                    "Front data was extracted successfully.\n\n"
                    "Do you want to continue with front data only?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply != QMessageBox.StandardButton.Yes:
                    return False

            return True

        except Exception as e:
            print(f"[validate_national_id_result_sync] Error: {e}")
            return False

    @staticmethod
    def match_national_id(qr_result: dict, front_result: dict) -> dict:
        mismatches: list[str] = []
        try:
            qr_subject = (qr_result.get("NationalID/QR") or {}).get("subject") or {}
            front_fields = front_result.get("parsed", {}).get("NationalID/Front", {})

            if not qr_subject or not front_fields:
                return {"passed": False, "mismatches": ["Could not extract data from one or both sides."]}
            if not front_fields:
                return {"passed": False, "mismatches": ["Could not extract any data from the front of the ID."]}

            if not front_fields.get("PCN"):
                return {"passed": False,
                        "mismatches": ["PCN not detected on front. Please recapture the front of the ID."]}

            # Match Full Name
            qr_fname = qr_subject.get("fName", "").strip().upper()
            qr_lname = qr_subject.get("lName", "").strip().upper()
            front_fname = front_fields.get("First Name", "").strip().upper()
            front_lname = front_fields.get("Last Name", "").strip().upper()

            if qr_fname != front_fname:
                mismatches.append(f"First Name: QR='{qr_fname}' vs Front='{front_fname}'")
            if qr_lname != front_lname:
                mismatches.append(f"Last Name: QR='{qr_lname}' vs Front='{front_lname}'")

            # Match DOB - normalize both to compare
            qr_dob = qr_subject.get("DOB", "").strip().upper()
            front_dob = front_fields.get("DOB", "").strip().upper()

            # Normalize format e.g. "February 14, 2005" vs "FEBRUARY 14, 2005"
            if qr_dob != front_dob:
                mismatches.append(f"Date of Birth: QR='{qr_dob}' vs Front='{front_dob}'")

            # Match PCN
            qr_pcn = qr_subject.get("PCN", "").strip()
            front_pcn = front_fields.get("PCN", "").strip()

            if qr_pcn != front_pcn:
                mismatches.append(f"PCN: QR='{qr_pcn}' vs Front='{front_pcn}'")

            if mismatches:
                return {"passed": False, "mismatches": mismatches}

            return {"passed": True, "mismatches": []}

        except Exception as e:
            print(f"[match_national_id] Error: {e}")
            return {"passed": False, "mismatches": [str(e)]}