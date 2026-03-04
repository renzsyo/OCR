import cv2, time, threading
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from .inference import scan_passport, scan_national_id, scan_driver_license


class InferenceHandler:
    def __init__(self, parent):
        self.parent = parent

    def _run_inference_passport(self, path):
        p = self.parent
        debug = getattr(p, "debug_mode", False)
        result = scan_passport(path, debug=debug)
        p.pendingResponse = result
        p.pendingDebugImage = result.get("debug_image")
        formatted = self._format_pending_response(result, "Passport")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))
        print(result)

    def _run_inference_national_id(self, image):
        p = self.parent
        result = scan_national_id(image)
        p.pendingResponse = result
        formatted = self._format_pending_response(result, "National ID")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))
        print(result)

    def _run_inference_driver_license(self, path):
        p = self.parent
        debug = getattr(p, "debug_mode", False)
        result = scan_driver_license(path, debug=debug)
        p.pendingResponse = result
        p.pendingDebugImage = result.get("debug_image")
        formatted = self._format_pending_response(result, "Driver's License")
        QTimer.singleShot(0, lambda: self._update_extracted_text(formatted))
        print(result)

    def _update_extracted_text(self, text):
        self.parent.extractedText.setText(text) # or whatever your text widget is


    def infer_page2_camera_passport(self):
        p = self.parent
        print("[DEBUG] infer_page2_camera_passport called")
        if not hasattr(p, "captured_frame"):
            QMessageBox.warning(p, "No Capture", "Please capture an image first.")
            return

        frame = p.captured_frame.copy()

        def task():
            self._run_inference_passport(frame)

            # after inference finishes, navigate on UI thread
            QTimer.singleShot(0, p.go_next)

        threading.Thread(target=task, daemon=True).start()

    def infer_page3_upload_passport(self):
        p = self.parent
        files = p.files.uploaded_files
        index = p.files.current_index

        if not files:
            QMessageBox.warning(p, "No file", "Please upload an image first.")
            return
        if index < 0:
            index = len(files) - 1

        path = files[index].get("path")
        if not path or not __import__("os").path.exists(path):
            QMessageBox.warning(p, "Error", "Selected file not found.")
            return

        def task():
            self._run_inference_passport(path)
            QTimer.singleShot(0, p.go_next)

        threading.Thread(target=task, daemon=True).start()

    def infer_page5(self):
        p = self.parent
        selected_id = p.idOption.currentText()

        if selected_id == "National ID":
            if not hasattr(p, "captured_back_frame") or p.captured_back_frame is None:
                QMessageBox.warning(p, "Missing Capture", "Please capture the back image.")
                return

            def task():
                self._run_inference_national_id(p.captured_back_frame)
                QTimer.singleShot(0, p.go_next)

            threading.Thread(target=task, daemon=True).start()

        elif selected_id == "Driver's License":
            if not hasattr(p, "captured_front_frame") or p.captured_front_frame is None:
                QMessageBox.warning(p, "Missing Capture", "Please capture the front image.")
                return

            frame = p.captured_front_frame.copy()

            def task():
                self._run_inference_driver_license(frame)
                QTimer.singleShot(0, p.go_next)

            threading.Thread(target=task, daemon=True).start()

        else:
            QMessageBox.warning(p, "Unknown ID", "Please select a valid ID type.")
    def infer_page6(self):
        p = self.parent
        selected_id = p.idOption.currentText()

        if not p.front_file:
            QMessageBox.warning(p, "Missing Front Image", "Please upload the front image.")
            return
        if not p.back_file:
            QMessageBox.warning(p, "Missing Back Image", "Please upload the back image.")
            return

        if selected_id == "National ID":
            image = cv2.imread(p.back_file["path"])

            def task():
                self._run_inference_national_id(image)
                QTimer.singleShot(0, p.go_next)

            threading.Thread(target=task, daemon=True).start()

        elif selected_id == "Driver's License":
            image = cv2.imread(p.front_file["path"])

            def task():
                self._run_inference_driver_license(p.front_file["path"])
                QTimer.singleShot(0, p.go_next)

            threading.Thread(target=task, daemon=True).start()

        else:
            QMessageBox.warning(p, "Invalid Selection", "Please select a valid ID type.")

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

