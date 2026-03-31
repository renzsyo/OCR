"""
inference_handler.py
--------------------
CHANGES FROM PREVIOUS VERSION:
  - REMOVED [run_inference_passport, run_inference_driver_license,
             run_inference_philhealth, run_inference_tin,
             run_inference_senior_citizen, run_inference_sss]:
             All six single-sided inference runners were identical in
             structure. Replaced with one generic method:
             _run_single_sided_inference(id_type, path).
             Driver's License still sets dl_inference_complete in addition
             to inference_complete — handled inside the new method.
  - REMOVED [run_full_ocr_camera, run_full_ocr_upload]:
             Both were explicitly marked "no longer called in normal flow"
             and only referenced the now-deleted individual runners.
             Dead code — deleted.
  - CHANGED [run_front_detection]: 6-branch if/elif dispatch replaced with
             a single call to _run_single_sided_inference(id_type, source).
             The standalone self.inference_complete = True line that followed
             the old dispatch was removed — the flag is now set inside
             _run_single_sided_inference under the lock, consistent with how
             all other flags are handled.
  - ADDED   [_SCAN_FN_MAP]: class-level dict mapping id_type strings to
             their scan functions. Adding a new ID type in future requires
             only one new import and one new entry here.
  - REMOVED [validate_philhealth_result_sync, validate_tin_result_sync,
             validate_senior_citizen_result_sync, validate_sss_result_sync]:
             All four were structurally identical — extract a parsed dict,
             warn if empty, check required fields, warn if any missing.
             Replaced with one generic helper:
             _validate_simple_id(result, parsed_key, id_label, required_fields).
             The four public methods are kept as thin one-line wrappers so
             all call sites in main.py remain unchanged.
"""

import cv2, time, threading, os
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from .inference import (
    scan_passport, scan_national_id_back, scan_driver_license,
    scan_national_id_front, classify_id_type,
    scan_philhealth, scan_tin, scan_senior_citizen, scan_sss
)
try:
    from .id_classifier import classify_and_gradcam as _classify_and_gradcam
    _ID_CLASSIFIER_AVAILABLE = True
except Exception as _e:
    print(f"[InferenceHandler] id_classifier not available: {_e}")
    _ID_CLASSIFIER_AVAILABLE = False
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow

# ID types that require a back side
TWO_SIDED_IDS = {"National ID", "UMID"}


class InferenceHandler:

    # ------------------------------------------------------------------
    # Scan function dispatch table — add new ID types here only
    # ------------------------------------------------------------------
    _SCAN_FN_MAP = {
        "Passport":         scan_passport,
        "Driver's License": scan_driver_license,
        "PhilHealth":       scan_philhealth,
        "TIN":              scan_tin,
        "Senior Citizen":   scan_senior_citizen,
        "SSS":              scan_sss,
    }

    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent

        self._result_lock = threading.Lock()

        # Inference completion flags
        self.inference_complete = False
        self.dl_inference_complete = False
        self.ni_inference_complete = False

        # Front-only detection flags
        self.detection_complete = False
        self.detection_result_type = None
        self.detection_confidence = 0.0
        self._retry_count = 0

        self.watch_timer = QTimer()
        self.watch_timer.setInterval(100)
        self.watch_timer.timeout.connect(self.check_inference_done)
        self.watch_timer.start()

    def reset_detection(self) -> None:
        """Call this at the start of each new session."""
        with self._result_lock:
            self.detection_complete = False
            self.detection_result_type = None
            self.detection_confidence = 0.0
            self._retry_count = 0

    # ------------------------------------------------------------------
    # Timer — polls all completion flags on the main thread
    # ------------------------------------------------------------------

    def check_inference_done(self) -> None:
        p = self.parent

        # Snapshot all flags under the lock, then act outside it so we
        # never hold the lock while calling Qt functions.
        with self._result_lock:
            inf_done = self.inference_complete
            dl_done  = self.dl_inference_complete
            ni_done  = self.ni_inference_complete
            det_done = self.detection_complete

            if inf_done:
                self.inference_complete = False
            if dl_done:
                self.dl_inference_complete = False
            if ni_done:
                self.ni_inference_complete = False
            if det_done:
                self.detection_complete = False

        if inf_done:
            try:
                p.continuep2.setEnabled(True)
                p.continuep3.setEnabled(True)
            except Exception:
                pass

        if dl_done:
            try:
                p.continuep5.setEnabled(True)
                p.continuep6.setEnabled(True)
            except Exception:
                pass

        if ni_done:
            try:
                p.continuep5.setEnabled(True)
                p.continuep6.setEnabled(True)
            except Exception:
                pass

        if det_done:
            self.on_detection_finished()

    # ------------------------------------------------------------------
    # Front-only detection entry points
    # ------------------------------------------------------------------

    def infer_front_camera(self) -> None:
        """Called after user captures front image on the single-cam page."""
        p = self.parent
        if not hasattr(p, "captured_frame"):
            QMessageBox.warning(p, "No Capture", "Please capture an image first.")
            return
        frame = p.captured_frame.copy()
        threading.Thread(
            target=self.run_front_detection,
            args=(frame,),
            daemon=True,
        ).start()

    def infer_front_upload(self) -> None:
        """Called after user uploads front image on the single-upload page."""
        p = self.parent
        if not p.front_file:
            QMessageBox.warning(p, "No file", "Please upload an image first.")
            return
        path = p.front_file.get("path")
        if not path or not os.path.exists(path):
            QMessageBox.warning(p, "Error", "Selected file not found.")
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(p, "Error", "Could not read uploaded image.")
            return
        threading.Thread(
            target=self.run_front_detection,
            args=(image,),
            daemon=True,
        ).start()

    def run_front_detection(self, image: np.ndarray) -> None:
        """
        Background thread: classify the front image, then immediately run
        full OCR if it's a single-sided ID — all in the same thread so
        PyTorch and PaddleOCR never run concurrently (prevents 0xC0000409).
        All flag writes go through _result_lock so the main-thread timer
        never reads a half-written value.
        """
        try:
            # Safety 1: resize large frames
            h, w = image.shape[:2]
            if w > 1280 or h > 1280:
                scale = 1280 / max(w, h)
                image = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

            # Safety 2: skip near-blank frames
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = float(np.var(gray))
            if variance < 100.0:
                print(f"[FrontDetection] Frame too uniform (variance={variance:.1f}), skipping.")
                with self._result_lock:
                    self.detection_result_type = None
                    self.detection_confidence = 0.0
                return

            # Step 1: MobileNet classifier
            id_type, confidence = classify_id_type(image)

            # Step 2: keyword OCR fallback if classifier is inconclusive
            if id_type is None:
                print("[FrontDetection] Classifier inconclusive — trying keyword fallback.")
                from .inference import auto_detect_all_ids
                hits = auto_detect_all_ids([image])
                if hits:
                    id_type = hits[0][0]
                    confidence = 1.0
                    print(f"[FrontDetection] Keyword fallback detected: {id_type}")

            with self._result_lock:
                self.detection_result_type = id_type
                self.detection_confidence = confidence
            print(f"[FrontDetection] final type={id_type}, confidence={confidence:.2%}")

            # Step 2b: YOLO classifier + Grad-CAM (non-fatal, runs in same thread)
            # Also promotes YOLO result to routing if MobileNet + keywords both failed.
            if _ID_CLASSIFIER_AVAILABLE:
                try:
                    clf_result = _classify_and_gradcam(image)
                    if clf_result is not None:
                        p = self.parent
                        p._classifier_result = clf_result
                        p._gradcam_path      = clf_result.gradcam_path
                        print(
                            f"[IDClassifier] {clf_result.class_name} "
                            f"({clf_result.confidence:.2%})"
                        )

                        if id_type is None and clf_result.class_name != "Uncertain":
                            _YOLO_TO_APP = {
                                "drivers_license": "Driver's License",
                                "passport":        "Passport",
                                "philhealth":      "PhilHealth",
                                "philid":          "National ID",
                                "senior":          "Senior Citizen",
                                "sss":             "SSS",
                                "tin":             "TIN",
                            }
                            mapped = _YOLO_TO_APP.get(clf_result.class_name)
                            if mapped:
                                id_type    = mapped
                                confidence = clf_result.confidence
                                with self._result_lock:
                                    self.detection_result_type = id_type
                                    self.detection_confidence  = confidence
                                print(
                                    f"[IDClassifier] Promoted YOLO result to routing: "
                                    f"{clf_result.class_name} → {id_type} ({confidence:.2%})"
                                )
                    else:
                        self.parent._gradcam_path = None
                except Exception as _clf_err:
                    print(f"[IDClassifier] Non-fatal error: {_clf_err}")
                    self.parent._gradcam_path = None

            # Step 3: for single-sided IDs, run OCR in this same thread so
            # PyTorch and PaddleOCR never overlap.
            if id_type is not None and id_type not in TWO_SIDED_IDS:
                p = self.parent
                source = image
                if p.front_file:
                    fp = p.front_file.get("path", "")
                    if fp and os.path.exists(fp):
                        source = fp
                self._run_single_sided_inference(id_type, source)

        except Exception as e:
            print(f"[FrontDetection] Error: {e}")
            with self._result_lock:
                self.detection_result_type = None
                self.detection_confidence = 0.0
        finally:
            with self._result_lock:
                self.detection_complete = True

    def on_detection_finished(self) -> None:
        """Main-thread callback after front detection completes."""
        p = self.parent
        with self._result_lock:
            id_type    = self.detection_result_type
            confidence = self.detection_confidence

        current_page = p.Form1.currentIndex()

        if id_type is not None:
            p.detected_id_type = id_type
            print(f"[FrontDetection] Detected: {id_type} ({confidence:.2%})")

            if id_type in TWO_SIDED_IDS:
                if current_page == 1:
                    p.proceed_to_back_camera()
                else:
                    p.proceed_to_back_upload()
            # Single-sided: OCR already ran inside run_front_detection,
            # inference_complete flag was set there — nothing more to do here.
            return

        # Inconclusive — ask for another picture
        with self._result_lock:
            self._retry_count += 1
            retry = self._retry_count
            confidence = self.detection_confidence
        print(f"[FrontDetection] Inconclusive (attempt {retry}), "
              f"confidence={confidence:.2%}")

        QMessageBox.warning(
            p,
            "Could Not Detect ID",
            "The ID type could not be detected from this image.\n\n"
            "Please try again with a clearer photo.",
        )
        try:
            p.continuep2.setEnabled(True)
            p.continuep3.setEnabled(True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Inference runners
    # ------------------------------------------------------------------

    def _run_single_sided_inference(self, id_type: str, path: "np.ndarray | str") -> None:
        """
        Generic runner for all single-sided ID types.

        Looks up the correct scan function from _SCAN_FN_MAP, runs it,
        then stores the result and sets the completion flag — all under
        _result_lock so the main-thread timer never reads a partial state.

        Driver's License also sets dl_inference_complete so its dedicated
        continue buttons (continuep5 / continuep6) are enabled by the timer.

        To add a new ID type: import its scan function at the top of this
        file and add one entry to _SCAN_FN_MAP. Nothing else changes.
        """
        scan_fn = self._SCAN_FN_MAP.get(id_type)
        if scan_fn is None:
            print(f"[InferenceHandler] No scan function mapped for id_type='{id_type}'")
            return

        p     = self.parent
        debug = getattr(p, "debug_mode", False)
        result = scan_fn(path, debug=debug)
        print(result)

        with self._result_lock:
            p.pendingResponse   = result
            p.pendingDebugImage = result.get("debug_image")
            self.inference_complete = True
            if id_type == "Driver's License":
                self.dl_inference_complete = True

    def run_inference_national_id(self, front_image: "np.ndarray | str",
                                   back_image: "np.ndarray | str") -> None:
        """
        Two-sided runner for National ID / UMID.
        Kept separate from _run_single_sided_inference because it involves
        two images, a QR decode, a front/back match check, and an optional
        back-side Grad-CAM — genuinely different from all other ID types.
        """
        p     = self.parent
        debug = getattr(p, "debug_mode", False)

        qr_result    = scan_national_id_back(back_image, debug=debug)
        print("[NationalID] QR result:", qr_result)

        front_result = scan_national_id_front(front_image, debug=debug)
        print("[NationalID] Front OCR result:", front_result)

        match_result = self.match_national_id(qr_result, front_result)
        print("[NationalID] Match result:", match_result)

        payload = {
            "qr":    qr_result,
            "front": front_result,
            "match": match_result,
            "valid": qr_result.get("valid", False) and match_result.get("passed", False),
        }

        back_gradcam_path = None
        if debug:
            try:
                from .id_classifier import classify_and_gradcam_back as _cag_back
                back_img = cv2.imread(back_image) if isinstance(back_image, str) else back_image
                if back_img is not None:
                    back_gradcam_path = _cag_back(back_img)
            except Exception as _e:
                print(f"[NationalID] Back Grad-CAM failed (non-fatal): {_e}")

        with self._result_lock:
            p.pendingResponse       = payload
            p.pendingDebugImage     = front_result.get("debug_image")
            p.pendingDebugImageBack = qr_result.get("debug_image")
            p._gradcam_path_back    = back_gradcam_path
            self.ni_inference_complete = True

    # ------------------------------------------------------------------
    # Two-sided triggers (page 4 camera / page 5 upload)
    # ------------------------------------------------------------------

    def infer_only_national_id_camera(self) -> None:
        p = self.parent
        if not hasattr(p, "captured_front_frame") or not hasattr(p, "captured_back_frame"):
            return
        front = p.captured_front_frame.copy()
        back  = p.captured_back_frame.copy()
        threading.Thread(
            target=lambda: self.run_inference_national_id(front, back),
            daemon=True,
        ).start()

    def infer_only_national_id_upload(self) -> None:
        p = self.parent
        if not p.front_file or not p.back_file:
            return
        front_path = p.front_file["path"]
        back_path  = p.back_file["path"]
        if not os.path.exists(front_path) or not os.path.exists(back_path):
            return
        threading.Thread(
            target=lambda: self.run_inference_national_id(front_path, back_path),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Format / Validate
    # ------------------------------------------------------------------

    def format_pending_response(self, result: dict, id_type: str) -> str:
        print(f"[FORMAT DEBUG] id_type='{id_type}', result type={type(result)}, result={result}")
        try:
            if id_type == "National ID":
                data = result.get("qr", {}).get("NationalID/QR") or {}
                subject = data.get("subject") or {}
                front_fields = (
                    result.get("front", {})
                          .get("parsed", {})
                          .get("NationalID/Front", {}) or {}
                )
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

            elif id_type == "UMID":
                data = result.get("qr", {}).get("NationalID/QR") or {}
                subject = data.get("subject") or {}
                front_fields = (
                    result.get("front", {})
                          .get("parsed", {})
                          .get("NationalID/Front", {}) or {}
                )
                return (
                    f"👤 PERSONAL INFORMATION (UMID)\n"
                    f"{'─' * 23}\n"
                    f"  Last Name   : {subject.get('lName') or front_fields.get('Last Name', 'N/A')}\n"
                    f"  First Name  : {subject.get('fName') or front_fields.get('First Name', 'N/A')}\n"
                    f"  Middle Name : {subject.get('mName') or front_fields.get('Middle Name', 'N/A')}\n"
                    f"  Sex         : {subject.get('sex', 'N/A')}\n"
                    f"  Birthday    : {subject.get('DOB') or front_fields.get('DOB', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  PCN         : {subject.get('PCN') or front_fields.get('PCN', 'N/A')}\n\n"
                )

            elif id_type == "PhilHealth":
                data = result.get("parsed", {}).get("PhilHealth/OCR", {})
                return (
                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Name        : {data.get('name', 'N/A')}\n"
                    f"  Sex         : {data.get('sex', 'N/A')}\n"
                    f"  Birthday    : {data.get('date_of_birth', 'N/A')}\n\n"
                    f"  ADDRESS\n"
                    f"{'─' * 23}\n"
                    f"  Address     : {data.get('address', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  PhilHealth No : {data.get('philhealth_id_number', 'N/A')}\n\n"
                )

            elif id_type == "TIN":
                data = result.get("parsed", {}).get("TIN/OCR", {})
                return (
                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Name        : {data.get('name', 'N/A')}\n\n"
                    f"  ADDRESS\n"
                    f"{'─' * 23}\n"
                    f"  Address     : {data.get('address', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  TIN         : {data.get('tin', 'N/A')}\n"
                    f"  Birthday    : {data.get('date_of_birth', 'N/A')}\n"
                    f"  Date Issued : {data.get('date_issued', 'N/A')}\n\n"
                )

            elif id_type == "Senior Citizen":
                data = result.get("parsed", {}).get("SeniorCitizen/OCR", {})
                return (
                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Name          : {data.get('name', 'N/A')}\n"
                    f"  Date of Birth : {data.get('date_of_birth', 'N/A')}\n"
                    f"  Age           : {data.get('age', 'N/A')}\n\n"
                    f"  ADDRESS\n"
                    f"{'─' * 23}\n"
                    f"  Address       : {data.get('address', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  ID Number     : {data.get('id_number', 'N/A')}\n"
                    f"  Date Issued   : {data.get('date_of_issue', 'N/A')}\n"
                    f"  Issuing Office: {data.get('issuing_office', 'N/A')}\n\n"
                )

            elif id_type == "SSS":
                data = result.get("parsed", {}).get("SSS/OCR", {})
                return (
                    f"👤 PERSONAL INFORMATION\n"
                    f"{'─' * 23}\n"
                    f"  Name          : {data.get('name', 'N/A')}\n"
                    f"  Date of Birth : {data.get('date_of_birth', 'N/A')}\n\n"
                    f"  ID DETAILS\n"
                    f"{'─' * 23}\n"
                    f"  SSS Number    : {data.get('sss_number', 'N/A')}\n\n"
                )

        except Exception as e:
            return f"⚠️ Could not format result: {e}\n\nRaw output:\n{result}"

    def validate_passport_result_sync(self, result: dict) -> bool:
        p = self.parent
        if not result:
            QMessageBox.warning(p, "Scan Failed",
                "No data was detected.\n\nPlease upload a clearer image or recapture.")
            return False
        try:
            mrz = result.get("parsed", {}).get("Passport/MRZ")
            if not mrz:
                QMessageBox.warning(p, "Scan Failed",
                    "No MRZ data was detected.\n\nPlease upload a clearer image or recapture.")
                return False
            missing = []
            if not mrz.get("Surname", "").strip():          missing.append("Surname")
            if not mrz.get("Given_names", "").strip():      missing.append("Given Names")
            if not mrz.get("Document_number", "").strip():  missing.append("Passport Number")
            if missing:
                QMessageBox.warning(p, "Incomplete Scan",
                    f"The following required fields were not detected:\n\n"
                    f"{', '.join(missing)}\n\n"
                    f"Please upload a clearer image or recapture.")
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
            missing = []
            if not (data.get("Name") or "").strip():             missing.append("Name")
            if not (data.get("License No") or "").strip():       missing.append("License No")
            if not (data.get("Expiration Date") or "").strip():  missing.append("Expiration Date")
            if not (data.get("Birthdate") or "").strip():        missing.append("Birthdate")
            if missing:
                QMessageBox.warning(p, "Incomplete Scan",
                    f"The following required fields were not detected:\n\n"
                    f"{', '.join(missing)}\n\n"
                    f"Please upload a clearer image or recapture.")
                return False
            return True
        except Exception as e:
            print(f"[validate_driver_license_result_sync] Error: {e}")
            return False

    def validate_national_id_result_sync(self, result: dict) -> bool:
        p = self.parent
        try:
            qr_valid     = result.get("qr", {}).get("valid", False)
            front_fields = (result.get("front", {})
                                  .get("parsed", {})
                                  .get("NationalID/Front", {}))
            front_valid  = bool(front_fields and front_fields.get("PCN"))

            if not front_valid:
                QMessageBox.warning(p, "Front Scan Failed",
                    "Could not extract data from the front.\n\nPlease recapture or re-upload.")
                return False

            if qr_valid:
                match = result.get("match", {})
                if not match.get("passed", False):
                    mismatches = match.get("mismatches", [])
                    QMessageBox.warning(p, "ID Verification Failed",
                        f"Front and back data do not match:\n\n"
                        f"{chr(10).join(mismatches)}\n\n"
                        f"Please recapture or re-upload.")
                    return False
            else:
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

    def _validate_simple_id(
        self,
        result: dict,
        parsed_key: str,
        id_label: str,
        required_fields: list[tuple[str, str]],
    ) -> bool:
        """
        Generic validator for single-sided IDs whose result lives at
        result["parsed"][parsed_key].

        Parameters
        ----------
        result         : the pendingResponse dict
        parsed_key     : e.g. "PhilHealth/OCR", "TIN/OCR", "SSS/OCR"
        id_label       : human-readable name shown in warning dialogs
        required_fields: list of (dict_key, display_name) pairs to check.
                         A field is considered missing when its value is
                         falsy or blank after stripping whitespace.

        To add a new ID type: add one entry to _VALIDATE_CONFIG below
        and one thin wrapper that calls this method. No other changes needed.
        """
        p = self.parent
        try:
            data = result.get("parsed", {}).get(parsed_key, {})
            if not data:
                QMessageBox.warning(
                    p, "Scan Failed",
                    f"No {id_label} data was detected.\n\n"
                    f"Please upload a clearer image or recapture.",
                )
                return False

            missing = [
                display
                for field_key, display in required_fields
                if not (data.get(field_key) or "").strip()
            ]

            if missing:
                QMessageBox.warning(
                    p, "Incomplete Scan",
                    f"The following required fields were not detected:\n\n"
                    f"{', '.join(missing)}\n\n"
                    f"Please upload a clearer image or recapture.",
                )
                return False

            return True
        except Exception as e:
            print(f"[_validate_simple_id:{parsed_key}] Error: {e}")
            return False

    # Thin wrappers — call sites in main.py stay completely unchanged.
    def validate_philhealth_result_sync(self, result: dict) -> bool:
        return self._validate_simple_id(
            result,
            parsed_key      = "PhilHealth/OCR",
            id_label        = "PhilHealth",
            required_fields = [("philhealth_id_number", "PhilHealth ID Number"),
                                ("name",                "Name")],
        )

    def validate_tin_result_sync(self, result: dict) -> bool:
        return self._validate_simple_id(
            result,
            parsed_key      = "TIN/OCR",
            id_label        = "TIN",
            required_fields = [("tin",  "TIN Number"),
                                ("name", "Name")],
        )

    def validate_senior_citizen_result_sync(self, result: dict) -> bool:
        return self._validate_simple_id(
            result,
            parsed_key      = "SeniorCitizen/OCR",
            id_label        = "Senior Citizen ID",
            required_fields = [("name",      "Name"),
                                ("id_number", "ID Number")],
        )

    def validate_sss_result_sync(self, result: dict) -> bool:
        return self._validate_simple_id(
            result,
            parsed_key      = "SSS/OCR",
            id_label        = "SSS ID",
            required_fields = [("sss_number", "SSS Number"),
                                ("name",       "Name")],
        )

    @staticmethod
    def match_national_id(qr_result: dict, front_result: dict) -> dict:
        mismatches: list[str] = []
        try:
            qr_subject   = (qr_result.get("NationalID/QR") or {}).get("subject") or {}
            front_fields = front_result.get("parsed", {}).get("NationalID/Front", {})

            if not qr_subject or not front_fields:
                return {"passed": False,
                        "mismatches": ["Could not extract data from one or both sides."]}
            if not front_fields.get("PCN"):
                return {"passed": False,
                        "mismatches": ["PCN not detected on front. Please recapture the front of the ID."]}

            qr_fname    = qr_subject.get("fName", "").strip().upper()
            qr_lname    = qr_subject.get("lName", "").strip().upper()
            front_fname = front_fields.get("First Name", "").strip().upper()
            front_lname = front_fields.get("Last Name", "").strip().upper()

            if qr_fname != front_fname:
                mismatches.append(f"First Name: QR='{qr_fname}' vs Front='{front_fname}'")
            if qr_lname != front_lname:
                mismatches.append(f"Last Name: QR='{qr_lname}' vs Front='{front_lname}'")

            qr_dob    = qr_subject.get("DOB", "").strip().upper()
            front_dob = front_fields.get("DOB", "").strip().upper()
            if qr_dob != front_dob:
                mismatches.append(f"Date of Birth: QR='{qr_dob}' vs Front='{front_dob}'")

            qr_pcn    = qr_subject.get("PCN", "").strip()
            front_pcn = front_fields.get("PCN", "").strip()
            if qr_pcn != front_pcn:
                mismatches.append(f"PCN: QR='{qr_pcn}' vs Front='{front_pcn}'")

            if mismatches:
                return {"passed": False, "mismatches": mismatches}
            return {"passed": True, "mismatches": []}

        except Exception as e:
            print(f"[match_national_id] Error: {e}")
            return {"passed": False, "mismatches": [str(e)]}