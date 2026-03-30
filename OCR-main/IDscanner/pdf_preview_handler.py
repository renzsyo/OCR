"""
pdf_preview_handler.py
BUGFIXES (latest):
  - FIXED   [numpy_to_pixmap]: QImage(resized.data, ...) was missing .copy() — the local
                               numpy array `resized` went out of scope before Qt finished
                               using the raw pointer, causing a 0xC0000409 crash whenever
                               the PDF preview page was rendered.

Flow:
  1. load_pdf()         -> copy pages, show preview, start worker thread.
  2. Worker thread      -> detect type, find front/back by QR, run OCR.
  3. Worker thread      -> emit scan_finished.
  4. Main thread slot   -> store results, update status label.
  5. on_continue()      -> called when user clicks Continue, navigates to review.
  6. reupload_pdf()     -> called when user clicks Re-Upload PDF, reloads preview.
"""

import os, threading, cv2
import numpy as np

from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QSizePolicy, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainWindow


class WorkerSignals(QObject):
    scan_finished = pyqtSignal(object)   # list[tuple] — [(id_type, result, front_img, back_img, debug_info)]
    status        = pyqtSignal(str)


class PdfPreviewHandler:

    def __init__(self, parent: "MainWindow") -> None:
        self.parent = parent
        self.pdf_pages: list[np.ndarray] = []
        self.pdf_path: str = ""
        self._detection_done: bool = False
        self._scan_done: bool = False
        self._scan_results: list = []

        self.signals = WorkerSignals()
        self.signals.scan_finished.connect(self.on_scan_finished)
        self.signals.status.connect(self.set_status)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def load_pdf(self, pages: list[np.ndarray], pdf_path: str = "") -> None:
        p = self.parent
        # Copy every page immediately so the worker owns its own memory
        self.pdf_pages = [pg.copy() for pg in pages]
        self.pdf_path = pdf_path
        self._detection_done = False
        self._scan_done = False
        self._scan_results = []

        self.render_pages(self.pdf_pages)
        self.populate_file_list(pdf_path, self.pdf_pages)
        self.set_status("Detecting IDs...")
        p.continuep7.setEnabled(True)

        p.page_history.append(0)
        p.Form1.setCurrentIndex(6)

        threading.Thread(target=self.worker_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Render scroll area
    # ------------------------------------------------------------------

    def render_pages(self, pages: list[np.ndarray]) -> None:
        p = self.parent
        new_container = QWidget()
        layout = QVBoxLayout(new_container)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for i, page in enumerate(pages):
            num_label = QLabel(f"Page {i + 1}")
            num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_label.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(num_label)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            pixmap = self.numpy_to_pixmap(page, max_width=460)
            img_label.setPixmap(pixmap)
            img_label.setFixedHeight(pixmap.height() + 4)
            layout.addWidget(img_label)

        try:
            p.scrollArea.setWidget(new_container)
            p.scrollAreaWidgetContents = new_container
        except Exception as e:
            print(f"[PdfPreviewHandler/_render_pages] {e}")

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def populate_file_list(self, pdf_path: str, pages: list[np.ndarray]) -> None:
        p = self.parent
        try:
            p.fileListWidget_2.clear()
            name = os.path.basename(pdf_path) if pdf_path else "Unknown"
            size_str = "N/A"
            if pdf_path and os.path.exists(pdf_path):
                size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB"
            p.fileListWidget_2.addItem(f"PDF: {name}")
            p.fileListWidget_2.addItem(f"    Size: {size_str}")
            p.fileListWidget_2.addItem(f"    Pages: {len(pages)}")
        except Exception as e:
            print(f"[PdfPreviewHandler/_populate_file_list] {e}")

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def worker_thread(self) -> None:
        print("[PdfPreviewHandler] worker started")
        from .inference import (
            auto_detect_all_ids, decode_qr_safe,
            scan_passport, scan_national_id_back,
            scan_national_id_front, scan_national_id_front_from_ocr,
            scan_driver_license,
            scan_philhealth, scan_tin
        )
        from .inference_handler import InferenceHandler

        pages = self.pdf_pages
        debug = getattr(self.parent, "debug_mode", False)

        # ── Step 1: identify what ID type is in the PDF ──────────────
        all_hits = auto_detect_all_ids(pages)   # [(id_type, page_idx, cached_ocr)]

        if not all_hits:
            self.signals.status.emit("No IDs detected. Please try a clearer PDF.")
            self._detection_done = True
            self._scan_done = True
            return

        id_type, page_idx, cached_ocr = all_hits[0]
        self._detection_done = True
        print(f"[PdfPreviewHandler] detected {id_type} on page {page_idx + 1}")

        # ── Step 2: assign front / back pages ────────────────────────
        if id_type == "Passport":
            scan_front = pages[page_idx]
            scan_back  = None

        else:
            # National ID and Driver's License have two sides:
            #   BACK  = the page that contains the QR code
            #   FRONT = any other page (the face/text side)
            #
            # We try to QR-decode every page to locate the back.
            # This makes no assumption about page order.
            qr_page_idx = None
            for i, pg in enumerate(pages):
                qr_data = decode_qr_safe(pg)
                if qr_data:
                    print(f"[PdfPreviewHandler] QR found on page {i + 1} -> back")
                    qr_page_idx = i
                    break

            if qr_page_idx is not None:
                front_candidates = [i for i in range(len(pages)) if i != qr_page_idx]
                front_idx = front_candidates[0] if front_candidates else qr_page_idx
                scan_back  = pages[qr_page_idx]
                scan_front = pages[front_idx]
                print(f"[PdfPreviewHandler] front=page {front_idx + 1}, back=page {qr_page_idx + 1}")
            else:
                # No QR found on any page — scan both pages as potential back,
                # use the one that returns valid QR data, fall back to page order.
                print("[PdfPreviewHandler] no QR found by quick scan, trying full QR scan on all pages")
                best_back_idx = None
                for i, pg in enumerate(pages):
                    from .inference import scan_national_id_back as _snid
                    probe = _snid(pg)
                    if probe.get("valid"):
                        best_back_idx = i
                        print(f"[PdfPreviewHandler] QR confirmed on page {i + 1} via full scan")
                        break
                if best_back_idx is not None:
                    front_candidates = [i for i in range(len(pages)) if i != best_back_idx]
                    front_idx = front_candidates[0] if front_candidates else best_back_idx
                    scan_back  = pages[best_back_idx]
                    scan_front = pages[front_idx]
                else:
                    # Truly no QR anywhere — use page order
                    print("[PdfPreviewHandler] no QR on any page, falling back to page order")
                    scan_front = pages[page_idx]
                    bi = page_idx + 1 if page_idx + 1 < len(pages) else page_idx
                    scan_back  = pages[bi]

        # ── Step 3: run OCR ───────────────────────────────────────────
        self.signals.status.emit(f"{id_type} found — Scanning...")

        try:
            if id_type == "Passport":
                result = scan_passport(scan_front, debug=debug)

            elif id_type == "National ID":
                qr_result = scan_national_id_back(scan_back, debug=debug)
                if cached_ocr is not None:
                    front_result = scan_national_id_front_from_ocr(
                        cached_ocr, scan_front, debug=debug)
                else:
                    front_result = scan_national_id_front(scan_front, debug=debug)
                match_result = InferenceHandler.match_national_id(qr_result, front_result)
                result = {
                    "qr":    qr_result,
                    "front": front_result,
                    "match": match_result,
                    "valid": qr_result.get("valid", False) and match_result.get("passed", False),
                }

            elif id_type == "Driver's License":
                result = scan_driver_license(scan_front, debug=debug)

            elif id_type == "PhilHealth":
                result = scan_philhealth(scan_front, debug=debug)

            elif id_type == "TIN":
                result = scan_tin(scan_front, debug=debug)

            else:
                result = {}

        except Exception as e:
            print(f"[PdfPreviewHandler] scan error: {e}")
            import traceback; traceback.print_exc()
            result = {}

        if debug and scan_front is not None:
            try:
                from .id_classifier import classify_and_gradcam as _cag
                clf_result = _cag(scan_front)
                if clf_result is not None:
                    self.parent._gradcam_path = clf_result.gradcam_path
                    self.parent._classifier_result = clf_result
                else:
                    self.parent._gradcam_path = None
                    self.parent._classifier_result = None
            except Exception as _e:
                print(f"[PdfPreviewHandler] Grad-CAM failed (non-fatal): {_e}")
                self.parent._gradcam_path = None
                self.parent._classifier_result = None
        else:
            self.parent._gradcam_path = None
            self.parent._classifier_result = None

        print("[PdfPreviewHandler] scan done")
        page_ids = [id(pg) for pg in pages]

        if debug and scan_front is not None and id_type in ("National ID", "UMID"):
            try:
                from .id_classifier import classify_and_gradcam_back as _cag_back
                back_gradcam_path = _cag_back(scan_back)
                self.parent._gradcam_path_back = back_gradcam_path
            except Exception as _e:
                print(f"[PdfPreviewHandler] Back Grad-CAM failed (non-fatal): {_e}")
                self.parent._gradcam_path_back = None
        else:
            self.parent._gradcam_path_back = None
        debug_info = {
            "id_type": id_type,
            "page_count": len(pages),
            "detected_page": page_idx + 1,
            "front_page": (page_ids.index(id(scan_front)) + 1) if scan_front is not None else None,
            "back_page": (page_ids.index(id(scan_back)) + 1) if scan_back is not None else None,
            "raw_result": result,
        }
        self.signals.scan_finished.emit([(id_type, result, scan_front, scan_back, debug_info)])

    # ------------------------------------------------------------------
    # Main-thread slot: scan finished
    # ------------------------------------------------------------------

    def on_scan_finished(self, results: list) -> None:
        self._scan_results = results
        self._scan_done = True
        p = self.parent
        p.pendingResponse = results[0][1] if results else {}
        p.pendingDebugImage = results[0][1].get("debug_image") if results else None
        print("[PDF DEBUG] pendingDebugImage:", p.pendingDebugImage)
        print("[PDF DEBUG] raw result keys:", results[0][1].keys() if results else "no results")
        print("[PDF DEBUG] front result keys:", results[0][1].get('front', {}).keys())
        # NEW: store PDF debug info for the debug tab in review
        p._pdf_debug_info = results[0][4] if results and len(results[0]) > 4 else None
        self.set_status("Scan complete — Ready to continue")

        if results:
            _result = results[0][1]
            p.pendingDebugImage = (
                    _result.get("debug_image")
                    or _result.get("front", {}).get("debug_image")
            )
            p.pendingDebugImageBack = _result.get("qr", {}).get("debug_image")
        else:
            p.pendingDebugImage = None
            p.pendingDebugImageBack = None

    # ------------------------------------------------------------------
    # Continue button
    # ------------------------------------------------------------------

    def on_continue(self) -> None:
        if not self._detection_done:
            QMessageBox.information(self.parent, "Please Wait",
                "ID detection is still running, please wait a moment.")
            return
        if self._scan_done and not self._scan_results:
            QMessageBox.warning(self.parent, "No IDs Found",
                "No IDs were detected in this PDF. Please try a clearer scan.")
            return
        if not self._scan_done:
            QMessageBox.information(self.parent, "Please Wait",
                "Scanning is still in progress, please wait a moment.")
            return
        self.proceed()

    def proceed(self) -> None:
        p = self.parent
        results = self._scan_results
        if not results:
            QMessageBox.warning(p, "No Results", "No scan results to show.")
            return

        id_type, result, front_img, back_img, *_ = results[0]
        self.save_images(id_type, front_img, back_img)

        p.detected_id_type = id_type
        p.lastIdType = id_type
        p.lastResult = result
        # Append page 6 so Back from review returns here, not to page 0
        p.page_history.append(6)
        p.Form1.setCurrentIndex(3)
        p.review.show_review_page()

    def save_images(self, id_type: str, front_img, back_img) -> None:
        p = self.parent
        try:
            from .file_handler import save_image, make_file_info
            folder = p.get_output_folder(id_type, "Upload")
            if front_img is not None:
                p.front_file = make_file_info(
                    save_image(front_img, folder, "pdf_scan_front.jpg"), "front")
            if back_img is not None:
                p.back_file = make_file_info(
                    save_image(back_img, folder, "pdf_scan_back.jpg"), "back")
        except Exception as e:
            print(f"[PdfPreviewHandler/_save_images] {e}")

    # ------------------------------------------------------------------
    # Re-upload PDF button
    # ------------------------------------------------------------------

    def reupload_pdf(self) -> None:
        """
        Called when user clicks Re-Upload PDF on the PDF preview page.
        Opens a new file dialog and reloads the preview with the new PDF.
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from .file_handler import convert_pdf_pages
        p = self.parent
        file_paths, _ = QFileDialog.getOpenFileNames(
            p, "Select PDF", "", "PDF Files (*.pdf)")
        if not file_paths:
            return
        pdf_path = file_paths[-1]
        if not __import__("os").path.exists(pdf_path):
            return
        pages = convert_pdf_pages(pdf_path)
        if not pages:
            QMessageBox.warning(p, "PDF Error",
                "Could not convert the PDF.\n\n"
                "Make sure pdf2image and poppler are installed.")
            return
        # Reset state and reload — stay on page 6
        self._detection_done = False
        self._scan_done = False
        self._scan_results = []
        p.front_file = None
        p.back_file = None
        p.detected_id_type = None
        self.load_pdf(pages, pdf_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        try:
            self.parent.pdfDetectionStatus.setText(text)
        except Exception:
            print(f"[PdfPreviewHandler] Status: {text}")

    @staticmethod
    def numpy_to_pixmap(image: np.ndarray, max_width: int = 460) -> QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        scale = min(1.0, max_width / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        # FIXED: .copy() forces Qt to take ownership of the pixel data.
        # Without it, Qt holds a raw pointer into `resized` which is a local
        # numpy array — as soon as this function returns, Python frees it and
        # Qt is left with a dangling pointer, causing a 0xC0000409 crash when
        # the PDF preview page is rendered.
        qimg = QImage(resized.data, new_w, new_h, ch * new_w,QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)