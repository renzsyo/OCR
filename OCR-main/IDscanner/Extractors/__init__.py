from .Scan_passport import scan_passport
from .scan_driver_license import scan_driver_license
from .scan_national_id import (
    scan_national_id_front,
    scan_national_id_front_from_ocr,
    scan_national_id_back,
    decode_qr_safe,
)
from .scan_philhealth import scan_philhealth
from .scan_tin import scan_tin
from .scan_senior_citizen import scan_senior_citizen
from .ocr_engine import ocr_predict