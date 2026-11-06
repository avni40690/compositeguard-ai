from datetime import datetime
import numpy as np
from report import make_report


def fake_result():
    return {
        "disclaimer": "Automated inspection aid only.", "method": "test method",
        "seconds": 0.25, "image_gray": np.full((224, 224), 128, dtype=np.uint8),
        "cam": None, "region_mask": np.zeros((224, 224), dtype=np.uint8),
        "verdict": {"severity": "None", "needs_review": False, "reasons": ["Test reason."]},
        "predicted_class": "negative",
        "class_probs": {"negative": 0.9, "recoverable_defects": 0.05,
                        "non_recoverable_defects": 0.05},
        "measurements": [], "cam_area_percent": 0.0, "localized": False,
        "notes": ["A note."], "inspected_at": datetime(2026, 10, 21, 18, 0),
    }


def test_report_is_a_pdf():
    pdf = make_report(fake_result(), "a.jpg", {})
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000


def test_special_characters_in_the_file_name_do_not_break_the_report():
    pdf = make_report(fake_result(), "a & b <c>.jpg", {})
    assert pdf.startswith(b"%PDF")