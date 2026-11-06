import numpy as np
from measurement import measure_mask


def test_rectangle_is_measured_exactly():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[60:80, 50:90] = 255                       # 40 wide x 20 tall
    region = measure_mask(mask)[0]
    assert region["area_px"] == 800
    assert (region["bbox_width_px"], region["bbox_height_px"]) == (40, 20)
    assert abs(region["length_px"] - (39 ** 2 + 19 ** 2) ** 0.5) < 0.01
    assert "area_mm2" not in region                # no scale, so no millimetres


def test_two_regions_are_sorted_largest_first():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[100:120, 100:120] = 255
    assert [r["area_px"] for r in measure_mask(mask)] == [400, 100]


def test_millimetres_only_appear_with_a_scale():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[60:80, 50:90] = 255
    region = measure_mask(mask, mm_per_pixel=0.1)[0]
    assert abs(region["area_mm2"] - 8.0) < 1e-6


def test_empty_mask_gives_no_regions():
    assert measure_mask(np.zeros((224, 224), dtype=np.uint8)) == []