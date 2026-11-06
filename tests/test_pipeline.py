import csv
from pathlib import Path
import pytest
from PIL import Image
from pipeline import InspectionError, Inspector, load_image

MODEL = Path("models/resnet18_neutral.pt")
SPLITS = Path("data/splits.csv")


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(InspectionError, match="not found"):
        load_image(tmp_path / "nothing.jpg")


def test_wrong_file_type_gives_a_clear_error(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(InspectionError, match="Unsupported"):
        load_image(tmp_path / "notes.txt")


def test_damaged_image_gives_a_clear_error(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"this is not an image")
    with pytest.raises(InspectionError, match="damaged"):
        load_image(tmp_path / "broken.jpg")


def test_tiny_image_gives_a_clear_error(tmp_path):
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "tiny.png")
    with pytest.raises(InspectionError, match="too small"):
        load_image(tmp_path / "tiny.png")


def test_valid_image_is_loaded(tmp_path):
    Image.new("RGB", (100, 80), (10, 20, 30)).save(tmp_path / "ok.png")
    assert load_image(tmp_path / "ok.png").size == (100, 80)


needs_model = pytest.mark.skipif(not MODEL.exists(), reason="trained model not found")


@pytest.fixture(scope="module")
def inspector():
    return Inspector()


@needs_model
def test_blank_image_is_rejected(inspector, tmp_path):
    Image.new("RGB", (300, 300), (128, 128, 128)).save(tmp_path / "blank.png")
    with pytest.raises(InspectionError, match="blank"):
        inspector.inspect(tmp_path / "blank.png")


@needs_model
@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.csv not found")
def test_real_image_gives_a_complete_and_repeatable_result(inspector):
    with open(SPLITS, newline="") as fh:
        path = next(r["path"] for r in csv.DictReader(fh) if r["split"] == "val")
    first = inspector.inspect(path)
    second = inspector.inspect(path)
    assert abs(sum(first["class_probs"].values()) - 1.0) < 1e-4
    assert first["verdict"]["severity"] in ("None", "Low", "Medium", "High")
    assert first["disclaimer"]
    assert first["class_probs"] == pytest.approx(second["class_probs"], abs=1e-6)
    mask_pixels = int((first["region_mask"] > 0).sum())
    assert sum(m["area_px"] for m in first["measurements"]) == mask_pixels