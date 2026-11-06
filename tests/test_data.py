import csv
import hashlib
from pathlib import Path
import pytest

SPLITS = Path("data/splits.csv")
pytestmark = pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.csv not found")


def load_rows():
    with open(SPLITS, newline="") as fh:
        return list(csv.DictReader(fh))


def test_no_image_is_listed_twice():
    paths = [r["path"] for r in load_rows()]
    assert len(paths) == len(set(paths))


def test_identical_files_never_sit_in_two_splits():
    rows = load_rows()
    if not all(Path(r["path"]).exists() for r in rows):
        pytest.skip("dataset images not found")
    split_of = {}
    for r in rows:
        digest = hashlib.md5(Path(r["path"]).read_bytes()).hexdigest()
        assert split_of.setdefault(digest, r["split"]) == r["split"]


def test_every_split_has_all_three_classes():
    seen = {(r["split"], r["label"]) for r in load_rows()}
    for split in ["train", "val", "test"]:
        for label in ["negative", "recoverable_defects", "non_recoverable_defects"]:
            assert (split, label) in seen


def test_the_test_split_is_locked():
    from dataset import make_loader
    with pytest.raises(ValueError):
        make_loader("test")