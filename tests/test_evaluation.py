import numpy as np
import pytest
from evaluation import (
    auc, auc_interval, balanced_accuracy, confusion_counts, iou_dice, precision_recall_f1,
)


def test_confusion_counts():
    assert confusion_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]) == (2, 1, 1, 1)


def test_precision_recall_f1():
    precision, recall, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
    assert precision == pytest.approx(0.8)
    assert recall == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)


def test_no_division_by_zero():
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_balanced_accuracy():
    assert balanced_accuracy(tp=6, fp=1, fn=4, tn=9) == pytest.approx(0.75)


def test_auc_perfect_tied_and_reversed():
    assert auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_auc_interval_is_a_valid_range():
    low, high = auc_interval([0.9, 0.8, 0.7, 0.6, 0.4], [0.5, 0.3, 0.2, 0.1, 0.65], n_boot=100)
    assert 0.0 <= low <= high <= 1.0


def test_iou_dice_identical_masks():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    assert iou_dice(mask, mask) == (1.0, 1.0)


def test_iou_dice_disjoint_masks():
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.uint8)
    a[0:5, 0:5] = 255
    b[10:15, 10:15] = 255
    assert iou_dice(a, b) == (0.0, 0.0)


def test_iou_dice_half_overlap():
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.uint8)
    a[0:10, 0:10] = 255
    b[0:10, 5:15] = 255            # overlap 50, union 150
    iou, dice = iou_dice(a, b)
    assert iou == pytest.approx(1 / 3)
    assert dice == pytest.approx(0.5)


def test_two_empty_masks_count_as_perfect_agreement():
    empty = np.zeros((20, 20), dtype=np.uint8)
    assert iou_dice(empty, empty) == (1.0, 1.0)