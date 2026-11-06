import pytest
from severity import assess


def test_clearly_good_is_none():
    assert assess([0.90, 0.05, 0.05])["severity"] == "None"


def test_recoverable_is_low():
    assert assess([0.10, 0.80, 0.10])["severity"] == "Low"


def test_unsure_is_medium():
    assert assess([0.10, 0.60, 0.30])["severity"] == "Medium"


def test_non_recoverable_is_high():
    assert assess([0.10, 0.20, 0.70])["severity"] == "High"


def test_low_confidence_flags_review():
    assert assess([0.40, 0.30, 0.30])["needs_review"]


def test_confident_result_does_not_flag_review():
    assert not assess([0.90, 0.05, 0.05])["needs_review"]


def test_size_rule_is_off_by_default():
    assert assess([0.10, 0.80, 0.10], region_area_percent=30)["severity"] == "Low"


def test_size_rule_raises_one_level_when_switched_on():
    result = assess([0.10, 0.80, 0.10], 12.0, {"use_size_rule": True})
    assert result["severity"] == "Medium"


def test_rules_can_be_overridden():
    assert assess([0.10, 0.60, 0.30], rules={"high_threshold": 0.3})["severity"] == "High"


@pytest.mark.parametrize("bad", [[0.5, 0.5, 0.5], [0.9, 0.2], [-0.1, 0.6, 0.5]])
def test_bad_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        assess(bad)