import json
from datetime import datetime
from history import delete_all, list_inspections, save_inspection

RULES = {"defect_threshold": 0.5}


def fake_result(severity="None", p_good=0.9, when=datetime(2026, 10, 21, 18, 0, 0)):
    rest = (1 - p_good) / 2
    return {"predicted_class": "negative",
            "verdict": {"severity": severity, "needs_review": False},
            "class_probs": {"negative": p_good, "recoverable_defects": rest,
                            "non_recoverable_defects": rest},
            "localized": False, "cam_area_percent": 0.0, "measurements": [],
            "method": "test", "inspected_at": when}


def test_empty_database_lists_nothing(tmp_path):
    assert list_inspections(db_file=tmp_path / "t.db") == []


def test_saved_values_come_back(tmp_path):
    db = tmp_path / "t.db"
    assert save_inspection(fake_result(), "a.jpg", RULES, db_file=db) == 1
    row = list_inspections(db_file=db)[0]
    assert row["file_name"] == "a.jpg"
    assert row["severity"] == "None"
    assert row["p_good"] == 0.9
    assert row["inspected_at"] == "2026-10-21 18:00:00"
    assert json.loads(row["rules_json"]) == RULES


def test_hostile_file_name_is_stored_as_plain_text(tmp_path):
    db = tmp_path / "t.db"
    hostile = "x'); DROP TABLE inspections;--.jpg"
    save_inspection(fake_result(), hostile, RULES, db_file=db)
    rows = list_inspections(db_file=db)
    assert rows[0]["file_name"] == hostile


def test_newest_first_and_limit(tmp_path):
    db = tmp_path / "t.db"
    save_inspection(fake_result("None"), "a.jpg", RULES, db_file=db)
    save_inspection(fake_result("High", 0.1), "b.jpg", RULES, db_file=db)
    rows = list_inspections(db_file=db)
    assert [r["severity"] for r in rows] == ["High", "None"]
    assert len(list_inspections(limit=1, db_file=db)) == 1


def test_delete_all_empties_the_history(tmp_path):
    db = tmp_path / "t.db"
    save_inspection(fake_result(), "a.jpg", RULES, db_file=db)
    delete_all(db_file=db)
    assert list_inspections(db_file=db) == []