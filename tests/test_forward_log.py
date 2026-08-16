import json

import numpy as np
import pandas as pd

from swing_it.report.forward_log import (
    append_forward_log,
    build_records,
    read_forward_log,
    score_forward_log,
)

COLS = ["tv_ratio", "llm_specificity"]


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["034020", "005930"],
            "date": pd.to_datetime(["2026-08-14", "2026-08-14"]),
            "tv_ratio": [4.6, 3.2],
            "llm_specificity": [2.5, 0.4],
            "prediction": [0.0481, 0.0192],
        }
    )


def _records(logged_at: str = "2026-08-17", **overrides) -> list[dict]:
    kwargs = {
        "feature_columns": COLS,
        "layer": "supply_news_llm",
        "gate_verdict": "FAIL",
        "logged_at": logged_at,
        "entry_date": "2026-08-17",
        "exit_date": "2026-10-13",
    }
    kwargs.update(overrides)
    return build_records(_candidates(), **kwargs)


def test_record_captures_features_and_context():
    rec = _records()[0]
    assert rec["code"] == "034020"
    assert rec["event_date"] == "2026-08-14"
    assert rec["logged_at"] == "2026-08-17"
    assert rec["layer"] == "supply_news_llm"
    assert rec["gate_verdict"] == "FAIL"
    assert rec["rank"] == 1
    assert rec["features"] == {"tv_ratio": 4.6, "llm_specificity": 2.5}
    assert rec["entry_date_planned"] == "2026-08-17"
    assert rec["exit_date_planned"] == "2026-10-13"
    assert "code_version" in rec


def test_records_are_ranked_by_prediction():
    assert {r["code"]: r["rank"] for r in _records()} == {"034020": 1, "005930": 2}


def test_append_writes_jsonl(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    assert append_forward_log(path, _records()) == 2
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["code"] == "034020"


def test_append_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "forward_log.jsonl"
    assert append_forward_log(path, _records()) == 2
    assert path.exists()


def test_rerunning_the_same_day_writes_nothing(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records()) == 0
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_existing_record_is_never_overwritten(tmp_path):
    """전진검증의 가치는 '그때 그렇게 예측했다'가 사후에 안 바뀌는 데 있다."""
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    tampered = _records()
    tampered[0]["prediction"] = 0.9999
    append_forward_log(path, tampered)
    log = read_forward_log(path)
    assert log[log["code"] == "034020"]["prediction"].tolist() == [0.0481]


def test_a_later_day_appends_a_new_row(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records("2026-08-18")) == 2
    assert len(read_forward_log(path)) == 4


def test_a_different_layer_is_a_different_record(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records(layer="supply_news")) == 2


def test_read_missing_file_returns_empty_frame(tmp_path):
    assert read_forward_log(tmp_path / "nope.jsonl").empty


def test_score_fills_realised_returns_where_available(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    labels = pd.DataFrame(
        {
            "code": ["034020"],
            "date": pd.to_datetime(["2026-08-14"]),
            "excess_return": [0.062],
        }
    )
    scored = score_forward_log(read_forward_log(path), labels).set_index("code")
    assert scored.loc["034020", "realized_excess"] == 0.062
    assert np.isnan(scored.loc["005930", "realized_excess"])


def test_score_on_an_empty_log_is_a_noop():
    assert score_forward_log(pd.DataFrame(), pd.DataFrame()).empty
