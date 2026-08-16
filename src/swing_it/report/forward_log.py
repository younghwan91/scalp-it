"""전진 기록 — 오늘의 후보와 그때의 피처를 영구 저장한다 (스펙 §9).

6개월 뒤 실전 검증을 가능하게 하는 유일한 장치다. walk-forward 는 과거
데이터를 시간순으로 잘라 쓴 것이라 '손 안 댄 창'이라고 부르기 어렵다.
오늘 기록해 6개월 뒤 채점하는 것만이 진짜 out-of-sample 이다.

**append-only.** 같은 (logged_at, layer, code, event_date) 는 조용히 건너뛴다.
모델을 고쳐 다시 돌려도 과거 예측은 남는다. 사후에 바뀔 수 있는 기록은
전진검증 자산이 아니다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

LOG_KEY: tuple[str, ...] = ("logged_at", "layer", "code", "event_date")


def build_records(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    layer: str,
    gate_verdict: str,
    logged_at: str,
    entry_date: str,
    exit_date: str,
    seed: int = 0,
    version: str | None = None,
) -> list[dict]:
    """후보 프레임을 전진 기록 레코드 목록으로 바꾼다."""
    from .. import __version__

    columns = list(feature_columns)
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    return [
        {
            "logged_at": logged_at,
            "layer": layer,
            "code": str(row["code"]),
            "event_date": str(pd.Timestamp(row["date"]).date()),
            "rank": rank,
            "prediction": float(row["prediction"]),
            "gate_verdict": gate_verdict,
            "entry_date_planned": entry_date,
            "exit_date_planned": exit_date,
            "features": {col: float(row[col]) for col in columns},
            "model_seed": int(seed),
            "code_version": version or __version__,
        }
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1)
    ]


def _key(record: dict) -> tuple:
    return tuple(record[field] for field in LOG_KEY)


def append_forward_log(path: str | Path, records: Sequence[dict]) -> int:
    """중복을 건너뛰고 JSONL 에 덧붙인다. 새로 쓴 건수를 돌려준다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple] = set()
    if target.exists():
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    seen.add(_key(json.loads(line)))

    written = 0
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            key = _key(record)
            if key in seen:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1
    return written


def read_forward_log(path: str | Path) -> pd.DataFrame:
    """전진 기록을 프레임으로 읽는다. 없으면 빈 프레임."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def score_forward_log(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """기록된 후보에 실현 초과수익을 붙인다. 아직 안 나온 건 ``NaN``.

    Args:
        labels: ``[code, date, excess_return]``. ``date`` 는 이벤트일.
    """
    if log.empty:
        return log
    out = log.copy()
    out["_event_date"] = pd.to_datetime(out["event_date"])
    realized = labels[["code", "date", "excess_return"]].rename(
        columns={"date": "_event_date", "excess_return": "realized_excess"}
    )
    realized["_event_date"] = pd.to_datetime(realized["_event_date"])
    out = out.merge(realized, on=["code", "_event_date"], how="left")
    return out.drop(columns=["_event_date"])
