"""운용=백테스트 검증 — 실시간 스트림 피처와 배치 피처가 같은 값인지 실데이터로 대조한다.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    ~/of80/.rlvenv/bin/python scripts/parity/feature_parity.py 2026-09-15 --codes 5

core `backtest.lob` 의 `build_second_grid`(배치)와 `SecondFeatureStream`(실시간 증분)이 **같은 입력에 같은 값**을 내야
정책을 실매매에 올릴 수 있다. core 가 37피처(FEATURE_SET_V2)를 내면 `--v2` 로 같은 검사를 돌린다.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="실시간 스트림 vs 배치 피처 일치 검사")
    ap.add_argument("day")
    ap.add_argument("--codes", type=int, default=5, help="검사할 종목 수(틱 많은 순)")
    ap.add_argument("--v2", action="store_true", help="core 가 FEATURE_SET_V2 를 내면 그걸로 검사")
    a = ap.parse_args()

    import psycopg2
    from krx_quant_core.backtest import lob

    stream_cls = getattr(lob, "SecondFeatureStreamV2", None) if a.v2 else lob.SecondFeatureStream
    if stream_cls is None:
        print("core 에 SecondFeatureStreamV2 가 아직 없다 — v0.3.0/v0.5.x 이관 대기")
        return 2
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    nxt = (pd.Timestamp(a.day) + pd.Timedelta(days=1)).date().isoformat()
    top = pd.read_sql("select code, count(*) n from ticks where ts >= %s and ts < %s group by 1 "
                      "order by n desc limit %s", conn, params=(a.day, nxt, a.codes))
    bad = 0
    for code in top.code:
        tk = pd.read_sql("select code, ts, seq, price, volume, side, strength, best_bid, best_ask from ticks "
                         "where code = %s and ts >= %s and ts < %s order by ts, seq", conn, params=(code, a.day, nxt))
        qt = pd.read_sql("select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
                         "from quotes where code = %s and ts >= %s and ts < %s order by ts",
                         conn, params=(code, a.day, nxt))
        feat, _path = lob.build_second_grid(tk, qt)
        inputs, _ = lob.aggregate_seconds(tk, qt)
        stream = stream_cls()
        live = pd.DataFrame([np.array(stream.update_array(x), copy=True) for x in inputs],  # 내부 버퍼 재사용 주의
                            columns=list(lob.STEP_COLUMNS))
        rows = feat.index.to_numpy()                       # 결정 행의 초 인덱스
        assert rows.max() < len(live), "초 격자 길이가 다르다 — 입력 범위 불일치"
        cols = [c for c in feat.columns if c in live.columns]
        assert cols, "비교할 공통 열이 없다 — 열 이름 규약이 어긋났다"
        sub = live.iloc[rows][cols]
        # 배치는 float32 로 저장하고 스트림은 float64 를 돌려준다 — 같은 자료형으로 맞춰 비교한다.
        diff = []
        for c in cols:
            ref = feat[c].to_numpy()
            got = sub[c].to_numpy().astype(ref.dtype)
            if not np.array_equal(ref, got, equal_nan=True):
                diff.append(c)
        print(f"{code}: 초 {len(inputs):,} · 결정행 {len(feat):,} · 비교열 {len(cols)} · 불일치 {len(diff)}"
              + (f" → {diff[:5]}" if diff else ""))
        bad += len(diff)
    conn.close()
    print("\n판정:", "일치 — 운용과 백테스트가 같은 값을 낸다" if bad == 0 else f"불일치 {bad}열 — 실매매 승격 불가")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
