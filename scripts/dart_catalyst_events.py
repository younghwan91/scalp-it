"""DART 재료 필터 — 이벤트 규모 조사 + 공용 태깅 로직.

사전등록: `docs/research/manju/21-dart-catalyst-preregistration.md`.

재료성 공시 = `report_nm` 이 화이트리스트 키워드(정정 마커 제거 후 부분일치)에 걸리는 공시.
버킷(`material_type`)이 아니라 제목 키워드로 정한다 — 버킷은 대량보유·ELS 일괄신고·IR 등
루틴으로 오염돼 있고, 임상·품목허가는 `투자판단관련주요경영사항`(재량공시)에 숨어 버킷으로 못 잡는다.

접수시각(`rcept_dt`)은 날짜만이라 룩어헤드 방지를 위해 **진입일 당일 공시는 판정 제외**,
**직전 K거래일 창**의 공시만 `cat_prior` 로 쓴다. 당일 공시는 `cat_same` 로 분리 보고.

이 모듈은 `verify_dart_catalyst.py` 가 재사용한다. DB 는 읽기만 한다.

실행(규모 조사):
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/dart_catalyst_events.py \
        --breakout-cache /path/bo.pkl --sangtta-cache /path/st.pkl \
        --dart-db /abs/data/dart_2024.db --dart-db /abs/data/dart.db
"""

from __future__ import annotations

import argparse
import bisect
import os
import sqlite3

import numpy as np
import pandas as pd

# ---- 화이트리스트 (사전등록 21번. 측정 후 변경 금지) --------------------------------------
K_TRADING_DAYS = 3

CORRECTION_MARKERS = ("[기재정정]", "[첨부정정]", "[첨부추가]", "[정정]")

# 위에서 아래로 처음 맞는 것을 서브유형 라벨로 쓴다(불리언은 어느 하나라도 걸리면 True).
# '최대주주변경'을 'M&A'(주식양수도) 보다 위에 둬서 최대주주변경 수반 양수도가 그쪽으로 라벨된다.
CATALYST_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("수주", ("공급계약",)),
    ("유상증자", ("유상증자결정",)),
    ("무상증자", ("무상증자결정",)),
    ("전환사채", ("전환사채권발행결정",)),
    ("신주인수권부사채", ("신주인수권부사채권발행결정",)),
    ("교환사채", ("교환사채권발행결정",)),
    ("자사주", ("자기주식취득", "자기주식처분", "주식소각")),
    ("실적잠정", ("영업(잠정)실적", "매출액또는손익구조")),
    ("임상허가특허", ("임상", "품목허가", "특허")),
    ("최대주주변경", ("최대주주변경",)),
    ("감자", ("감자결정",)),
    ("시설투자", ("신규시설투자",)),
    ("M&A", ("회사합병결정", "분할", "영업양수", "영업양도", "주식양수도")),
    ("투자판단", ("투자판단관련주요경영사항",)),
]


def classify_catalyst(report_nm: str) -> str | None:
    """재료성 공시면 서브유형 문자열, 아니면 None."""
    text = str(report_nm or "")
    for marker in CORRECTION_MARKERS:
        text = text.replace(marker, "")
    for subtype, keywords in CATALYST_RULES:
        if any(k in text for k in keywords):
            return subtype
    return None


def load_disclosures(db_paths: list[str], start: str, end: str) -> pd.DataFrame:
    """여러 DART sqlite 를 합쳐 [start,end] 재료성 공시만: code·rcept·subtype. rcept_no 로 중복 제거."""
    frames = []
    for path in db_paths:
        if not os.path.exists(path):
            continue
        con = sqlite3.connect(path)
        df = pd.read_sql_query(
            "SELECT rcept_no, rcept_dt, code, report_nm FROM disclosures "
            "WHERE rcept_dt >= ? AND rcept_dt <= ?",
            con, params=(start, end))
        con.close()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["code", "rcept", "subtype"])
    d = pd.concat(frames, ignore_index=True).drop_duplicates("rcept_no")
    d["subtype"] = [classify_catalyst(v) for v in d["report_nm"]]
    d = d[d["subtype"].notna()].copy()
    d["code"] = d["code"].astype(str).str.strip().str.zfill(6)
    d["rcept"] = pd.to_datetime(d["rcept_dt"], format="%Y%m%d")
    return d[["code", "rcept", "subtype"]].reset_index(drop=True)


def trading_days(conn, start: str, end: str) -> list[pd.Timestamp]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT date FROM daily_bars WHERE date >= %s AND date <= %s ORDER BY date",
                (start, end))
    days = [pd.Timestamp(r[0]) for r in cur.fetchall()]
    cur.close()
    return days


def tag_events(ev: pd.DataFrame, disc: pd.DataFrame, tdays: list[pd.Timestamp],
               k: int = K_TRADING_DAYS) -> pd.DataFrame:
    """이벤트에 cat_prior(직전 k거래일 재료)·cat_same(당일 재료)·prior_subtypes 를 붙인다."""
    ev = ev.copy()
    ev["code"] = ev["code"].astype(str).str.strip().str.zfill(6)
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    tpos = {d: i for i, d in enumerate(tdays)}

    # code -> 정렬된 (rcept 리스트, subtype 리스트)
    by_code: dict[str, tuple[list, list]] = {}
    for code, g in disc.sort_values("rcept").groupby("code", sort=False):
        by_code[code] = (list(g["rcept"]), list(g["subtype"]))

    prior = np.zeros(len(ev), dtype=int)
    same = np.zeros(len(ev), dtype=int)
    subtypes: list[str] = [""] * len(ev)
    for i, (code, d) in enumerate(zip(ev["code"].to_numpy(), ev["date"].to_numpy(), strict=True)):
        rec = by_code.get(code)
        if rec is None:
            continue
        d = pd.Timestamp(d)
        pos = tpos.get(d)
        if pos is None:               # 진입일이 거래일 집합 밖(창 경계) — 보수적으로 무재료 취급
            continue
        win_start = tdays[max(0, pos - k)]
        rlist, slist = rec
        lo = bisect.bisect_left(rlist, win_start)
        hi = bisect.bisect_left(rlist, d)          # rcept < d (당일 제외)
        subs = {slist[j] for j in range(lo, hi)}
        if subs:
            prior[i] = 1
            subtypes[i] = "|".join(sorted(subs))
        # 당일 공시
        hi2 = bisect.bisect_right(rlist, d)
        if hi2 > hi:
            same[i] = 1
    ev["cat_prior"] = prior
    ev["cat_same"] = same
    ev["prior_subtypes"] = subtypes
    return ev


def _load_env(path: str | None) -> None:
    path = path or "/home/young/Documents/git/kr-quant/.env"
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _scale_report(name: str, ev: pd.DataFrame, out: list) -> None:
    n = len(ev)
    p = ev["cat_prior"].sum()
    s = ev["cat_same"].sum()
    out.append(f"\n## {name} — 이벤트 {n:,}건")
    out.append(f"- 재료有(직전{K_TRADING_DAYS}거래일, cat_prior): **{p:,}건 ({p/n:.1%})**")
    out.append(f"- 당일 공시(cat_same, 분리 보고): {s:,}건 ({s/n:.1%})")
    # 서브유형 분포 (중복 계수)
    from collections import Counter
    c: Counter = Counter()
    for s_ in ev.loc[ev["cat_prior"] == 1, "prior_subtypes"]:
        for t in s_.split("|"):
            if t:
                c[t] += 1
    out.append("- 재료有 서브유형 분포(중복 계수): "
               + ", ".join(f"{k} {v}" for k, v in c.most_common()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakout-cache", required=True)
    ap.add_argument("--sangtta-cache", required=True)
    ap.add_argument("--dart-db", action="append", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--start", default="20250701")
    ap.add_argument("--end", default="20260831")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _load_env(args.env)
    from scalp_it.adapters.db import connect

    disc = load_disclosures(args.dart_db, args.start, args.end)
    conn = connect()
    tdays = trading_days(conn, "2025-06-01", "2026-08-31")
    conn.close()

    out: list[str] = [f"# DART 재료 필터 — 이벤트 규모 (K={K_TRADING_DAYS}거래일)"]
    out.append(f"재료성 공시 {len(disc):,}건 · 거래일 {len(tdays)}일 "
               f"· DART 범위 {disc['rcept'].min().date()}~{disc['rcept'].max().date()}")

    bo = pd.read_pickle(args.breakout_cache)
    bo = bo["ev"] if isinstance(bo, dict) else bo
    st = pd.read_pickle(args.sangtta_cache)
    st = st["ev"] if isinstance(st, dict) else st
    bo = tag_events(bo, disc, tdays)
    st = tag_events(st, disc, tdays)
    _scale_report("돌파(breakout)", bo, out)
    _scale_report("상따(sangtta)", st, out)

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
