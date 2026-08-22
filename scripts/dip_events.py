"""낙주 매매 — 이벤트 **규모만** 센다 (`docs/research/manju/10-dip-preregistration.md`).

  일봉   당일 저가가 고가 대비 -10% 이상인 (종목, 일자). 거래대금 하한 적용 후
         월별 건수 · 종목 수 · 그중 minute_bars 에 분봉이 있는 종목 수
  분봉   사전등록의 기계적 정의로 실제 이벤트 분(minute) 건수
         = 당일 누적 고가 대비 -10% 를 처음 찍은 분, 그 분까지 누적 거래대금 >= 50억,
           트리거 09:00~14:59

**수익률은 계산하지 않는다.** 이 스크립트는 진입 이후의 봉을 한 줄도 읽지 않는다 —
트리거 분 이후로 보는 것은 "트리거 ±3분 분봉 결측(VI 근사)" 의 존재 여부뿐이고
가격은 보지 않는다. 사전등록 전에 결과를 보는 것이 이 프로젝트의 최대 금기다.

DB 는 읽기만 한다. 분봉 이벤트는 일봉 이벤트의 부분집합이므로 (분봉 누적고가 <=
일봉 고가, 트리거 저가 >= 일봉 저가, 누적 거래대금 <= 당일 거래대금) 일봉 이벤트
(종목, 일자) 의 분봉만 읽는다.

사용:
  uv run --extra pg python scripts/dip_events.py --need-out /path/dip_need.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import pandas as pd

from scalp_it.intraday.loader import available_codes, load_minute_bars
from scalp_it.intraday.store import connect

DD_TRIGGER = 0.10            # 고점 대비 -10%
TV_FLOOR_DAILY = 5_000.0     # 50억 — daily_bars_adjusted.trade_value 는 백만원
TV_FLOOR_REF = 50_000.0      # 500억 — 참고용 (08번 문서 상따 하한)
TV_FLOOR_MINUTE = 5e9        # 50억 — minute_bars.trade_value 는 원
LAST_TRIGGER_MINUTE = 14 * 60 + 59
REGULAR_END_MINUTE = 15 * 60 + 19
GAP_WINDOW = 3               # 트리거 ±3분 안의 분봉 결측을 VI 근사로 센다


def _load_env(path: str | None) -> None:
    if not path or os.environ.get("KR_QUANT_DB"):
        return
    for line in pathlib.Path(path).read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def daily_events(conn, start: str, end: str) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        "select code, date, open, high, low, close, trade_value from daily_bars_adjusted "
        "where date >= %s and date <= %s and high > 0 and low > 0",
        (start, end),
    )
    d = pd.DataFrame(cur.fetchall(), columns=["code", "date", "open", "high", "low", "close", "tv"])
    cur.close()
    d["code"] = d["code"].astype(str).str.strip()
    d["date"] = pd.to_datetime(d["date"])
    d["tv"] = pd.to_numeric(d["tv"], errors="coerce")
    d["dd"] = d["low"] / d["high"] - 1.0
    return d[d["dd"] <= -DD_TRIGGER].copy()


def minute_days(conn) -> set[tuple[str, pd.Timestamp]]:
    """minute_bars 에 실제로 봉이 있는 (종목, 일자)."""
    cur = conn.cursor()
    cur.execute("select code, date(ts) from minute_bars group by code, date(ts)")
    out = {(str(c).strip(), pd.Timestamp(d)) for c, d in cur.fetchall()}
    cur.close()
    return out


def static_vi_sim(b: pd.DataFrame) -> tuple[int | None, bool]:
    """정적 VI 를 분봉으로 흉내 낸다 — **근사**다 (문서 '한계' 참고).

    기준가 = 시가. 어느 분의 고가가 기준가 x 1.10 이상이거나 저가가 x 0.90 이하이면
    VI 발동으로 보고, 기준가를 그 분의 종가로 바꾼다 (실제로는 2분 뒤 단일가가
    새 기준가다 — 그 값을 분봉으로는 모른다). **첫 하락 발동** 분과, 그 전에 상승
    발동이 있었는지를 돌려준다. 하락 발동이 없으면 (None, False).
    """
    ref = float(b.at[0, "open"])
    seen_up = False
    for j in range(len(b)):
        lo, hi = float(b.at[j, "low"]), float(b.at[j, "high"])
        if lo <= ref * 0.90:
            return int(b.at[j, "minute"]), seen_up
        if hi >= ref * 1.10:
            seen_up = True
            ref = float(b.at[j, "close"])
    return None, seen_up


def minute_events(conn, pairs: pd.DataFrame) -> pd.DataFrame:
    """일봉 이벤트 (종목, 일자) 의 분봉만 읽어 기계적 정의로 트리거 분을 찾는다.

    주 정의(A): 당일 누적 고가 대비 -10% 첫 도달 분.
    대안 정의(B): 정적 VI 시뮬레이션의 첫 하락 발동 분 (`static_vi_sim`).
    둘 다 누적 거래대금 >= 50억, 트리거 09:00~14:59 를 같은 방식으로 적용한다.
    """
    rows = []
    for date, grp in pairs.groupby("date"):
        day = date.strftime("%Y-%m-%d")
        bars = load_minute_bars(conn, day, day, codes=grp["code"].tolist())
        if bars.empty:
            continue
        bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
        bars = bars[(bars["minute"] >= 9 * 60) & (bars["minute"] <= REGULAR_END_MINUTE)]
        for code, b in bars.groupby("code", sort=False):
            b = b.sort_values("ts").reset_index(drop=True)
            run_high = b["high"].cummax()
            cum_tv = b["trade_value"].cumsum()
            minute_to_idx = {int(m): k for k, m in enumerate(b["minute"].tolist())}
            row = {"code": code, "date": date}

            # --- 대안 정의 B: 정적 VI 시뮬레이션 ---
            vi_minute, vi_after_up = static_vi_sim(b)
            row["vi_b_minute"] = vi_minute
            row["vi_b_after_up"] = vi_after_up
            row["vi_b_event"] = bool(
                vi_minute is not None
                and vi_minute <= LAST_TRIGGER_MINUTE
                and cum_tv.iat[minute_to_idx[vi_minute]] >= TV_FLOOR_MINUTE
            )

            # --- 주 정의 A: 고점 대비 -10% 첫 도달 ---
            hit = b["low"] <= run_high * (1.0 - DD_TRIGGER)
            if not hit.any():
                rows.append({**row, "status": "no_touch"})
                continue
            i = int(hit.idxmax())
            trig_minute = int(b.at[i, "minute"])
            row["trig_minute"] = trig_minute
            if cum_tv.iat[i] < TV_FLOOR_MINUTE:
                rows.append({**row, "status": "below_tv_floor"})
                continue
            if trig_minute > LAST_TRIGGER_MINUTE:
                rows.append({**row, "status": "after_1459"})
                continue
            # 트리거 시점까지의 정보만. 가격 결과는 보지 않는다.
            high_idx = int(b["high"].iloc[: i + 1].idxmax())
            lo_m = max(9 * 60, trig_minute - GAP_WINDOW)
            hi_m = min(REGULAR_END_MINUTE, trig_minute + GAP_WINDOW)
            missing = [m for m in range(lo_m, hi_m + 1) if m not in minute_to_idx]
            gap2 = any(m + 1 in missing for m in missing)      # 2분 이상 연속 결측
            day_open = float(b.at[0, "open"])
            rows.append({
                **row, "status": "event",
                "minutes_from_high": trig_minute - int(b.at[high_idx, "minute"]),
                "cum_tv_bn": float(cum_tv.iat[i]) / 1e8,          # 억원
                "gap2_near": bool(gap2),
                "static_vi_proxy": bool(float(b.at[i, "close"]) <= day_open * 0.90),
                "vi_b_near": bool(vi_minute is not None and abs(vi_minute - trig_minute) <= GAP_WINDOW),
                "n_bars_before": i + 1,
            })
    return pd.DataFrame(rows)


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--env-file", default="/home/young/Documents/git/kr-quant/.env")
    ap.add_argument("--need-out", required=True, help="분봉 없는 종목 {code: YYYYMMDD} JSON")
    ap.add_argument("--summary-out", default=None, help="요약 JSON (선택)")
    args = ap.parse_args()
    _load_env(args.env_file)
    conn = connect()

    ev = daily_events(conn, args.start, args.end)
    ev["month"] = ev["date"].dt.strftime("%Y-%m")
    avail = set(available_codes(conn))
    mdays = minute_days(conn)

    def block(tv_floor: float, label: str) -> pd.DataFrame:
        e = ev[ev["tv"] >= tv_floor].copy()
        e["has_code"] = e["code"].isin(avail)
        e["has_day"] = [((c, d) in mdays) for c, d in zip(e["code"], e["date"])]
        print(f"\n## 일봉 이벤트 — 거래대금 >= {label}")
        print(f"건수 {len(e):,} · 종목 {e['code'].nunique():,} · "
              f"거래일당 {len(e) / max(1, ev['date'].nunique()):.2f}건")
        print(f"분봉 있는 종목 {e.loc[e['has_code'], 'code'].nunique():,} / "
              f"그 종목의 건수 {int(e['has_code'].sum()):,} / "
              f"그중 해당 일자 분봉도 있는 건수 {int(e['has_day'].sum()):,}")
        m = e.groupby("month").agg(n=("code", "size"), codes=("code", "nunique"),
                                   n_minute=("has_day", "sum"))
        print("\n| 월 | 건수 | 종목 | 분봉 있는 건수 |\n|---|---|---|---|")
        for mo, r in m.iterrows():
            print(f"| {mo} | {int(r.n):,} | {int(r.codes):,} | {int(r.n_minute):,} |")
        return e

    e50 = block(TV_FLOOR_DAILY, "50억 (사전등록 하한)")
    block(TV_FLOOR_REF, "500억 (참고)")

    # 필요 데이터 — 분봉이 전혀 없는 종목, 최초 이벤트일 - 10일
    need = (e50[~e50["has_code"]].groupby("code")["date"].min() - pd.Timedelta(days=10))
    need_json = {c: d.strftime("%Y%m%d") for c, d in need.items()}
    pathlib.Path(args.need_out).write_text(json.dumps(need_json, ensure_ascii=False, indent=0, sort_keys=True))
    partial = e50[e50["has_code"] & ~e50["has_day"]]
    print(f"\n분봉 없는 종목 {len(need_json):,}개 → {args.need_out}")
    print(f"분봉은 있으나 해당 일자가 빠진 건수 {len(partial):,} (종목 {partial['code'].nunique():,})")

    # 분봉 이벤트 — 일봉 이벤트 중 분봉이 있는 (종목, 일자) 만
    pairs = e50[e50["has_day"]][["code", "date"]].drop_duplicates()
    me = minute_events(conn, pairs)
    st = me["status"].value_counts() if not me.empty else pd.Series(dtype=int)
    print("\n## 분봉 이벤트 — 기계적 정의 (고점 대비 -10% 첫 도달 분, 누적 거래대금 >= 50억, 09:00~14:59)")
    print(f"검사한 (종목,일자) {len(pairs):,}")
    for k in ("event", "below_tv_floor", "after_1459", "no_touch"):
        print(f"  {k}: {int(st.get(k, 0)):,}")
    evm = me[me["status"] == "event"].copy() if not me.empty else pd.DataFrame()
    summary = {"daily_50": len(e50), "need": len(need_json), "minute_checked": len(pairs),
               "minute_status": {k: int(v) for k, v in st.items()}}
    if not evm.empty:
        evm["month"] = evm["date"].dt.strftime("%Y-%m")
        print(f"이벤트 {len(evm):,} · 종목 {evm['code'].nunique():,} · "
              f"거래일당 {len(evm) / max(1, evm['date'].nunique()):.2f}건 (이벤트 있는 날 기준) · "
              f"같은 날 2건 이상인 날 {int((evm.groupby('date').size() >= 2).sum()):,}")
        m = evm.groupby("month").agg(n=("code", "size"), codes=("code", "nunique"))
        print("\n| 월 | 이벤트 | 종목 |\n|---|---|---|")
        for mo, r in m.iterrows():
            print(f"| {mo} | {int(r.n):,} | {int(r.codes):,} |")
        hour = (evm["trig_minute"] // 60).value_counts().sort_index()
        print("\n| 트리거 시각대 | 건수 |\n|---|---|")
        for h, n in hour.items():
            print(f"| {int(h):02d}시 | {int(n):,} |")
        q = evm["minutes_from_high"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print("\n고점→트리거 경과 분 p10/p25/p50/p75/p90: " + " / ".join(f"{v:.0f}" for v in q))
        print(f"경과 <= 30분 {int((evm['minutes_from_high'] <= 30).sum()):,} · "
              f"<= 60분 {int((evm['minutes_from_high'] <= 60).sum()):,} · "
              f"> 120분 {int((evm['minutes_from_high'] > 120).sum()):,}")
        print(f"트리거 ±{GAP_WINDOW}분 내 분봉 2분 이상 연속 결측 (VI 근사) "
              f"{int(evm['gap2_near'].sum()):,} ({evm['gap2_near'].mean():.1%})")
        print(f"트리거 분 종가 <= 시가 x 0.90 (정적 VI 근사) "
              f"{int(evm['static_vi_proxy'].sum()):,} ({evm['static_vi_proxy'].mean():.1%})")
        print(f"주 정의 트리거 ±{GAP_WINDOW}분 내에 정적 VI 시뮬레이션(B) 하락 발동이 있는 이벤트 "
              f"{int(evm['vi_b_near'].sum()):,} ({evm['vi_b_near'].mean():.1%})")
        qt = evm["cum_tv_bn"].quantile([0.25, 0.5, 0.75])
        print("트리거까지 누적 거래대금(억) p25/p50/p75: " + " / ".join(f"{v:,.0f}" for v in qt))
        summary.update({"minute_events": len(evm), "minute_codes": int(evm["code"].nunique()),
                        "by_month": {k: int(v) for k, v in m["n"].items()},
                        "by_hour": {int(k): int(v) for k, v in hour.items()},
                        "gap2_near": float(evm["gap2_near"].mean()),
                        "static_vi_proxy": float(evm["static_vi_proxy"].mean()),
                        "minutes_from_high_q": {str(k): float(v) for k, v in q.items()}})
    vb = me[me["vi_b_event"]] if not me.empty else pd.DataFrame()
    print("\n## 대안 정의 B — 정적 VI 시뮬레이션 첫 하락 발동 (같은 하한·시간대)")
    if not vb.empty:
        print(f"이벤트 {len(vb):,} · 종목 {vb['code'].nunique():,} · "
              f"그중 상승 VI 뒤의 하락 발동(급등 후 낙주) {int(vb['vi_b_after_up'].sum()):,} · "
              f"주 정의 A 이벤트와 같은 (종목,일자) {int((vb['status'] == 'event').sum()):,}")
        vb_month = vb.assign(month=vb["date"].dt.strftime("%Y-%m")).groupby("month").size()
        print("월별: " + ", ".join(f"{k} {v:,}" for k, v in vb_month.items()))
        summary.update({"vi_b_events": len(vb), "vi_b_after_up": int(vb["vi_b_after_up"].sum()),
                        "vi_b_codes": int(vb["code"].nunique()),
                        "vi_b_overlap_A": int((vb["status"] == "event").sum())})
    if args.summary_out:
        pathlib.Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
