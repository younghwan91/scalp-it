"""청산 재측정 — 사전등록 `docs/research/manju/18-exit-remeasurement-preregistration.md` 그대로.

진입 이벤트는 **기존 확정 검증 스크립트의 이벤트 생성부를 그대로 import** 해 같은 집합을
만든다(재선별·피처·임계 변경 없음). 각 진입에 대해 진입 분 다음 봉부터 15:19 까지 분봉으로
MFE·MAE 와 청산 9전략(트레일링 3 · 고정목표 3 · 3:3:4 · 기존 · 종가)을 잰다.

  상따   verify_sangtta.build_events
  돌파   verify_breakout.{prepare_daily,_key_frame,events_from_batch}
  낙주   verify_dip.build_events
  짝꿍   verify_pair.collect (주 규칙 main) + 후속주 분봉 재로드

1분봉 해상도 한계 → 같은 분 손절 우선(보수). 상한가 이상 매도가는 상한가로 캡.
DB 는 읽기만 한다. 실행:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/verify_exits.py --tech all --out <scratchpad>/exit_result.md
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from limitup_timing import _load_env, limit_up_price

from scalp_it.adapters.db import load_ohlcv
from scalp_it.intraday.loader import available_codes, load_minute_bars
from scalp_it.intraday.store import connect

COST, COST_HI = 0.0034, 0.0055
OPEN_MIN, CONT_LAST_MIN = 540, 919      # 09:00 ~ 15:19 (마지막 접속매매 봉)
TRAIL_KS = (0.03, 0.05, 0.08)
TGT_TS = (0.05, 0.08, 0.10)
SPLIT_STOP = 0.02
SPLIT_T1, SPLIT_T2 = 0.03, 0.05         # 3:3:4 목표 (측정 전 고정)
SPLIT_W = (0.3, 0.3, 0.4)
NEG_MONTHS_MAX = 3
MFE_PROMOTE, MFE_REJECT = 0.05, 0.02

# 청산 9전략 컬럼명 (기존·종가 포함)
STRATS = ["trail3", "trail5", "trail8", "tgt5", "tgt8", "tgt10", "split334", "existing", "close"]
NEW_STRATS = ["trail3", "trail5", "trail8", "tgt5", "tgt8", "tgt10", "split334", "close"]


# ─────────────────────────────────────────────────────────── 청산 시뮬레이션
def sim_one(o, h, l, c, entry: float, limit: float) -> dict:
    """진입 이후 측정창(o/h/l/c 배열)에서 MFE·MAE 와 신규 8전략 수익을 계산.

    보수(1분봉): 같은 봉 손절 우선(저가 먼저). 매도가 상한가 이상이면 상한가로 캡.
    """
    n = len(c)
    if n == 0:
        return {"mfe": 0.0, "mae": 0.0, "limit_suspect": False, "is_limit_close": False,
                **{s: 0.0 for s in NEW_STRATS}}
    hi, lo = float(h.max()), float(l.min())
    hc = float(c[-1])
    cap = lambda px: min(px, limit)

    # 트레일링 k
    trail = {}
    for k, name in zip(TRAIL_KS, ("trail3", "trail5", "trail8")):
        peak, sold = entry, None
        for j in range(n):
            lvl = peak * (1 - k)
            if l[j] <= lvl:
                sold = min(lvl, o[j])       # 갭 관통이면 시가 체결(더 낮음)
                break
            peak = max(peak, h[j])
        trail[name] = (sold if sold is not None else hc) / entry - 1

    # 고정 목표 t + 손절 −2%
    tgt = {}
    for t, name in zip(TGT_TS, ("tgt5", "tgt8", "tgt10")):
        sl, tp = entry * (1 - SPLIT_STOP), entry * (1 + t)
        sold = None
        for j in range(n):
            if l[j] <= sl:
                sold = min(sl, o[j]); break
            if h[j] >= tp:
                sold = cap(tp); break
        tgt[name] = (sold if sold is not None else cap(hc)) / entry - 1

    # 3:3:4 분할 (30%@+3% · 30%@+5% · 40%@시한종가, 미체결분 −2% 하드손절)
    sl = entry * (1 - SPLIT_STOP)
    t1, t2 = entry * (1 + SPLIT_T1), entry * (1 + SPLIT_T2)
    leg1 = leg2 = None
    stop_px = None
    for j in range(n):
        if l[j] <= sl:
            stop_px = min(sl, o[j]); break
        if leg1 is None and h[j] >= t1:
            leg1 = cap(t1)
        if leg2 is None and h[j] >= t2:
            leg2 = cap(t2)
    if stop_px is not None:
        p1 = leg1 if leg1 is not None else stop_px
        p2 = leg2 if leg2 is not None else stop_px
        p3 = stop_px
    else:
        p1 = leg1 if leg1 is not None else cap(hc)
        p2 = leg2 if leg2 is not None else cap(hc)
        p3 = cap(hc)
    split = (SPLIT_W[0] * (p1 / entry - 1) + SPLIT_W[1] * (p2 / entry - 1)
             + SPLIT_W[2] * (p3 / entry - 1))

    return {"mfe": hi / entry - 1, "mae": lo / entry - 1,
            "limit_suspect": hi >= limit - 1e-6, "is_limit_close": hc >= limit - 1e-6,
            **trail, **tgt, "split334": split,
            "close": cap(hc) / entry - 1}


def path_index(bars: pd.DataFrame, keys: set | None = None) -> dict:
    """(code, date_normalized) → (minute, o, h, l, c) 배열. 접속매매(09:00~15:19)만.

    keys 가 주어지면 그 (code, date) 만 만든다(이벤트 있는 것만).
    """
    b = bars.copy()
    b["minute"] = b["ts"].dt.hour * 60 + b["ts"].dt.minute
    b["date"] = b["ts"].dt.normalize()
    b = b[(b["minute"] >= OPEN_MIN) & (b["minute"] <= CONT_LAST_MIN)]
    if keys is not None:
        b = b[[(c, d) in keys for c, d in zip(b["code"], b["date"])]]
    b = b.sort_values(["code", "date", "minute"])
    out = {}
    for (code, date), g in b.groupby(["code", "date"], sort=False):
        out[(code, date)] = (g["minute"].to_numpy(), g["open"].to_numpy(float),
                             g["high"].to_numpy(float), g["low"].to_numpy(float),
                             g["close"].to_numpy(float))
    return out


def measure(events: list[dict], pidx: dict) -> list[dict]:
    """events: [{code,date,entry,entry_min,limit,existing_ret,include_entry_bar, ...}]."""
    rows = []
    for ev in events:
        key = (ev["code"], ev["date"])
        p = pidx.get(key)
        if p is None:
            r = {"mfe": np.nan, "mae": np.nan, "limit_suspect": False, "is_limit_close": False,
                 **{s: np.nan for s in NEW_STRATS}}
        else:
            minute, o, h, l, c = p
            if ev.get("include_entry_bar"):
                m = minute >= ev["entry_min"]
            else:
                m = minute > ev["entry_min"]
            r = sim_one(o[m], h[m], l[m], c[m], ev["entry"], ev["limit"])
        r["existing"] = ev["existing_ret"]
        for kk in ("code", "date", "entry", "entry_min"):
            r[kk] = ev[kk]
        rows.append(r)
    return rows


# ─────────────────────────────────────────────────────────── 기법별 이벤트 생성
def gen_sangtta(conn) -> list[dict]:
    import verify_sangtta as vs
    codes = available_codes(conn)
    print(f"[상따] 분봉 {len(codes)} 종목", file=sys.stderr)
    daily = load_ohlcv(conn, "2025-06-01", "2026-08-21")
    daily = daily[daily["code"].isin(codes)]
    out = []
    BATCH = 40
    t0 = time.time()
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i + BATCH]
        bars = load_minute_bars(conn, "2025-08-01", "2026-08-21", codes=chunk)
        if bars.empty:
            continue
        b2 = bars.rename(columns={"trade_value": "trade_value_bar"})
        ev = vs.build_events(b2, daily[daily["code"].isin(chunk)])
        if len(ev):
            evs = [{"code": r.code, "date": r.date, "entry": float(r.entry),
                    "entry_min": int(r.ts.hour * 60 + r.ts.minute),
                    "limit": limit_up_price(float(r.prev_close)),
                    "existing_ret": float(r.ret), "include_entry_bar": False}
                   for r in ev.itertuples()]
            pidx = path_index(bars, {(e["code"], e["date"]) for e in evs})
            out.extend(measure(evs, pidx))
        print(f"  [상따] 배치 {i // BATCH + 1}: 이벤트 {len(ev):,} (누적 {len(out):,}, {time.time() - t0:.0f}s)",
              file=sys.stderr)
        del bars
    return out


def gen_breakout(conn) -> list[dict]:
    import verify_breakout as vb
    codes = available_codes(conn)
    print(f"[돌파] 분봉 {len(codes)} 종목", file=sys.stderr)
    daily = load_ohlcv(conn, vb.DAILY_LEAD, vb.END)
    daily["code"] = daily["code"].astype(str).str.strip()
    cur = conn.cursor()
    cur.execute("select code, date, close from daily_bars where date >= %s and date <= %s",
                (vb.START, vb.END))
    raw = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
    cur.execute("select code from stocks where name like %s", ("%스팩%",))
    spac = {str(r[0]).strip() for r in cur.fetchall()}
    cur.close()
    raw["code"] = raw["code"].astype(str).str.strip()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    w = vb.prepare_daily(daily, raw, spac)
    key = vb._key_frame(w)
    dates = sorted(w.loc[(w["date"] >= vb.START) & (w["date"] <= vb.END), "date"].unique())
    acc = {"date_idx": {d: i for i, d in enumerate(dates)},
           "sum": np.zeros((len(dates), vb.N_SLOTS)),
           "cnt": np.zeros((len(dates), vb.N_SLOTS), dtype=np.int64)}
    cnt = {k: 0 for k in ("univ_days_with_bars", "daily_main_with_bars", "drop_no0900",
                          "drop_no1530", "drop_scale", "gap_break", "daily_main_after_dayfilters",
                          "intraday_break_any_minute", "daily_main_no_minute_break",
                          "trig_0900", "trig_1500plus", "limit_at_entry")}
    print(f"[돌파] 유니버스 {len(key):,} · 일봉 주 이벤트 {int(key['ev_daily'].sum()):,}", file=sys.stderr)
    out = []
    BATCH = 40
    t0 = time.time()
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i + BATCH]
        bars = load_minute_bars(conn, vb.START, vb.END, codes=chunk)
        if bars.empty:
            continue
        ev = vb.events_from_batch(bars, key[key["code"].isin(chunk)], acc, cnt)
        if len(ev):
            evs = [{"code": r.code, "date": pd.Timestamp(r.date).normalize(), "entry": float(r.entry),
                    "entry_min": int(r.touch_minute) + OPEN_MIN,
                    "limit": limit_up_price(float(r.prev_close_raw)),
                    "existing_ret": float(r.ret_stop2), "include_entry_bar": False}
                   for r in ev.itertuples()]
            pidx = path_index(bars, {(e["code"], e["date"]) for e in evs})
            out.extend(measure(evs, pidx))
        print(f"  [돌파] 배치 {i // BATCH + 1}: 이벤트 {len(ev):,} (누적 {len(out):,}, {time.time() - t0:.0f}s)",
              file=sys.stderr)
        del bars
    return out


def gen_dip(conn) -> list[dict]:
    import itertools

    import verify_dip as vd
    codes = available_codes(conn)
    print(f"[낙주] 분봉 {len(codes)} 종목", file=sys.stderr)
    d_start = (pd.Timestamp("2025-08-01") - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily_all = load_ohlcv(conn, d_start, "2026-08-21")
    cal = list(pd.DatetimeIndex(daily_all["date"].unique()).sort_values())
    cal_prev = {d: p for p, d in itertools.pairwise(cal)}
    out = []
    BATCH = 40
    t0 = time.time()
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i + BATCH]
        bars = load_minute_bars(conn, "2025-08-01", "2026-08-21", codes=chunk)
        if bars.empty:
            continue
        ev, _ = vd.build_events(bars, daily_all[daily_all["code"].isin(chunk)], cal_prev)
        if len(ev):
            evs = [{"code": r.code, "date": pd.Timestamp(r.date).normalize(), "entry": float(r.entry),
                    "entry_min": int(r.trig_minute) + OPEN_MIN,
                    "limit": limit_up_price(float(r.prev_close)),
                    "existing_ret": float(r.ret), "include_entry_bar": False}
                   for r in ev.itertuples()]
            pidx = path_index(bars, {(e["code"], e["date"]) for e in evs})
            out.extend(measure(evs, pidx))
        print(f"  [낙주] 배치 {i // BATCH + 1}: 이벤트 {len(ev):,} (누적 {len(out):,}, {time.time() - t0:.0f}s)",
              file=sys.stderr)
        del bars
    return out


def gen_pair(conn) -> list[dict]:
    import verify_pair as vp
    data = vp.collect(conn, "2025-08-04", "2026-08-21")
    main = data["main"]
    print(f"[짝꿍] 주 규칙 거래 {len(main):,}건", file=sys.stderr)
    out = []
    # 후속주 분봉을 (종목, 일자)별로 재로드해 T+1..15:19 측정창을 만든다.
    for date, g in main.groupby("date"):
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        fols = sorted(g["fol"].astype(str).unique())
        bars = load_minute_bars(conn, ds, ds, codes=fols)
        pidx = path_index(bars) if not bars.empty else {}
        evs = [{"code": str(r.fol), "date": pd.Timestamp(r.date).normalize(), "entry": float(r.entry),
                "entry_min": int(r.T) + 1, "limit": float(r.fol_limit),
                "existing_ret": float(r.gross), "include_entry_bar": True}
               for r in g.itertuples()]
        out.extend(measure(evs, pidx))
    return out


GENERATORS = {"sangtta": gen_sangtta, "breakout": gen_breakout, "dip": gen_dip, "pair": gen_pair}
LABEL = {"sangtta": "상따", "breakout": "돌파", "dip": "낙주", "pair": "짝꿍"}


# ─────────────────────────────────────────────────────────── 보고·판정
def strat_stats(df: pd.DataFrame, col: str) -> dict:
    r = df[col].dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    m = df.dropna(subset=[col]).copy()
    m["ym"] = m["date"].dt.strftime("%Y-%m")
    mon = m.groupby("ym")[col].mean() - COST
    excl = df.loc[~df["limit_suspect"], col].dropna()
    top99 = r[r <= r.quantile(0.99)]
    return {"n": n, "mean": r.mean(), "net34": r.mean() - COST, "net55": r.mean() - COST_HI,
            "median": r.median(), "win": (r > 0).mean(), "neg_months": int((mon < 0).sum()),
            "n_months": len(mon), "excl_net34": (excl.mean() - COST) if len(excl) else np.nan,
            "top99_net34": (top99.mean() - COST) if len(top99) else np.nan}


def analyze(tech: str, df: pd.DataFrame, out: list[str]) -> None:
    P = out.append
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    P(f"\n# {LABEL[tech]} — 청산 재측정 (n={n:,})")
    valid = df["mfe"].notna()
    P(f"측정창 있는 진입 {int(valid.sum()):,} / 측정창 없음(진입 후 봉 없음) {int((~valid).sum()):,}")
    mfe = df.loc[valid, "mfe"]
    mae = df.loc[valid, "mae"]
    P(f"\n**MFE** 중앙 {mfe.median():+.2%} · p25 {mfe.quantile(.25):+.2%} · "
      f"p75 {mfe.quantile(.75):+.2%} · p90 {mfe.quantile(.90):+.2%} · 평균 {mfe.mean():+.2%}")
    P(f"**MAE** 중앙 {mae.median():+.2%} · p25 {mae.quantile(.25):+.2%} · p75 {mae.quantile(.75):+.2%}")
    P(f"장중 상한가 도달(상한가 의심) {int(df['limit_suspect'].sum()):,}건 "
      f"({df['limit_suspect'].mean():.1%}) · 15:19 종가 상한가 이상 {int(df['is_limit_close'].sum()):,}건 "
      f"({df['is_limit_close'].mean():.1%})")

    P("\n| 청산 | net34 | net55 | 중앙 | 승률 | 음수월 | 상한가의심제외 net34 | 상위1%제거 net34 |")
    P("|---|---|---|---|---|---|---|---|")
    best_net, best_name = -1e9, None
    passing = []
    for s in STRATS:
        st = strat_stats(df, s)
        if st.get("n", 0) == 0:
            P(f"| {s} | — | — | — | — | — | — | — |"); continue
        P(f"| {s} | {st['net34']:+.2%} | {st['net55']:+.2%} | {st['median']:+.2%} | {st['win']:.1%} | "
          f"{st['neg_months']}/{st['n_months']} | {st['excl_net34']:+.2%} | {st['top99_net34']:+.2%} |")
        if st["net34"] > best_net:
            best_net, best_name = st["net34"], s
        cond = (st["net34"] > 0 and st["neg_months"] <= NEG_MONTHS_MAX
                and st["excl_net34"] > 0 and st["top99_net34"] > 0)
        if cond:
            passing.append(s)

    med = float(mfe.median())
    any_pos = any(strat_stats(df, s).get("net34", -1) > 0 for s in STRATS)
    if med >= MFE_PROMOTE and passing:
        verdict = f"**승격** — MFE 중앙 {med:+.2%} ≥ +5% 이고 [{', '.join(passing)}] 이(가) 4조건 통과"
    elif med < MFE_REJECT or not any_pos:
        why = f"MFE 중앙 {med:+.2%} < +2%" if med < MFE_REJECT else "어떤 청산도 net34>0 못 만듦"
        verdict = f"**기각 강화** — {why}"
    elif med >= MFE_PROMOTE:
        verdict = (f"**회색 상단** — MFE 중앙 {med:+.2%} ≥ +5% 이나 승격 청산 조건 미달 "
                   f"(진입 신호는 있으나 실현 청산 없음)")
    else:
        verdict = f"**회색** — MFE 중앙 {med:+.2%} (2~5%): 약한 신호, 호가·틱 없이는 실현 불가"
    P(f"\n판정: {verdict}")
    P(f"최선 청산 = {best_name} net34 {best_net:+.2%}")

    # 상위 20건 — 눈으로 본다
    top = df.loc[valid].sort_values("mfe", ascending=False).head(20)
    P("\n## MFE 상위 20건 (진입·MFE·MAE·주요 청산 수익)")
    P("| code | date | 진입가 | 진입분 | MFE | MAE | trail5 | tgt8 | 3:3:4 | 기존 | 종가 | 상한가의심 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in top.itertuples():
        hh = f"{int(r.entry_min) // 60:02d}:{int(r.entry_min) % 60:02d}"
        P(f"| {r.code} | {pd.Timestamp(r.date).date()} | {r.entry:,.0f} | {hh} | {r.mfe:+.1%} | "
          f"{r.mae:+.1%} | {r.trail5:+.1%} | {r.tgt8:+.1%} | {r.split334:+.1%} | {r.existing:+.1%} | "
          f"{r.close:+.1%} | {'Y' if r.limit_suspect else ''} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="all", help="sangtta|breakout|dip|pair|all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default=None, help="이벤트 측정 결과 pickle 저장/재사용")
    ap.add_argument("--env-file", default="/home/young/Documents/git/kr-quant/.env")
    args = ap.parse_args()
    _load_env(args.env_file)

    techs = list(GENERATORS) if args.tech == "all" else args.tech.split(",")
    out: list[str] = ["# 청산 재측정 결과 — 사전등록 18번",
                      f"실행 {time.strftime('%Y-%m-%d %H:%M')} · 비용 34/55bp · 측정창 진입 다음봉~15:19"]
    conn = connect()
    for tech in techs:
        cache = None
        if args.cache_dir:
            cache = os.path.join(args.cache_dir, f"exit_{tech}.pkl")
        if cache and os.path.exists(cache):
            df = pd.read_pickle(cache)
            print(f"[{tech}] 캐시 로드 {len(df):,}", file=sys.stderr)
        else:
            rows = GENERATORS[tech](conn)
            df = pd.DataFrame(rows)
            if cache:
                df.to_pickle(cache)
        analyze(tech, df, out)
    conn.close()

    text = "\n".join(out)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
        print(f"→ {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
