#!/usr/bin/env python3
"""짝꿍 이벤트 수집 — **테마별 후속주**까지 뽑는다(만쥬 §2·§3).

만쥬가 우리 백테스트에 준 지적 중 가장 센 것이 이거였다:

> "27종 표준 산업분류로는 실전 테마를 절대 잡아낼 수 없습니다."

`stocks.sector` 는 '전기/전자' 같은 27종짜리다. 삼성전자와 초전도체 테마 잡주가
같은 칸에 들어간다. 만쥬가 보는 테마는 **같은 재료로 같이 뛰는 무리**다.

그걸 데이터로 만든다. 정의를 셋 두고 서로 겨루게 한다 — 어느 게 실전 테마에
가까운지는 우리가 정할 게 아니라 성적이 정할 일이다.

``theme``   잔차 동조. 최근 W 거래일 일간 수익률에서 **시장수익률(횡단 중앙값)을
            뺀 잔차**의 상관. 시장 베타를 걷어내야 "같이 오르는 것"과 "같은 재료로
            같이 오르는 것"이 갈린다. 기존 ``comove`` 가 진 이유가 이 자리다 —
            그건 당일 장중 분봉 상관이라 베타와 "둘 다 지금 뛰는 중"을 잡았다.
``cosurge`` 동반 급등 이력. 최근 W 거래일 중 **대장과 후보가 같이 +5% 이상** 오른
            날의 수. 사람이 "같은 테마"라 부를 때 실제로 근거로 삼는 것에 가장 가깝다.
``kiwoom``  키움 테마 스냅샷(`theme_members`). 검증용이다 — 스냅샷이 4일치뿐이고
            643종목만 덮어 백테스트 본선엔 못 쓴다. 우리가 만든 테마가 사람이 붙인
            테마와 얼마나 겹치는지 보는 자尺으로 둔다.

**미래를 안 본다.** 테마는 이벤트 당일을 **뺀** 직전 W 거래일로만 만든다.

    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/manju_events.py --out data/manju_events.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from limitup_timing import _load_env, limit_up_price

from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect

EPS = 1e-6
OPEN_M = 9 * 60
CLOSE_M = 15 * 60 + 30
LAST_TRIG = 15 * 60 + 5
HMAX = 30
LEAD_TV_MIN = 300e8
FOL_TV_MIN = 50e8
N_KEEP = 5              # 방식별 상위 후보 저장 수. 만쥬의 "2·3등주"를 보려면 3 초과여야 한다
THEME_W = 60            # 테마를 만드는 직전 거래일 수
SURGE = 0.05            # 동반 급등으로 셀 하루 등락률 문턱
COSURGE_MIN = 2         # 이 횟수 이상 같이 뛰어야 같은 테마로 본다
THEME_CORR_MIN = 0.25   # 잔차 상관이 이보다 낮으면 테마가 아니다


def q(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    return df


def load_daily_raw(conn, start, end):
    df = q(conn, "select code, date, open, high, low, close, trade_value from daily_bars "
                 "where date >= %s and date <= %s order by code, date", (start, end))
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "trade_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    return df


def daily_residuals(raw: pd.DataFrame) -> pd.DataFrame:
    """일간 수익률에서 시장수익률을 뺀 **잔차** 행렬(날짜 × 종목).

    시장수익률은 그날 전 종목 등락률의 **중앙값**이다. 평균이 아니라 중앙값인 건
    상한가 몇 개가 시장을 대표하게 두지 않으려는 것이다. 베타를 따로 추정하지 않는
    건 일부러다 — 60일 표본에서 종목별 베타는 잡음이 커서 빼면 오히려 더 지저분해진다.
    """
    piv = raw.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    ret = piv.pct_change(fill_method=None)
    mkt = ret.median(axis=1)
    return ret.sub(mkt, axis=0)


def kiwoom_themes(conn) -> dict[str, set[str]]:
    """키움 테마 스냅샷 → ``{종목: {테마코드}}``. 없으면 빈 dict(수집은 계속된다)."""
    try:
        df = q(conn, "select code, theme_code from theme_members")
    except Exception:  # noqa: BLE001 — 테마 테이블이 없어도 수집은 돌아야 한다
        return {}
    out: dict[str, set[str]] = {}
    for code, th in zip(df["code"], df["theme_code"], strict=False):
        out.setdefault(str(code), set()).add(str(th))
    return out


def path_arr(piv, code, m0, n):
    if code not in piv.columns:
        return np.full(n, np.nan)
    return piv[code].reindex(range(m0, m0 + n)).to_numpy(dtype=float)


def theme_picks(lead: str, cand: pd.Index, resid: pd.DataFrame,
                ret_hist: pd.DataFrame) -> tuple[dict[str, list[str]], pd.Series, pd.Series]:
    """직전 W 거래일로 만든 테마 후보. ``resid``·``ret_hist`` 는 **당일을 뺀** 창이다."""
    if lead not in resid.columns or len(resid) < 20:
        empty = pd.Series(np.nan, index=cand)
        return {"theme": [], "cosurge": []}, empty, empty
    cols = [c for c in cand if c in resid.columns]
    tcorr = pd.Series(np.nan, index=cand)
    if cols:
        tcorr.loc[cols] = resid[cols].corrwith(resid[lead])
    # 동반 급등 이력 — 같은 날 둘 다 +5% 이상 오른 횟수.
    cos = pd.Series(0.0, index=cand)
    if lead in ret_hist.columns and cols:
        lead_up = ret_hist[lead] >= SURGE
        if lead_up.any():
            cos.loc[cols] = (ret_hist.loc[lead_up, cols] >= SURGE).sum(axis=0).astype(float)
    return (
        {
            "theme": list(tcorr[tcorr >= THEME_CORR_MIN]
                          .sort_values(ascending=False).index),
            "cosurge": list(cos[cos >= COSURGE_MIN].sort_values(ascending=False).index),
        },
        tcorr, cos,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="테마 짝꿍 이벤트 수집")
    ap.add_argument("--start", default="2025-08-04")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _load_env()
    conn = connect(os.environ.get("KR_QUANT_DB"))
    # 테마 창(THEME_W 거래일)을 채우려면 달력일로 넉넉히 앞을 더 읽어야 한다.
    warm = (pd.Timestamp(a.start) - timedelta(days=int(THEME_W * 1.8) + 20)).strftime("%Y-%m-%d")
    print("일봉 로드…", file=sys.stderr)
    raw = load_daily_raw(conn, warm, a.end)
    resid_all = daily_residuals(raw)
    ret_all = raw.pivot_table(index="date", columns="code", values="close",
                              aggfunc="last").pct_change(fill_method=None)
    kiwoom = kiwoom_themes(conn)
    print(f"잔차행렬 {resid_all.shape} · 키움테마 종목 {len(kiwoom)}", file=sys.stderr)

    stocks = q(conn, "select code, name, market, sector from stocks")
    sector_of = dict(zip(stocks["code"], stocks["sector"], strict=False))
    name_of = dict(zip(stocks["code"], stocks["name"], strict=False))
    market_of = dict(zip(stocks["code"], stocks["market"], strict=False))

    days = sorted(raw.loc[(raw["date"] >= a.start) & (raw["date"] <= a.end), "date"].unique())
    print(f"거래일 {len(days)}", file=sys.stderr)

    ev_rows, paths = [], {}
    rng = np.random.default_rng(20260826)

    for di, d in enumerate(days):
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        raw_day = raw[raw["date"] == d]
        bars = load_minute_bars(conn, ds, ds)
        if bars.empty:
            continue
        # **당일을 뺀** 직전 W 거래일 — 여기서 하루라도 새면 미래를 보는 것이다.
        hist = resid_all.loc[resid_all.index < d].tail(THEME_W)
        rhist = ret_all.loc[ret_all.index < d].tail(THEME_W)

        bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
        bars = bars[bars["minute"] <= CLOSE_M]
        prev = raw_day.set_index("code")["prev_close"]
        rawc = raw_day.set_index("code")["close"]
        last = bars.groupby("code")["close"].last()
        ok = {c: bool(np.isfinite(rawc.get(c, np.nan)) and rawc.get(c, 0) > 0
                      and abs(v / rawc[c] - 1) <= 0.002) for c, v in last.items()}
        bars = bars[bars["code"].map(ok).fillna(False).astype(bool)]
        if bars.empty:
            continue
        bars["cum_tv"] = bars.groupby("code")["trade_value"].cumsum()
        lim = {c: limit_up_price(p) for c, p in prev.items() if np.isfinite(p) and p > 0}
        bars = bars[bars["code"].isin(lim)]
        if bars.empty:
            continue
        cw = bars.pivot(index="minute", columns="code", values="close")
        ow = bars.pivot(index="minute", columns="code", values="open")
        hw = bars.pivot(index="minute", columns="code", values="high")
        lw = bars.pivot(index="minute", columns="code", values="low")
        tw = bars.pivot(index="minute", columns="code", values="cum_tv").ffill()
        cwf = cw.ffill()
        limit_s = pd.Series({c: lim[c] for c in cw.columns})
        prev_s = prev.reindex(cw.columns)
        ret_w = cwf.div(prev_s, axis=1) - 1

        touched = (hw >= limit_s - EPS)
        at_lim = (cw - limit_s).abs() < EPS
        for code in cw.columns:
            tcol = touched[code]
            if not tcol.any():
                continue
            t_touch = int(tcol.idxmax())
            acol = at_lim[code]
            t_lock = int(acol.idxmax()) if acol.any() else None
            last_c = float(cw[code].dropna().iloc[-1]) if cw[code].notna().any() else np.nan
            held = bool(np.isfinite(last_c) and abs(last_c - limit_s[code]) < EPS)
            brk = None
            if t_lock is not None:
                sub = cw[code].dropna()
                sub = sub[sub.index > t_lock]
                bad = sub[sub < limit_s[code] - EPS]
                brk = int(bad.index[0]) if len(bad) else None
            if t_lock is None or t_lock > LAST_TRIG:
                continue
            cum_tv_lock = float(tw[code].reindex([t_lock]).iloc[0]) if t_lock in tw.index else np.nan
            if not np.isfinite(cum_tv_lock) or cum_tv_lock < LEAD_TV_MIN:
                continue
            T = t_lock
            if (T + 1) not in ow.index:
                continue
            o1 = ow.loc[T + 1]
            has1 = o1.notna() & cw.loc[T].notna() if T in cw.index else o1.notna()
            gate = (has1 & (tw.loc[T].reindex(o1.index) >= FOL_TV_MIN)
                    & ((o1 - limit_s).abs() >= EPS) & (o1 < limit_s - EPS))
            gate[code] = False
            cand = gate[gate].index
            if len(cand) == 0:
                continue
            r_T = ret_w.loc[T].reindex(cand)
            sec = pd.Series([sector_of.get(c) for c in cand], index=cand)
            lead_sec = sector_of.get(code)
            df = pd.DataFrame({"ret": r_T, "sector": sec})
            df["same_sector"] = df["sector"] == lead_sec if lead_sec else False
            df = df.sort_values("ret", ascending=False)
            df["ret_rank"] = np.arange(1, len(df) + 1)

            tp, tcorr, cos = theme_picks(code, cand, hist, rhist)
            df["tcorr"], df["cosurge"] = tcorr.reindex(df.index), cos.reindex(df.index)

            picks: dict[str, list[str]] = {}
            picks["sector"] = list(df[df["same_sector"]].head(N_KEEP).index)
            # 테마 후보는 상관/이력 순으로 걸러 두되, **저장은 등락률 순**으로 한다 —
            # 만쥬가 고르는 건 테마 안의 등락률 2·3등이지 상관 2·3등이 아니다.
            for k in ("theme", "cosurge"):
                sel = [c for c in df.index if c in set(tp[k][:N_KEEP * 4])]
                picks[k] = sel[:N_KEEP]
            lead_th = kiwoom.get(code, set())
            picks["kiwoom"] = ([c for c in df.index if kiwoom.get(c, set()) & lead_th][:N_KEEP]
                               if lead_th else [])
            picks["topret"] = list(df.head(N_KEEP).index)
            taken = set().union(*picks.values()) if picks else set()
            pool = [c for c in cand if c not in taken]
            picks["random"] = (list(rng.choice(pool, size=min(N_KEEP, len(pool)),
                                               replace=False)) if pool else [])

            keep = sorted(set().union(*picks.values()))
            if not keep:
                continue
            key = (pd.Timestamp(d), code, T)
            ev_rows.append({
                "date": d, "lead": code, "lead_name": name_of.get(code, code), "T": T,
                "t_touch": t_touch, "lead_limit": float(limit_s[code]),
                "lead_cum_tv": cum_tv_lock, "lead_sector": lead_sec,
                "lead_market": market_of.get(code), "lead_held": held, "lead_brk": brk,
                "n_cand": len(cand), "picks": picks,
                "fol_info": {c: {
                    "ret": float(df.at[c, "ret"]),
                    "ret_rank": int(df.at[c, "ret_rank"]),
                    "same_sector": bool(df.at[c, "same_sector"]),
                    "tcorr": (float(df.at[c, "tcorr"])
                              if np.isfinite(df.at[c, "tcorr"]) else np.nan),
                    "cosurge": float(df.at[c, "cosurge"]),
                    "same_kiwoom": bool(kiwoom.get(c, set()) & lead_th),
                    "name": name_of.get(c, c), "sector": sector_of.get(c),
                    "cum_tv": float(tw.at[T, c]), "prev_close": float(prev_s[c]),
                    "limit": float(limit_s[c])} for c in keep},
                "key": key})
            p = {"lead_close": path_arr(cw, code, T, HMAX + 2),
                 "lead_close_ff": path_arr(cwf, code, T, HMAX + 2)}
            for c in keep:
                p[c] = np.stack([path_arr(ow, c, T, HMAX + 2), path_arr(hw, c, T, HMAX + 2),
                                 path_arr(lw, c, T, HMAX + 2), path_arr(cw, c, T, HMAX + 2)])
            paths[key] = p
        if di % 10 == 0:
            print(f"  {ds}  이벤트 {len(ev_rows)}", file=sys.stderr)
    conn.close()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"events": pd.DataFrame(ev_rows), "paths": paths, "HMAX": HMAX,
                     "start": a.start, "end": a.end, "theme_w": THEME_W}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"저장 {out}: 이벤트 {len(ev_rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
