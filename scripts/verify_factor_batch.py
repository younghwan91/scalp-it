"""다중팩터 배치 측정 — 사전등록 31번 그대로 1회.

PEAD 와 동일 프레임(폴드내 횡단면 데실·롱숏 Q10-Q1·연도 일관성·비용후 net)으로
6개 팩터를 각 1회 측정. 결과 → scratchpad factor_batch.md. DB 쓰기 없음.

팩터(Q10=롱 방향, 부호 사전고정):
 1 모멘텀 12-1  = ret(-252~-21)                 월리밸, 보유21
 2 단기 반전    = -ret(직전5거래일)              주리밸, 보유5
 3 저변동성     = -std(직전60거래일 일수익)       월리밸, 보유21
 4 사이즈       = -log(시총=원종가×상장주식수)     월리밸, 보유21
 5 거래량 쇼크  = 평균거래량(-5~0)/평균거래량(-25~-5)  주리밸, 보유5 (추종, 사전고정)
 6 수급 지속성  = Σ_{-5..-1}(외국인+기관)/거래량   주리밸, 보유5 (supply_demand 2024+만)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2

SCRATCH = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
           "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad")
OUT = f"{SCRATCH}/factor_batch.md"

RT = 0.0034            # 편도 왕복 34bp (한 다리)
BORROW_ANNUAL = 0.03  # 숏 차입 연 3%
LIQ_FLOORS = {"50억": 5000.0, "500억": 50000.0}
T_BONF = 2.6          # Bonferroni 6검정 |t| 기준
MIN_NAMES = 10        # 폴드당 최소 종목 (< 이면 폴드 제외)


def conn():
    return psycopg2.connect(os.environ["KR_QUANT_DB"])


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    return m, m / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def rollmean(x, w):
    """트레일링 w 평균(바 i 포함, i-w+1..i). 부족하면 nan."""
    x = np.asarray(x, float)
    cs = np.concatenate([[0.0], np.nancumsum(x)])
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


def _last_idx_per_group(keys, n):
    """각 그룹(월/주)의 마지막 바 인덱스."""
    return [i for i in range(n) if i == n - 1 or keys[i] != keys[i + 1]]


def _emit(ctx, i, period, H, sig_dict, rows):
    """리밸 바 i 에서 익일 시가 진입·+H 시가 청산 이벤트 1건 생성."""
    n, o, d = ctx["n"], ctx["o"], ctx["d"]
    rb, shb = ctx["rb"], ctx["shb"]
    if i + 1 >= n:
        return
    entry = o[i + 1]
    xi = i + 1 + H
    if xi >= n or not (entry > 0):
        return
    ret = o[xi] / entry - 1.0
    # 상한가(원주가): 진입 시가 / 전일 종가 >= 1.29
    lu = False
    if rb is not None:
        rd, ro, rc = rb
        rei = int(np.searchsorted(rd, d[i + 1], side="left"))
        if 0 <= rei < len(rd) and rd[rei] == d[i + 1] and rei > 0:
            pc = rc[rei - 1]
            if pc > 0 and ro[rei] / pc >= 1.29:
                lu = True
    # 시총(원종가 × 발행주식 as-of backward)
    mcap = np.nan
    if rb is not None and shb is not None:
        rd, ro, rc = rb
        rii = int(np.searchsorted(rd, d[i], side="right")) - 1
        sdt, sval = shb
        sj = int(np.searchsorted(sdt, d[i], side="right")) - 1
        if 0 <= rii < len(rc) and sj >= 0 and rc[rii] > 0 and sval[sj] > 0:
            mcap = rc[rii] * sval[sj]
    rows.append({"code": ctx["code"], "period": period, "date": d[i],
                 "liq20": ctx["liq20"][i], "entry_limit_up": lu, "ret": ret,
                 "mcap": mcap, **sig_dict})


def build_price_events():
    """일봉으로 월/주 리밸 이벤트 프레임 2개 생성. 팩터 1~5 신호·라벨·유동성·상한가 포함."""
    c = conn()
    adj = pd.read_sql(
        "select code, date, open, close, volume, trade_value "
        "from daily_bars_adjusted order by code, date",
        c, parse_dates=["date"])
    raw = pd.read_sql(
        "select code, date, open, close from daily_bars order by code, date",
        c, parse_dates=["date"])
    shares = pd.read_sql(
        "select code, date, shares_outstanding from shares_outstanding_history "
        "order by code, date", c, parse_dates=["date"])
    excl = pd.read_sql(
        "select code from stocks where name like '%%스팩%%' or name like "
        "'%%기업인수목적%%' or name like '%%리츠%%' "
        "union select code from delisted_stocks where name like '%%스팩%%' "
        "or name like '%%기업인수목적%%' or name like '%%리츠%%'", c)
    c.close()
    excl_set = set(excl["code"])

    raw_by = {}
    for code, g in raw.groupby("code", sort=False):
        raw_by[code] = (g["date"].values,
                        g["open"].values.astype(float),
                        g["close"].values.astype(float))
    sh_by = {}
    for code, g in shares.groupby("code", sort=False):
        sh_by[code] = (g["date"].values,
                       g["shares_outstanding"].values.astype(float))

    mon_rows, wk_rows = [], []
    for code, g in adj.groupby("code", sort=False):
        if code in excl_set:
            continue
        g = g.sort_values("date")
        d = g["date"].values
        o = g["open"].values.astype(float)
        cl = g["close"].values.astype(float)
        vol = g["volume"].values.astype(float)
        tv = g["trade_value"].values.astype(float)
        n = len(d)
        if n < 30:
            continue
        rets = np.concatenate([[np.nan], cl[1:] / cl[:-1] - 1.0])

        liq20 = rollmean(tv, 20)
        vr_num = rollmean(vol, 5)                      # -4..0
        vr_den5 = rollmean(vol, 25)                    # -24..0
        # 평균거래량(-25..-5) = (25*mean25 - 5*mean5)/20
        vr_den = (25.0 * vr_den5 - 5.0 * vr_num) / 20.0
        volratio = np.where((vr_den > 0), vr_num / vr_den, np.nan)
        vol60 = pd.Series(rets).rolling(60).std().values  # -59..0 일수익 std

        # 원주가/발행주식 lookup 준비
        rb = raw_by.get(code)
        shb = sh_by.get(code)

        dts = pd.DatetimeIndex(d)
        mon_key = dts.to_period("M").astype(str).values
        wk_key = dts.to_period("W").astype(str).values
        ctx = {"code": code, "d": d, "o": o, "cl": cl, "n": n,
               "liq20": liq20, "rb": rb, "shb": shb}

        for i in _last_idx_per_group(mon_key, n):
            if i < 252:
                continue
            mom = (cl[i - 21] / cl[i - 252] - 1.0) if cl[i - 252] > 0 else np.nan
            _emit(ctx, i, mon_key[i], 21,
                  {"f1_mom": mom,
                   "f3_lowvol": (-vol60[i] if np.isfinite(vol60[i]) else np.nan)},
                  mon_rows)
        for i in _last_idx_per_group(wk_key, n):
            if i < 30:
                continue
            rev = -(cl[i] / cl[i - 5] - 1.0) if cl[i - 5] > 0 else np.nan
            _emit(ctx, i, wk_key[i], 5,
                  {"f2_rev": rev, "f5_volshock": volratio[i]}, wk_rows)

    mon = pd.DataFrame(mon_rows)
    wk = pd.DataFrame(wk_rows)
    # 사이즈 신호 = -log(시총)
    mon["f4_size"] = -np.log(mon["mcap"].where(mon["mcap"] > 0))
    return mon, wk


def build_flow_events(wk):
    """팩터6: 주리밸 이벤트에 (외국인+기관)/거래량 5일합 병합. supply_demand 2024+만."""
    c = conn()
    sd = pd.read_sql(
        "select code, date, foreign_, institution, acc_trde_qty from supply_demand "
        "where date >= '2023-11-01' order by code, date",
        c, parse_dates=["date"])
    c.close()
    sd["fi"] = (sd["foreign_"].astype(float) + sd["institution"].astype(float))
    q = sd["acc_trde_qty"].astype(float)
    sd["intensity"] = np.where(q > 0, sd["fi"] / q, np.nan)
    # 종목별 5일 트레일링 합
    parts = []
    for code, g in sd.groupby("code", sort=False):
        g = g.sort_values("date").copy()
        g["f6_flow"] = g["intensity"].rolling(5).sum().values
        parts.append(g[["code", "date", "f6_flow"]])
    flow = pd.concat(parts, ignore_index=True)
    m = wk.merge(flow, on=["code", "date"], how="left")
    return m


def decile_within(df, col):
    def _q(s):
        s = s.dropna()
        if len(s) < MIN_NAMES:
            return pd.Series(np.nan, index=s.index)
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(float)
    return df.groupby("period", group_keys=False)[col].apply(_q)


def period_to_quarter(period):
    # 'YYYY-MM' 또는 'YYYY-MM-DD/...' (주) → 'YYYYQn'
    s = str(period)
    y = int(s[:4]); mo = int(s[5:7])
    return f"{y}Q{(mo - 1) // 3 + 1}"


def measure_factor(ev, sigcol, floor, H):
    """단일 팩터 × 유동성 하한 측정. 반환 dict."""
    d = ev[(ev["liq20"] >= floor) & (~ev["entry_limit_up"])
           & ev[sigcol].notna() & ev["ret"].notna()].copy()
    n_lu = int((ev[(ev["liq20"] >= floor) & ev["entry_limit_up"]
                   & ev[sigcol].notna() & ev["ret"].notna()]).shape[0])
    if d.empty:
        return None
    d["decile"] = decile_within(d, sigcol)
    d = d[d["decile"].notna()].copy()
    d["decile"] = d["decile"].astype(int)
    if d.empty:
        return None

    g = d.groupby("decile")["ret"]
    dec_mean = g.mean()
    spear = np.nan
    if len(dec_mean) >= 3:
        sp = dec_mean.sort_index()
        spear = np.corrcoef(sp.index.values, sp.rank().values)[0, 1]

    q10 = d[d.decile == 10].groupby("period")["ret"].mean()
    q1 = d[d.decile == 1].groupby("period")["ret"].mean()
    spread = (q10 - q1).dropna()
    m, t, npf = tstat(spread)
    borrow = BORROW_ANNUAL * (H / 252.0)
    net_ls = m - 2 * RT - borrow
    net_ls34 = m - RT - borrow  # 참고: 왕복 34bp

    # 연도별
    dd = d.copy()
    dd["year"] = pd.DatetimeIndex(dd["date"]).year
    yq10 = dd[dd.decile == 10].groupby("year")["ret"].mean()
    yq1 = dd[dd.decile == 1].groupby("year")["ret"].mean()
    yspread = (yq10 - yq1).dropna()
    neg_y = int((yspread < 0).sum())

    # 롱온리
    net_lo = q10.mean() - RT

    # 직교성용 분기 스프레드 시계열
    sp_q = spread.copy()
    qkey = pd.Index([period_to_quarter(p) for p in sp_q.index])
    qspread = pd.Series(sp_q.values, index=qkey).groupby(level=0).mean()

    return {
        "n_events": len(d), "n_periods": npf, "n_lu": n_lu,
        "spear": spear, "ls_mean": m, "ls_t": t,
        "net_ls": net_ls, "net_ls34": net_ls34,
        "neg_y": neg_y, "n_y": len(yspread),
        "q1": dec_mean.get(1, np.nan), "q10": dec_mean.get(10, np.nan),
        "lo_gross": q10.mean(), "lo_net": net_lo,
        "dec_mean": dec_mean, "yspread": yspread, "qspread": qspread,
    }


def pead_quarterly_spread():
    """PEAD 분기 롱숏 스프레드(50억, +40) 시계열 — 직교성 기준."""
    from pead_events import build_events, load_raw
    earn, bars, raw, spac = load_raw()
    ev = build_events(earn, bars, raw, spac)
    base = ev[ev["in_universe"] & ~ev["entry_limit_up"] & (ev["liq"] >= 5000)].copy()

    def _q(s):
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(int)
    base["decile"] = base.groupby("period", group_keys=False)["sue_raw"].apply(_q)
    d = base[base["ret40"].notna()]
    q10 = d[d.decile == 10].groupby("period")["ret40"].mean()
    q1 = d[d.decile == 1].groupby("period")["ret40"].mean()
    return (q10 - q1).dropna()  # index like '2020Q1'


def verdict(r):
    if r is None:
        return "N/A"
    c1 = (r["spear"] > 0) and (r["q10"] > r["q1"])
    c2 = (r["ls_mean"] > 0) and (abs(r["ls_t"]) > T_BONF)
    c3 = (r["n_y"] > 0) and (r["neg_y"] / r["n_y"] < 0.5)
    c4 = r["net_ls"] > 0
    if c1 and c2 and c3 and c4:
        return "채택"
    if c1 and r["ls_mean"] > 0:
        return "부분"
    return "기각"


FACTORS = [
    ("1 모멘텀12-1", "f1_mom", "mon", 21),
    ("2 단기반전", "f2_rev", "wk", 5),
    ("3 저변동성", "f3_lowvol", "mon", 21),
    ("4 사이즈", "f4_size", "mon", 21),
    ("5 거래량쇼크", "f5_volshock", "wk", 5),
    ("6 수급지속성", "f6_flow", "wk", 5),
]


def main():
    L = []
    def w(s=""):
        L.append(s); print(s)

    w("# 다중팩터 배치 측정 결과 (사전등록 31번, 1회)")
    w()
    w(f"- 비용: 롱숏 net = gross − 왕복 68bp − 숏차입(연{BORROW_ANNUAL*100:.0f}%×보유/252). "
      f"롱온리 net = gross − 34bp. 다중검정 Bonferroni |t|>{T_BONF}.")
    w(f"- 유니버스: {', '.join(LIQ_FLOORS)} (20일 평균 거래대금), 상폐포함·스팩/리츠 제외, 상한가 진입 제외.")
    w()

    print("[1/3] 가격 이벤트 구성 중 ...")
    mon, wk = build_price_events()
    print(f"  월 이벤트 {len(mon):,}행, 주 이벤트 {len(wk):,}행")
    print("[2/3] 수급(팩터6) 병합 중 ...")
    wk6 = build_flow_events(wk)
    wk6_2024 = wk6[pd.DatetimeIndex(wk6["date"]).year >= 2024].copy()
    print(f"  팩터6 2024+ 이벤트 {wk6_2024['f6_flow'].notna().sum():,}행")
    print("[3/3] PEAD 분기 스프레드 ...")
    pead_q = pead_quarterly_spread()
    print(f"  PEAD 분기 {len(pead_q)}개")

    results = {}
    for label, col, tbl, H in FACTORS:
        if col == "f6_flow":
            src = wk6_2024
        elif tbl == "mon":
            src = mon
        else:
            src = wk
        results[label] = {}
        for fname, floor in LIQ_FLOORS.items():
            results[label][fname] = measure_factor(src, col, floor, H)

    # 요약표 (50억 주판정)
    w("## 요약 — 50억 유니버스 (주 판정)")
    w()
    w("| 팩터 | N이벤트 | 폴드 | Q1% | Q10% | Spear | 롱숏gross% | t | net68% | 음수연도 | 판정 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, _c, _t, _H in FACTORS:
        r = results[label]["50억"]
        if r is None:
            w(f"| {label} | - | - | - | - | - | - | - | - | - | N/A |")
            continue
        v = verdict(r)
        w(f"| {label} | {r['n_events']:,} | {r['n_periods']} | {r['q1']*100:+.2f} | "
          f"{r['q10']*100:+.2f} | {r['spear']:+.2f} | {r['ls_mean']*100:+.2f} | "
          f"{r['ls_t']:+.2f} | {r['net_ls']*100:+.2f} | {r['neg_y']}/{r['n_y']} | **{v}** |")
    w()

    # 500억 생존
    w("## 500억 유니버스 (실전 실행가능성)")
    w()
    w("| 팩터 | N이벤트 | 폴드 | Spear | 롱숏gross% | t | net68% | 판정 |")
    w("|---|---|---|---|---|---|---|---|")
    for label, _c, _t, _H in FACTORS:
        r = results[label]["500억"]
        if r is None:
            w(f"| {label} | - | - | - | - | - | - | N/A |")
            continue
        v = verdict(r)
        w(f"| {label} | {r['n_events']:,} | {r['n_periods']} | {r['spear']:+.2f} | "
          f"{r['ls_mean']*100:+.2f} | {r['ls_t']:+.2f} | {r['net_ls']*100:+.2f} | **{v}** |")
    w()

    # 직교성 (PEAD 분기 상관) — 50억
    w("## PEAD 직교성 — 분기 롱숏 스프레드 Pearson 상관 (50억)")
    w()
    w("| 팩터 | 공통분기 | PEAD 상관 |")
    w("|---|---|---|")
    for label, _c, _t, _H in FACTORS:
        r = results[label]["50억"]
        if r is None:
            w(f"| {label} | - | - |")
            continue
        common = r["qspread"].index.intersection(pead_q.index)
        if len(common) >= 6:
            corr = np.corrcoef(r["qspread"].loc[common].values,
                               pead_q.loc[common].values)[0, 1]
            w(f"| {label} | {len(common)} | {corr:+.2f} |")
        else:
            w(f"| {label} | {len(common)} | (표본부족) |")
    w()

    # 데실 프로파일 (50억)
    w("## 데실 평균수익 프로파일 (50억, %)")
    w()
    for label, _c, _t, _H in FACTORS:
        r = results[label]["50억"]
        if r is None:
            continue
        dm = r["dec_mean"].reindex(range(1, 11))
        w(f"- **{label}**: " + " ".join(
            f"Q{k}:{dm[k]*100:+.1f}" if pd.notna(dm[k]) else f"Q{k}:-"
            for k in range(1, 11)))
    w()

    # 연도별 롱숏 (50억)
    w("## 연도별 롱숏 스프레드 (50억, %)")
    w()
    for label, _c, _t, _H in FACTORS:
        r = results[label]["50억"]
        if r is None:
            continue
        ys = r["yspread"]
        w(f"- **{label}**: " + ", ".join(f"{int(y)}:{v*100:+.1f}" for y, v in ys.items()))
    w()

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
