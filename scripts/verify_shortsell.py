"""공매도 팩터 측정 — 사전등록 26번 그대로 1회.

주가설 (ii): 공매도 비중(short_ratio=short_qty/volume) 높음 → 추종 숏(약세 지속).
시그널 sr5 = 형성일 T 직전(포함) 최근 5거래일 short_ratio 평균(관측 ≥3).
분기(주)내 횡단면 데실, 롱숏 D1-D10(저강도 롱 - 고강도 숏), 주간 리밸, 주보유 +5거래일, 비용+숏차입 후.
결과를 스크래치패드 shortsell.md 로 출력. DB 쓰기 없음. 읽기 전용.
DB: 환경변수 KR_QUANT_DB.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/shortsell.md")

START = "2024-05-09"          # short_selling 데이터 시작
END = "2026-08-21"
LIQ_FLOORS = (5000, 50000)    # 백만원: 50억(주), 500억(병행)
LIQ_WINDOW = 20
SR_WINDOW = 5                 # sr5: 최근 5거래일 short_ratio 평균
SR_MIN_OBS = 3               # 창 내 공매도 관측 최소
HORIZONS = (5, 20)           # +5 주보유(판정), +20 진단
PRIMARY = 5
RT = 0.0034                  # 왕복 34bp/leg
RT55 = 0.0055
BORROW_YR = 0.03             # 숏 차입 연 3%
BORROW_YR_HI = 0.10          # 민감도 연 10%


def _conn():
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        raise SystemExit("KR_QUANT_DB 환경변수 없음")
    return psycopg2.connect(dsn)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    return m, m / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def load():
    c = _conn()
    ss = pd.read_sql(
        f"select code, date, short_ratio, short_balance from short_selling "
        f"where date between '{START}' and '{END}' order by code, date",
        c, parse_dates=["date"])
    bars = pd.read_sql(
        f"select code, date, open, trade_value from daily_bars_adjusted "
        f"where date between '2024-01-01' and '{END}' order by code, date",
        c, parse_dates=["date"])
    raw = pd.read_sql(
        f"select code, date, open, close from daily_bars "
        f"where date between '2024-01-01' and '{END}' order by code, date",
        c, parse_dates=["date"])
    spac1 = pd.read_sql("select code from stocks where name like '%%스팩%%' "
                        "or name like '%%기업인수목적%%'", c)
    spac2 = pd.read_sql("select code from delisted_stocks where name like '%%스팩%%' "
                        "or name like '%%기업인수목적%%'", c)
    c.close()
    return ss, bars, raw, set(spac1.code) | set(spac2.code)


def build_events(ss, bars, raw, spac):
    # 주간 형성일 T = 각 ISO주 마지막 거래일 (전 종목 공통 거래달력)
    td = np.array(sorted(bars["date"].unique()))
    tdf = pd.Series(td)
    iso = tdf.dt.isocalendar()
    wk = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    week_last = tdf.groupby(wk.values).max().values  # 각 주 마지막 거래일
    # short_selling 시작 이후의 주만
    week_last = np.array([d for d in week_last if d >= np.datetime64(START)])

    # 종목별 배열
    adj = {}
    for code, g in bars.groupby("code", sort=False):
        g = g.sort_values("date")
        adj[code] = (g["date"].values, g["open"].values.astype(float),
                     g["trade_value"].values.astype(float))
    rawd = {}
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("date")
        rawd[code] = (g["date"].values, g["open"].values.astype(float),
                      g["close"].values.astype(float))
    srd = {}
    for code, g in ss.groupby("code", sort=False):
        g = g.sort_values("date")
        srd[code] = (g["date"].values, g["short_ratio"].values.astype(float))

    rows = []
    for T in week_last:
        Tn = np.datetime64(pd.Timestamp(T), "ns")
        for code, (sdates, sr) in srd.items():
            # sr5: date<=T 인 마지막 5 관측 평균, >=3
            si = int(np.searchsorted(sdates, Tn, side="right"))
            lo = max(0, si - SR_WINDOW)
            win = sr[lo:si]
            win = win[~np.isnan(win)]
            if len(win) < SR_MIN_OBS:
                continue
            sr5 = float(win.mean())
            if code not in adj:
                continue
            dates, opens, tvs = adj[code]
            ei = int(np.searchsorted(dates, Tn, side="right"))  # 진입 = T 초과 첫 거래일
            if ei >= len(dates):
                continue
            entry = opens[ei]
            if not entry > 0:
                continue
            # 유동성: 진입 직전 20거래일
            l0 = max(0, ei - LIQ_WINDOW)
            liq = float(np.nanmean(tvs[l0:ei])) if ei - l0 == LIQ_WINDOW else np.nan
            labels = {}
            for h in HORIZONS:
                xi = ei + h
                labels[h] = opens[xi] / entry - 1.0 if xi < len(dates) else np.nan
            # 진입 상한가 (원주가)
            entry_lu = False
            if code in rawd:
                rd, ro, rc = rawd[code]
                rei = int(np.searchsorted(rd, Tn, side="right"))
                if 0 < rei < len(rd) and rc[rei - 1] > 0 and ro[rei] / rc[rei - 1] >= 1.29:
                    entry_lu = True
            rows.append({"T": pd.Timestamp(T), "code": code, "sr5": sr5, "liq": liq,
                         "entry_lu": entry_lu,
                         **{f"ret{h}": labels[h] for h in HORIZONS}})
    ev = pd.DataFrame(rows)
    ev["spac"] = ev["code"].isin(spac)
    ev["year"] = ev["T"].dt.year
    return ev


def decile_within(df, col="sr5"):
    def _q(s):
        if s.nunique() < 10:
            return pd.Series([np.nan] * len(s), index=s.index)
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(float)
    return df.groupby("T")[col].transform(_q)


def measure(ev, floor, W, L):
    base = ev[(ev["liq"] >= floor) & (~ev["spac"]) & (~ev["entry_lu"])].copy()
    base["decile"] = decile_within(base)
    base = base[base["decile"].notna()]
    rc = f"ret{W}"
    d = base[base[rc].notna()].copy()
    d["decile"] = d["decile"].astype(int)
    L(f"### 유동성 하한 {floor}백만, 보유 +{W}거래일")
    g = d.groupby("decile")[rc]
    dec = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
    L("| 데실(강도) | 평균% | 중앙% | N |")
    L("|---|---|---|---|")
    for dd in range(1, 11):
        if dd in dec.index:
            L(f"| D{dd} | {dec.loc[dd,'mean']*100:+.3f} | {dec.loc[dd,'median']*100:+.3f} | {int(dec.loc[dd,'n'])} |")
    sp = dec["mean"].sort_index()
    spear = np.corrcoef(sp.index, sp.rank())[0, 1]
    # 롱숏 D1 - D10 (저강도 롱 - 고강도 숏)
    d1 = d[d.decile == 1].groupby("T")[rc].mean()
    d10 = d[d.decile == 10].groupby("T")[rc].mean()
    spread = (d1 - d10).dropna()
    m, t, n = tstat(spread)
    borrow = BORROW_YR * (W / 252)
    borrow_hi = BORROW_YR_HI * (W / 252)
    net = m - 2 * RT - borrow
    net_hi = m - 2 * RT - borrow_hi
    net55 = m - 2 * RT55 - borrow
    neg = int((spread < 0).sum())
    L(f"- 데실-수익 순위상관 Spearman = {spear:+.2f} (가설: <0); D1−D10 gross = "
      f"{(dec.loc[1,'mean']-dec.loc[10,'mean'])*100:+.3f}%")
    L(f"- 롱숏 D1−D10 주평균 {m*100:+.3f}%, t={t:+.2f}, N={n}주; "
      f"net(68bp+차입{borrow*100:.2f}%) = {net*100:+.3f}%; "
      f"net(차입10% {borrow_hi*100:.2f}%)={net_hi*100:+.3f}%; net55={net55*100:+.3f}%")
    L(f"- 음수 주: {neg}/{n} ({neg/n*100:.0f}%)")
    # 롱온리 D1 vs 유니버스
    uni = d.groupby("T")[rc].mean()
    excess = (d1 - uni).dropna()
    me, te, _ = tstat(excess)
    L(f"- 롱온리 D1: gross {d1.mean()*100:+.3f}%, net34 {(d1.mean()-RT)*100:+.3f}%; "
      f"vs 유니버스 초과 {me*100:+.3f}% (t={te:+.2f})")
    # 연도별
    d["year"] = d["T"].dt.year
    yd1 = d[d.decile == 1].groupby("year")[rc].mean()
    yd10 = d[d.decile == 10].groupby("year")[rc].mean()
    ysp = (yd1 - yd10).dropna()
    negy = int((ysp < -0.002).sum())
    L("- 연도별 D1−D10: " + ", ".join(f"{int(y)}:{v*100:+.3f}" for y, v in ysp.items())
      + f" (유의미 음수 {negy}년)")
    L("")
    return {"spear": spear, "m": m, "t": t, "n": n, "net": net, "neg": neg,
            "d1": dec.loc[1, "mean"], "d10": dec.loc[10, "mean"], "negy": negy,
            "ysp": ysp, "spread": spread}


def main():
    Lst = []
    def w(s=""):
        Lst.append(s); print(s)

    ss, bars, raw, spac = load()
    ev = build_events(ss, bars, raw, spac)

    w("# 공매도 팩터 측정 결과 (사전등록 26번, 1회)")
    w()
    w(f"- 기간 {START}~{END}, 주간 리밸, 시그널 sr5(최근5일 short_ratio 평균, ≥{SR_MIN_OBS}관측).")
    w(f"- 롱숏 D1−D10(저강도 롱−고강도 숏), 주보유 +{PRIMARY}거래일(판정), +20 진단.")
    w(f"- 총 이벤트(형성×종목): {len(ev):,}, 주 수: {ev['T'].nunique()}, 종목: {ev['code'].nunique()}.")
    w(f"- 진입 상한가(제외): {int(ev['entry_lu'].sum())}, 스팩: {int(ev['spac'].sum())}.")
    w()

    results = {}
    for floor in LIQ_FLOORS:
        w(f"## 유동성 하한 {floor}백만 ({'50억 주' if floor==5000 else '500억 병행'})")
        for W in HORIZONS:
            tag = " (판정)" if (W == PRIMARY and floor == 5000) else ""
            w(f"#### 보유 +{W}{tag}")
            r = measure(ev, floor, W, w)
            results[(floor, W)] = r

    # 판정 (주: floor=5000, W=5)
    r = results[(5000, PRIMARY)]
    r500 = results[(50000, PRIMARY)]
    c1 = r["spear"] < 0 and r["d1"] > r["d10"]
    c2 = r["m"] > 0 and abs(r["t"]) >= 2
    c3 = r["negy"] == 0 and (r["neg"] / r["n"] < 0.5)
    c4 = r["net"] > 0
    c5 = (r500["spear"] < 0 and r500["m"] > 0)  # 500억에서도 방향 유지
    w("## 판정 (사전등록 5절)")
    w(f"1. 데실 단조 Spearman<0 & D1>D10: {'OK' if c1 else 'FAIL'} (spear={r['spear']:+.2f})")
    w(f"2. 롱숏 양수 & |t|>=2: {'OK' if c2 else 'FAIL'} (m={r['m']*100:+.3f}%, t={r['t']:+.2f}, N={r['n']})")
    w(f"3. 연도 일관성(유의미 음수년 0 & 음수주<50%): {'OK' if c3 else 'FAIL'} "
      f"(음수년 {r['negy']}, 음수주 {r['neg']}/{r['n']})")
    w(f"4. net>0: {'OK' if c4 else 'FAIL'} (net={r['net']*100:+.3f}%)")
    w(f"5. 500억에서도 방향 유지: {'OK' if c5 else 'FAIL'} "
      f"(spear={r500['spear']:+.2f}, m={r500['m']*100:+.3f}%)")
    allok = c1 and c2 and c3 and c4 and c5
    verdict = "채택" if allok else ("부분" if (c1 and r["m"] > 0) else "기각")
    w(f"\n### 최종: **{verdict}**")

    with open(OUT, "w") as f:
        f.write("\n".join(Lst) + "\n")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
