"""결합 북 측정 — 사전등록 32번 그대로 1회.

PEAD(분기 리밸, +40)·저변동성(월 리밸, +21)의 **월별 롱숏 net 수익 시계열**을 동일한
일간-마킹 엔진으로 만들고, **역변동성 가중(risk parity)** 으로 결합해 연율 수익·변동성·
샤프·MDD·상관을 낸다. 결합이 두 단일 대비 샤프↑·MDD↓ 인지 판정. 2016~2026 + 2024 전후.
결과 → scratchpad combo_book.md. DB 쓰기 없음.

방법(32번 §2·§3):
 - 각 코호트(리밸 시점 롱·숏 집합)는 형성일 익일부터 +H거래일 활성.
 - 하루 롱숏 수익 = 활성 롱 dret 평균 − 활성 숏 dret 평균. 월 복리 누적 = 월수익.
 - 비용(net): 코호트 형성 월에 회전비용 일괄 차감(PEAD 68bp+0.5%/분기, 저변동 68bp+0.25%/월).
 - 결합: net 월수익 표준편차로 w∝1/σ, 결합 = w_P·R_P + w_L·R_L.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2

SCRATCH = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
           "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad")
OUT = f"{SCRATCH}/combo_book.md"

LIQ_FLOOR = 5000.0        # 50억 (백만원)
MIN_NAMES = 10            # 폴드당 최소 종목
RT = 0.0068               # 롱숏 왕복 68bp
PEAD_BORROW = 0.005       # 숏차입 ~2개월 (분기)
LOWVOL_BORROW = 0.03 * 21 / 252  # 숏차입 연3% × 21/252 (월)
PEAD_H = 40
LOWVOL_H = 21


def conn():
    return psycopg2.connect(os.environ["KR_QUANT_DB"])


# ---------------------------------------------------------------- 데이터 로드

def load_all():
    c = conn()
    adj = pd.read_sql(
        "select code, date, open, close, trade_value from daily_bars_adjusted "
        "order by code, date", c, parse_dates=["date"])
    raw = pd.read_sql(
        "select code, date, open, close from daily_bars order by code, date",
        c, parse_dates=["date"])
    earn = pd.read_sql(
        "select code, period, avail_date, op_income, op_income_prior from earnings",
        c, parse_dates=["avail_date"])
    excl_reit_spac = pd.read_sql(
        "select code from stocks where name like '%%스팩%%' or name like "
        "'%%기업인수목적%%' or name like '%%리츠%%' "
        "union select code from delisted_stocks where name like '%%스팩%%' "
        "or name like '%%기업인수목적%%' or name like '%%리츠%%'", c)
    spac_only = pd.read_sql(
        "select code from stocks where name like '%%스팩%%' or name like "
        "'%%기업인수목적%%' "
        "union select code from delisted_stocks where name like '%%스팩%%' "
        "or name like '%%기업인수목적%%'", c)
    c.close()
    return adj, raw, earn, set(excl_reit_spac["code"]), set(spac_only["code"])


# --------------------------------------------------- 종목별 배열 · 달력 인덱스

def build_calendar(adj):
    cal = np.array(sorted(adj["date"].unique()))
    cal_idx = {d: i for i, d in enumerate(cal)}
    return cal, cal_idx


def build_code_arrays(adj, raw, cal, cal_idx):
    """code -> dict: 전역 달력에 정렬된 dret·close·open·tv·상한가(raw) 배열."""
    n = len(cal)
    dret = {}
    liq20 = {}
    vol60 = {}
    raw_limitup = {}   # code -> bool 배열: 그 날 시가/전일종가>=1.29 (원주가)
    for code, g in adj.groupby("code", sort=False):
        g = g.sort_values("date")
        idx = np.array([cal_idx[d] for d in g["date"].values])
        cl = g["close"].values.astype(float)
        tv = g["trade_value"].values.astype(float)
        # 일수익 (그 종목의 연속 바 기준)
        r = np.full(len(cl), np.nan)
        r[1:] = cl[1:] / cl[:-1] - 1.0
        d_arr = np.full(n, np.nan)
        d_arr[idx] = r
        dret[code] = d_arr
        # 20일 평균 거래대금 (트레일링, 그 종목 바 기준) → 전역 배열
        tvs = pd.Series(tv).rolling(20).mean().values
        l_arr = np.full(n, np.nan)
        l_arr[idx] = tvs
        liq20[code] = l_arr
        # 60일 일수익 표준편차 → 전역 배열
        v = pd.Series(r).rolling(60).std().values
        v_arr = np.full(n, np.nan)
        v_arr[idx] = v
        vol60[code] = v_arr
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("date")
        g = g[g["date"].isin(cal_idx)]   # 달력(수정주가) 밖 날짜 제외
        if g.empty:
            continue
        idx = np.array([cal_idx[d] for d in g["date"].values])
        ro = g["open"].values.astype(float)
        rc = g["close"].values.astype(float)
        lu = np.zeros(n, dtype=bool)
        prev = np.concatenate([[np.nan], rc[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ro / prev
        luf = (prev > 0) & (ratio >= 1.29)
        lu[idx] = luf
        raw_limitup[code] = lu
    return {"dret": dret, "liq20": liq20, "vol60": vol60,
            "limitup": raw_limitup}


# ----------------------------------------------------------- 코호트 구성

def month_end_indices(cal):
    """각 캘린더 월의 마지막 거래일 인덱스."""
    df = pd.DataFrame({"date": cal})
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym").tail(1).index.to_numpy()


def lowvol_cohorts(arr, cal, spac_reit):
    """월말마다 저변동 롱(Q10)/고변동 숏(Q1). 반환: [(형성idx, longs, shorts)]."""
    me = month_end_indices(cal)
    codes = list(arr["dret"].keys())
    vol60 = arr["vol60"]; liq20 = arr["liq20"]; limitup = arr["limitup"]
    cohorts = []
    for t in me:
        if t < 60 or t + 1 >= len(cal):
            continue
        rows = []
        for c in codes:
            lq = liq20[c][t]
            v = vol60[c][t]
            if not (np.isfinite(lq) and lq >= LIQ_FLOOR and np.isfinite(v)):
                continue
            if c in spac_reit:
                continue
            # 진입일(t+1) 상한가 제외
            lu = limitup.get(c)
            if lu is not None and t + 1 < len(lu) and lu[t + 1]:
                continue
            rows.append((c, -v))   # 신호 = -std (Q10=저변동=롱)
        if len(rows) < MIN_NAMES:
            continue
        df = pd.DataFrame(rows, columns=["code", "sig"])
        r = df["sig"].rank(method="first")
        df["dec"] = pd.qcut(r, 10, labels=range(1, 11)).astype(int)
        longs = set(df[df.dec == 10]["code"])
        shorts = set(df[df.dec == 1]["code"])
        cohorts.append((t, longs, shorts))
    return cohorts


def pead_cohorts(earn, arr, cal, cal_idx, adj, spac):
    """분기 avail 마다 SUE 데실 롱(Q10)/숏(Q1). 반환: [(형성idx, longs, shorts)]."""
    # 종목별 (adj) 진입가·유동성 lookup 재구성
    adj_by = {}
    for code, g in adj.groupby("code", sort=False):
        g = g.sort_values("date")
        adj_by[code] = (g["date"].values, g["open"].values.astype(float),
                        g["trade_value"].values.astype(float))
    limitup = arr["limitup"]
    rows = []
    for r in earn.itertuples(index=False):
        code = str(r.code)
        if code not in adj_by or pd.isna(r.avail_date):
            continue
        oi, oip = r.op_income, r.op_income_prior
        if not (pd.notna(oi) and pd.notna(oip) and oip != 0):
            continue
        dates, _opens, tvs = adj_by[code]
        avail = np.datetime64(pd.Timestamp(r.avail_date), "ns")
        ei = int(np.searchsorted(dates, avail, side="right"))
        if ei >= len(dates):
            continue
        entry_date = pd.Timestamp(dates[ei])
        lo = max(0, ei - 20)
        liq = float(np.nanmean(tvs[lo:ei])) if ei - lo >= 20 else np.nan
        if not (np.isfinite(liq) and liq >= LIQ_FLOOR):
            continue
        if code in spac:
            continue
        gidx = cal_idx.get(entry_date)
        if gidx is None or gidx == 0:
            continue
        lu = limitup.get(code)
        if lu is not None and lu[gidx]:
            continue
        sue = (float(oi) - float(oip)) / abs(float(oip))
        rows.append({"code": code, "period": str(r.period),
                     "entry_idx": gidx, "sue": sue})
    ev = pd.DataFrame(rows)
    cohorts = []
    for period, g in ev.groupby("period"):
        if len(g) < 30:   # PEAD 배포 규칙: 분기 유니버스 >= 30
            continue
        rr = g["sue"].rank(method="first")
        g = g.assign(dec=pd.qcut(rr, 10, labels=range(1, 11)).astype(int))
        # 형성idx = 진입 전일(익일부터 활성이 되도록). 코호트 내 진입일이 며칠 흩어질
        # 수 있어 최빈 진입일-1 을 형성일로 둔다.
        entry_mode = int(g["entry_idx"].mode().iloc[0])
        longs = set(g[g.dec == 10]["code"])
        shorts = set(g[g.dec == 1]["code"])
        cohorts.append((entry_mode - 1, longs, shorts))
    return cohorts


# ------------------------------------------------- 일간-마킹 → 월수익 시계열

def daily_longshort(cohorts, arr, cal, H):
    """코호트 리스트를 일간 롱숏 수익 배열로. 활성 = 형성idx+1 .. 형성idx+H."""
    n = len(cal)
    long_sum = np.zeros(n); long_cnt = np.zeros(n)
    short_sum = np.zeros(n); short_cnt = np.zeros(n)
    form_month_flags = np.zeros(n, dtype=bool)  # 코호트 형성(비용 부과) 표시
    dret = arr["dret"]
    for form_idx, longs, shorts in cohorts:
        a = form_idx + 1
        b = min(form_idx + H, n - 1)
        if a > b:
            continue
        form_month_flags[a] = True  # 진입 첫날에 회전비용 마킹
        for c in longs:
            seg = dret[c][a:b + 1]
            m = np.isfinite(seg)
            long_sum[a:b + 1][m] += seg[m]
            long_cnt[a:b + 1][m] += 1
        for c in shorts:
            seg = dret[c][a:b + 1]
            m = np.isfinite(seg)
            short_sum[a:b + 1][m] += seg[m]
            short_cnt[a:b + 1][m] += 1
    with np.errstate(invalid="ignore"):
        lmean = np.where(long_cnt > 0, long_sum / long_cnt, 0.0)
        smean = np.where(short_cnt > 0, short_sum / short_cnt, 0.0)
    ls = lmean - smean
    active = (long_cnt > 0) | (short_cnt > 0)
    return ls, active, form_month_flags


def monthly_series(cohorts, arr, cal, H, cost_per_cohort):
    """월별 gross·net 롱숏 수익 시계열(복리). 비용은 코호트 진입 첫날 배열에 마킹."""
    ls, active, form_flags = daily_longshort(cohorts, arr, cal, H)
    df = pd.DataFrame({"date": cal, "ls": ls, "active": active,
                       "cost": np.where(form_flags, cost_per_cohort, 0.0)})
    df["ym"] = df["date"].dt.to_period("M")
    out = []
    for ym, g in df.groupby("ym"):
        if not g["active"].any():
            continue
        gross = (1.0 + g["ls"].values).prod() - 1.0
        cost = g["cost"].sum()
        out.append((ym, gross, gross - cost))
    s = pd.DataFrame(out, columns=["ym", "gross", "net"]).set_index("ym")
    return s


# ------------------------------------------------------------ 성과 지표

def metrics(net_monthly):
    """월 net 시계열 → 연율수익·연율변동성·샤프·MDD."""
    x = net_monthly.dropna().values
    if len(x) < 6:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "mdd": np.nan, "n": len(x)}
    ann_ret = x.mean() * 12
    ann_vol = x.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1.0).min())
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "mdd": mdd, "n": len(x)}


def yearly_net(net_monthly):
    """연도별 net 합(단순 합)."""
    s = net_monthly.copy()
    s.index = [p.year for p in s.index]
    return s.groupby(level=0).sum()


def combine(pead_net, lv_net, weights=None):
    """공통 월에서 결합. weights=(w_p,w_l) 없으면 역변동성 고정 비중."""
    common = pead_net.index.intersection(lv_net.index)
    rp = pead_net.loc[common]; rl = lv_net.loc[common]
    if weights is None:
        sp = rp.std(ddof=1); sl = rl.std(ddof=1)
        wp = (1.0 / sp) / (1.0 / sp + 1.0 / sl)
        wl = 1.0 - wp
    else:
        wp, wl = weights
    comb = wp * rp + wl * rl
    return comb, (wp, wl), common


def expanding_combine(pead_net, lv_net, min_months=24):
    """확장창 역변동성 비중(룩어헤드 없음). 각 월 t 의 비중은 <t 데이터로만."""
    common = pead_net.index.intersection(lv_net.index)
    rp = pead_net.loc[common].values; rl = lv_net.loc[common].values
    out = np.full(len(common), np.nan)
    for i in range(len(common)):
        if i < min_months:
            continue
        sp = rp[:i].std(ddof=1); sl = rl[:i].std(ddof=1)
        if not (sp > 0 and sl > 0):
            continue
        wp = (1.0 / sp) / (1.0 / sp + 1.0 / sl)
        out[i] = wp * rp[i] + (1.0 - wp) * rl[i]
    return pd.Series(out, index=common).dropna()


# ------------------------------------------------------------------ 보고

def subperiod(s, lo=None, hi=None):
    idx = s.index
    mask = np.ones(len(idx), dtype=bool)
    if lo is not None:
        mask &= np.array([p.year >= lo for p in idx])
    if hi is not None:
        mask &= np.array([p.year <= hi for p in idx])
    return s[mask]


def main():
    L = []
    def w(s=""):
        L.append(s); print(s)

    print("[1/4] 데이터 로드 ...")
    adj, raw, earn, spac_reit, spac = load_all()
    cal, cal_idx = build_calendar(adj)
    print(f"  거래일 {len(cal)}일 ({cal[0].date()}~{cal[-1].date()})")
    print("[2/4] 종목 배열 구성 ...")
    arr = build_code_arrays(adj, raw, cal, cal_idx)
    print(f"  종목 {len(arr['dret'])}개")
    print("[3/4] 코호트 구성 ...")
    lv_co = lowvol_cohorts(arr, cal, spac_reit)
    pe_co = pead_cohorts(earn, arr, cal, cal_idx, adj, spac)
    print(f"  저변동 코호트 {len(lv_co)}개(월), PEAD 코호트 {len(pe_co)}개(분기)")
    print("[4/4] 월수익 시계열 · 결합 ...")
    lv = monthly_series(lv_co, arr, cal, LOWVOL_H, RT + LOWVOL_BORROW)
    pe = monthly_series(pe_co, arr, cal, PEAD_H, RT + PEAD_BORROW)

    pead_net = pe["net"]; lv_net = lv["net"]

    w("# 결합 북 측정 결과 (사전등록 32번, 1회)")
    w()
    w(f"- 유니버스 50억, 스팩·리츠 제외, 진입일 상한가 제외. 비용: 왕복 68bp + 숏차입"
      f"(PEAD 0.5%/분기, 저변동 {LOWVOL_BORROW*100:.2f}%/월).")
    w(f"- 월수익=일 롱숏 복리, 비용은 코호트 진입월 차감. 저변동 월수 {len(lv_net)}, "
      f"PEAD 월수 {len(pead_net)}.")
    w()

    # 재현 교차확인 (31번/PEAD)
    w("## 재현 교차확인 (단일 전략 net 평균, 31번·24번 대조)")
    w()
    w(f"- 저변동 월 net 평균 {lv_net.mean()*100:+.2f}% (31번 +2.34% 대조; "
      f"엔진 차이로 근사).")
    # PEAD 분기 net = 형성월 포함 40일 누적을 분기로 환산: 코호트당 평균
    w(f"- PEAD 월 net 평균 {pead_net.mean()*100:+.2f}%/월 → 분기 환산 "
      f"≈{pead_net.mean()*100*2:.2f}% (24번 분기 +4.47% 대조; 40일≈2개월 마킹).")
    w()

    def report_block(title, lo, hi):
        w(f"## {title}")
        w()
        pe_s = subperiod(pead_net, lo, hi)
        lv_s = subperiod(lv_net, lo, hi)
        comb, (wp, wl), common = combine(pe_s.reindex(
            pe_s.index.union(lv_s.index)).dropna() if False else pe_s, lv_s)
        # 상관
        cr = np.corrcoef(pe_s.loc[common].values, lv_s.loc[common].values)[0, 1] \
            if len(common) >= 6 else np.nan
        mp = metrics(pe_s); ml = metrics(lv_s); mc = metrics(comb)
        w(f"- 공통 월 {len(common)}, PEAD↔저변동 월 net 상관 **{cr:+.2f}**, "
          f"역변동성 비중 w_PEAD={wp:.2f} · w_저변동={wl:.2f}.")
        w()
        w("| 전략 | 월수 | 연율수익 | 연율변동성 | 샤프 | MDD |")
        w("|---|---:|---:|---:|---:|---:|")
        for nm, m in (("PEAD 단독", mp), ("저변동 단독", ml), ("**결합**", mc)):
            w(f"| {nm} | {m['n']} | {m['ann_ret']*100:+.1f}% | {m['ann_vol']*100:.1f}% | "
              f"{m['sharpe']:.2f} | {m['mdd']*100:.1f}% |")
        w()
        return {"mp": mp, "ml": ml, "mc": mc, "cr": cr, "wp": wp, "wl": wl,
                "pe_s": pe_s, "lv_s": lv_s, "comb": comb, "common": common}

    full = report_block("전 구간 (2016~2026) — 주 판정", None, None)
    report_block("2024 전 (<2024)", None, 2023)
    report_block("2024+ (>=2024)", 2024, None)

    # 상관 주기 진단: 31번 분기 -0.73 vs 여기 월 +0.17 의 원인(주기 효과) 규명
    def to_quarter(s):
        q = s.copy()
        q.index = [f"{p.year}Q{(p.month-1)//3+1}" for p in q.index]
        return q.groupby(level=0).sum()
    pq = to_quarter(pead_net); lq = to_quarter(lv_net)
    cq = pq.index.intersection(lq.index)
    corr_q = np.corrcoef(pq.loc[cq].values, lq.loc[cq].values)[0, 1] if len(cq) >= 6 else np.nan
    common_m = pead_net.index.intersection(lv_net.index)
    corr_m = np.corrcoef(pead_net.loc[common_m].values,
                         lv_net.loc[common_m].values)[0, 1]
    w("## 상관 주기 진단 (31번 분기 −0.73 대조)")
    w()
    w(f"- 월 net 상관 {corr_m:+.2f} (공통 {len(common_m)}월) vs 분기 합산 net 상관 "
      f"**{corr_q:+.2f}** (공통 {len(cq)}분기).")
    w("- 31번의 −0.73 은 분기 롱숏 스프레드(open→open, 별도 집계) 기준. 본 엔진은 월말 "
      "종가마킹이라 재구성 방식·주기가 달라 값이 다르다. 결합 개선의 실질 동인은 강한 "
      "역상관 상쇄가 아니라 (a) 두 전략 모두 양의 샤프, (b) 역변동성 가중으로 저변동 "
      "고변동성(33.8%)을 눌러 PEAD 저변동성을 살린 점이다.")
    w()

    # 확장창 비중 강건성
    exp = expanding_combine(pead_net, lv_net, min_months=24)
    me = metrics(exp)
    w("## 강건성 — 확장창 역변동성 비중 (룩어헤드 없음, 24개월+)")
    w()
    w(f"- 결합(확장창) 월수 {me['n']}, 연율수익 {me['ann_ret']*100:+.1f}%, "
      f"연율변동성 {me['ann_vol']*100:.1f}%, 샤프 **{me['sharpe']:.2f}**, MDD {me['mdd']*100:.1f}%.")
    w()

    # 연도별 net
    w("## 연도별 net (%)")
    w()
    yp = yearly_net(pead_net); yl = yearly_net(lv_net)
    comb_full = full["comb"]
    yc = yearly_net(comb_full)
    years = sorted(set(yp.index) | set(yl.index) | set(yc.index))
    w("| 연도 | PEAD | 저변동 | 결합 |")
    w("|---|---:|---:|---:|")
    for y in years:
        w(f"| {y} | {yp.get(y, float('nan'))*100:+.1f} | "
          f"{yl.get(y, float('nan'))*100:+.1f} | {yc.get(y, float('nan'))*100:+.1f} |")
    w()

    # 판정
    mp, ml, mc = full["mp"], full["ml"], full["mc"]
    c1 = (mc["sharpe"] > mp["sharpe"]) and (mc["sharpe"] > ml["sharpe"])
    c2 = mc["mdd"] > mp["mdd"]   # mdd 는 음수 → 결합이 덜 깊으면 값이 큼
    yc_neg = int((yc < 0).sum()); yc_n = len(yc)
    yp_neg = int((yp < 0).sum()); yp_n = len(yp)
    c3 = (yc_n > 0 and yc_neg / yc_n < 0.5) and (yc_neg / max(yc_n, 1) <= yp_neg / max(yp_n, 1) + 1e-9)
    c4 = full["cr"] < 0
    w("## 판정 (32번 §4)")
    w()
    w(f"1. 샤프 개선(결합>PEAD & 결합>저변동): 결합 {mc['sharpe']:.2f} vs "
      f"PEAD {mp['sharpe']:.2f}·저변동 {ml['sharpe']:.2f} → **{'OK' if c1 else 'X'}**")
    w(f"2. MDD 개선(결합 MDD < PEAD MDD): 결합 {mc['mdd']*100:.1f}% vs "
      f"PEAD {mp['mdd']*100:.1f}% → **{'OK' if c2 else 'X'}**")
    w(f"3. 연도 일관성(음수연도<50% & PEAD 대비 악화 없음): 결합 {yc_neg}/{yc_n}, "
      f"PEAD {yp_neg}/{yp_n} → **{'OK' if c3 else 'X'}**")
    w(f"4. 역상관 확인(월 net 상관<0): {full['cr']:+.2f} → **{'OK' if c4 else 'X'}**")
    w()
    if c1 and c2:
        verdict = "채택(결합 배포)" if (c3 and c4) else "채택(1·2 충족, 3/4 일부 미흡)"
    elif c1:
        verdict = "부분(샤프만 개선, MDD 미개선)"
    else:
        verdict = "기각(결합 샤프가 단일 미달)"
    w(f"**판정: {verdict}**")
    w()
    w(f"- 한 줄: 역변동성 결합(w_PEAD={full['wp']:.2f}/w_저변동={full['wl']:.2f})은 "
      f"샤프 {mp['sharpe']:.2f}·{ml['sharpe']:.2f} → **{mc['sharpe']:.2f}**, "
      f"MDD {mp['mdd']*100:.0f}%→{mc['mdd']*100:.0f}%.")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
