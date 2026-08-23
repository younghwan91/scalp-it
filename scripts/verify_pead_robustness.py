"""PEAD 강건성 측정 — 사전등록 24번 그대로 1회.

23번 채택(롱숏 +4.38%, net +3.2%) 이후 실전 실행가능성 스트레스:
  A 유동성 계층(5/50/200/500억) B SUE 극단처리(raw/std/rank)
  C 비용 민감도 D 시총·섹터 중립화 E 보유기간 확인 F 숏 실행가능성.
결과를 스크래치패드 pead_robustness.md 로 출력. DB 쓰기 없음.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2
from pead_events import build_events, load_raw

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/pead_robustness.md")
RT = 0.0034          # 왕복 34bp (한 다리)
BORROW_40D = 0.005   # 숏 차입 연3% * ~2개월 ≈ 0.5%p (주보유 +40)
PRIMARY = 40
LIQ_TIERS = [(500, "5억"), (5000, "50억"), (20000, "200억"), (50000, "500억")]


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    return m, m / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def decile_within(df, groupcols, col, minn=30):
    """group 안에서 col 을 데실(1..10). group 크기 < minn 이면 NaN(경계 불안정 회피)."""
    def _q(s):
        if s.notna().sum() < minn:
            return pd.Series(np.nan, index=s.index)
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(float)
    return df.groupby(groupcols, group_keys=False)[col].apply(_q)


def longshort_by_period(d, retcol, deccol):
    """분기별 Q10-Q1 스프레드 시계열 + 롱온리 Q10, 유니버스 평균."""
    dd = d[d[retcol].notna() & d[deccol].notna()]
    q10 = dd[dd[deccol] == 10].groupby("period")[retcol].mean()
    q1 = dd[dd[deccol] == 1].groupby("period")[retcol].mean()
    uni = dd.groupby("period")[retcol].mean()
    spread = (q10 - q1).dropna()
    return spread, q10, q1, uni


def main():
    L = []
    def w(s=""):
        L.append(s); print(s)

    earn, bars, raw, spac = load_raw()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    names = pd.read_sql("select code, name, sector from stocks", conn)
    sector_map = dict(zip(names.code, names.sector))
    shares = pd.read_sql(
        "select code, date, shares_outstanding from shares_outstanding_history "
        "order by code, date", conn, parse_dates=["date"])
    ss_codes = set(pd.read_sql("select distinct code from short_selling", conn).code)
    ss_min = pd.read_sql("select min(date) mn from short_selling", conn).mn[0]
    conn.close()

    ev = build_events(earn, bars, raw, spac)

    # --- SUE 표준화(직전 8분기 서프라이즈 절대액 표준편차) ---
    e = earn.copy()
    e = e[e["op_income"].notna() & e["op_income_prior"].notna() & (e["op_income_prior"] != 0)]
    e["surp"] = e["op_income"].astype(float) - e["op_income_prior"].astype(float)
    e = e.sort_values(["code", "period"])
    e["sd8"] = (e.groupby("code")["surp"]
                .transform(lambda s: s.shift(1).rolling(8, min_periods=8).std()))
    e["sue_std"] = np.where(e["sd8"] > 0, e["surp"] / e["sd8"], np.nan)
    ev = ev.merge(e[["code", "period", "sue_std"]], on=["code", "period"], how="left")

    # --- 시가총액(avail as-of 발행주식수 × 진입 시가) ---
    ev_s = ev[["code", "avail_date"]].dropna().sort_values("avail_date")
    ev_s = pd.merge_asof(ev_s, shares.sort_values("date"),
                         left_on="avail_date", right_on="date", by="code",
                         direction="backward")
    ev = ev.merge(ev_s[["code", "avail_date", "shares_outstanding"]].drop_duplicates(
        ["code", "avail_date"]), on=["code", "avail_date"], how="left")
    ev["mcap"] = ev["shares_outstanding"] * ev["entry_open"]
    ev["sector"] = ev["code"].map(sector_map)

    # B~F 기준 유니버스 = 23번과 동일한 5억(500백만) 유동성 하한
    base_all = ev[~ev["spac"] & ev["sue_raw"].notna() & ~ev["entry_limit_up"]
                  & (ev["liq"] >= 500)].copy()

    w("# PEAD 강건성 측정 (사전등록 24번, 1회)")
    w()
    w("판정 = 롱숏 Q10−Q1, +40거래일. 롱온리 Q10 병행. 원 SUE=(oi-oip)/|oip|.")
    w(f"비용 기본: 왕복 {RT*1e4:.0f}bp/다리, 숏차입 +40일 {BORROW_40D*100:.1f}%p.")
    w()

    # ========== A. 유동성 계층 ==========
    w("## A. 유동성 계층 (진입 직전 20일 평균 거래대금 하한)")
    w()
    w("| 계층 | 이벤트수 | 분기당중앙N | 롱숏 평균% | t | net34% | 롱온리Q10 net34% | Q10 초과%(t) |")
    w("|---|---|---|---|---|---|---|---|")
    tier_res = {}
    for floor, lab in LIQ_TIERS:
        d = base_all[base_all["liq"] >= floor].copy()
        d["decile"] = decile_within(d, "period", "sue_raw")
        rc = f"ret{PRIMARY}"
        spread, q10, q1, uni = longshort_by_period(d, rc, "decile")
        m, t, _n = tstat(spread)
        net = m - 2 * RT - BORROW_40D
        dd = d[d[rc].notna() & d["decile"].notna()]
        nq_med = int(dd[dd.decile.isin([1, 10])].groupby("period").size().median() / 2) \
            if len(dd) else 0
        lo_net = q10.mean() - RT
        excess = (q10 - uni).dropna()
        me, te, _ = tstat(excess)
        n_ev = int(dd.shape[0])
        w(f"| {lab} | {n_ev:,} | {nq_med} | {m*100:+.2f} | {t:+.2f} | {net*100:+.2f} | "
          f"{lo_net*100:+.2f} | {me*100:+.2f}({te:+.2f}) |")
        tier_res[floor] = {"m": m, "t": t, "net": net, "lo_net": lo_net,
                           "exc": me, "exc_t": te, "n": n_ev}
    w()
    r500 = tier_res[50000]
    w(f"**★500억:** 롱숏 net34 {r500['net']*100:+.2f}%, t={r500['t']:+.2f} "
      f"→ {'net>0 & t>2 통과' if (r500['net']>0 and r500['t']>2) else '미달'}. "
      f"롱온리 Q10 초과 {r500['exc']*100:+.2f}% (t={r500['exc_t']:+.2f}).")
    w()

    # ========== B. SUE 극단 처리 ==========
    w("## B. 서프라이즈 극단 처리 (유동성 5억 유니버스, 나란히)")
    w()
    d = base_all.copy()  # 5억 = 23번 유니버스
    rc = f"ret{PRIMARY}"
    w("| SUE 정의 | Q1 평균% | Q10 평균% | Spearman | 롱숏 평균% | t | net34% |")
    w("|---|---|---|---|---|---|---|")
    b_res = {}
    for col, lab in [("sue_raw", "원(raw)"), ("sue_std", "표준화(std,8분기)"),
                     ("sue_raw", "순위(rank)")]:
        dc = d[d[col].notna()].copy()
        dc["dec"] = decile_within(dc, "period", col)
        g = dc[dc[rc].notna() & dc["dec"].notna()].groupby("dec")[rc].mean()
        sp = g.sort_index()
        spear = np.corrcoef(sp.index.astype(float), sp.rank())[0, 1] if len(sp) > 2 else np.nan
        spread, q10, q1, _ = longshort_by_period(dc, rc, "dec")
        m, t, _ = tstat(spread)
        net = m - 2 * RT - BORROW_40D
        q1v = g.get(1, np.nan) * 100
        q10v = g.get(10, np.nan) * 100
        note = " *(rank: decile 이 rank기반→raw와 동일)*" if lab.startswith("순위") else ""
        w(f"| {lab}{note} | {q1v:+.2f} | {q10v:+.2f} | {spear:+.2f} | {m*100:+.2f} | {t:+.2f} | {net*100:+.2f} |")
        b_res[lab] = {"spear": spear, "m": m, "t": t, "net": net}
    w()
    std_r = b_res["표준화(std,8분기)"]
    w(f"**표준화(초저베이스 왜곡 제거):** Spearman {std_r['spear']:+.2f}, 롱숏 {std_r['m']*100:+.2f}% "
      f"t={std_r['t']:+.2f} → {'생존(엣지는 실제 신호)' if (std_r['spear']>0 and std_r['m']>0 and std_r['t']>2) else '약화/붕괴(저베이스 왜곡 기여)'}.")
    w()

    # ========== C. 비용 민감도 ==========
    w("## C. 비용 민감도 (유동성 5억, 롱숏 +40 net%)")
    w()
    d = base_all.copy()
    d["decile"] = decile_within(d, "period", "sue_raw")
    spread, _, _, _ = longshort_by_period(d, f"ret{PRIMARY}", "decile")
    gross, _, _ = tstat(spread)
    w(f"gross 롱숏 = {gross*100:+.2f}%.")
    w()
    w("| 왕복bp(양다리) \\ 숏차입 | 0.5%p | 2%p | 5%p |")
    w("|---|---|---|---|")
    for rt in [34, 68, 100, 150]:
        cells = []
        for bor in [0.005, 0.02, 0.05]:
            net = gross - (rt / 1e4) - bor  # rt = 양다리 합산 왕복bp (base=68bp와 일치)
            cells.append(f"{net*100:+.2f}")
        w(f"| {rt}bp | {cells[0]} | {cells[1]} | {cells[2]} |")
    w()
    # 손익분기 왕복bp (숏차입별)
    be = []
    for bor in [0.005, 0.02, 0.05]:
        bp = (gross - bor) * 1e4  # net=0 -> 왕복bp = (gross-bor)*1e4
        be.append(f"숏{bor*100:.1f}%p→왕복 {bp:.0f}bp")
    w(f"**손익분기(net=0):** {', '.join(be)}.")
    w()

    # ========== D. 시총 / 섹터 중립화 ==========
    w("## D. 중립화 (유동성 5억, 롱숏 +40)")
    w()
    d = base_all.copy()
    rc = f"ret{PRIMARY}"
    # 시총 3분위 (분기내)
    d["size_t"] = d.groupby("period")["mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=[1, 2, 3]).astype(float)
        if s.notna().sum() >= 30 else np.nan)
    d["dec_size"] = decile_within(d, ["period", "size_t"], "sue_raw", minn=30)
    d["dec_sec"] = decile_within(d, ["period", "sector"], "sue_raw", minn=30)
    w("| 중립화 | 롱숏 평균% | t | net34% | 이벤트수(데실배정) |")
    w("|---|---|---|---|---|")
    d_res = {}
    for deccol, lab in [("dec_size", "시총 3분위 내"), ("dec_sec", "섹터 내")]:
        spread, _, _, _ = longshort_by_period(d, rc, deccol)
        m, t, _ = tstat(spread)
        net = m - 2 * RT - BORROW_40D
        nev = int(d[d[rc].notna() & d[deccol].notna()].shape[0])
        w(f"| {lab} | {m*100:+.2f} | {t:+.2f} | {net*100:+.2f} | {nev:,} |")
        d_res[lab] = {"m": m, "t": t, "net": net}
    w()

    # ========== E. 보유기간 확인 ==========
    w("## E. 보유기간 확인 (유동성 5억, 롱숏)")
    w()
    d = base_all.copy()
    d["decile"] = decile_within(d, "period", "sue_raw")
    w("| 보유 | 롱숏 평균% | t | net% |")
    w("|---|---|---|---|")
    for H in (20, 40, 60):
        spread, _, _, _ = longshort_by_period(d, f"ret{H}", "decile")
        m, t, _ = tstat(spread)
        net = m - 2 * RT - BORROW_40D * (H / PRIMARY)
        w(f"| +{H} | {m*100:+.2f} | {t:+.2f} | {net*100:+.2f} |")
    # 분기 간격 vs +40
    per_gap = base_all.groupby("code")["avail_date"].apply(
        lambda s: s.sort_values().diff().dt.days.median()).median()
    w(f"\n분기 avail 간격 중앙값 ≈ {per_gap:.0f}일(캘린더) → +40거래일(≈60일) 청산 후 다음 신호까지 겹침 없음.")
    w()

    # ========== F. 숏 실행가능성 ==========
    w("## F. 하위 데실(Q1) 숏 실행가능성")
    w()
    d = base_all.copy()
    d["decile"] = decile_within(d, "period", "sue_raw")
    q1 = d[(d.decile == 1) & (d[f"ret{PRIMARY}"].notna())]
    q1_recent = q1[q1["avail_date"] >= pd.Timestamp(ss_min)]
    frac_all = q1["code"].isin(ss_codes).mean()
    frac_recent = q1_recent["code"].isin(ss_codes).mean()
    w(f"short_selling 커버: {ss_min}~ (2545종목). Q1 이벤트 중 등장 코드 비율.")
    w(f"- 전체 Q1({len(q1):,}건): {frac_all*100:.1f}% (단, 2024-05 이전은 프록시 부재).")
    w(f"- 2024-05+ Q1({len(q1_recent):,}건): **{frac_recent*100:.1f}%**.")
    verd_short = "숏 실행가능(롱숏 유지)" if frac_recent >= 0.60 else "숏 비현실 → 롱온리+지수숏 헤지 권고"
    w(f"- 판정(≥60%): **{verd_short}**")
    w()

    # ========== 종합 ==========
    w("## 종합 판정")
    c500 = r500["net"] > 0 and r500["t"] > 2
    cstd = std_r["spear"] > 0 and std_r["m"] > 0 and std_r["t"] > 2
    cneut = all(v["m"] > 0 and v["t"] > 2 for v in d_res.values())
    cshort = frac_recent >= 0.60
    w(f"- A ★500억 롱숏 net>0 & t>2: {'OK' if c500 else 'FAIL'}")
    w(f"- B 표준화 SUE 생존: {'OK' if cstd else 'FAIL'}")
    w(f"- D 시총·섹터 중립 생존: {'OK' if cneut else 'FAIL'}")
    w(f"- F 숏 실행가능 ≥60%: {'OK' if cshort else 'FAIL'}")
    if c500 and cstd and cneut and cshort:
        v = "실전 채택 (풀 롱숏)"
    elif c500 and cstd and cneut and not cshort:
        v = "실전 채택 (롱온리 Q10 + 지수숏 헤지) — 숏 다리 비현실"
    elif (not c500) and cstd:
        lo50 = tier_res[5000]
        v = f"소형주 한정 판정 — 500억서 롱숏 t={r500['t']:+.2f}(미달), 50억 net {lo50['net']*100:+.2f}%. 실전 실행 제약."
    else:
        v = "기각 — 표준화/중립화서 붕괴(저베이스·섹터 산물)"
    w(f"\n### 최종: **{v}**")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
