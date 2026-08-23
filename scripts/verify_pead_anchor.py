"""PEAD 앵커 정교화 측정 — 사전등록 27번 그대로 1회.

earnings.avail_date(법정마감 일괄부여)를 DART 실적 관련 공시 rcept_dt 로 교체하고
PEAD 롱숏 Q10-Q1(+40)을 재측정 → 앵커가 날카로워지면 엣지가 커지는가.
진입/데실/서프라이즈 정의는 23번 그대로(pead_events.build_events 재사용), avail_date 소스만 교체.

- 주 앵커: 잠정실적(material_type='실적', 영업(잠정)실적 등). 커버 31.6%, median +31일.
- 진단 앵커: 정기보고(분기/반기/사업보고서). 커버 86%, shift 0(정교화 없음 확인).
- 공정 비교: 동일 이벤트 집합(잠정 매칭)에서 (A)원avail vs (B)잠정앵커.
- DART 커버리지 2016-09~2023-12 → avail<=2024-01(2016Q1~2023Q3)만 정교화 가능.

결과를 스크래치패드 pead_anchor.md 로 출력. DB/파일 쓰기 없음(sqlite 읽기 전용).
DB: 환경변수 KR_QUANT_DB. DART: data/dart_full.db.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from pead_events import build_events, load_raw

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/pead_anchor.md")
DART = "/home/young/Documents/git/scalp-it/data/dart_full.db"
RT = 0.0034
BORROW_40D = 0.005
PRIMARY = 40
DART_END = pd.Timestamp("2024-01-31")   # avail 컷: DART 커버 내


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    return m, m / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def period_end(p):
    y = int(p[:4]); end = {"1": "03", "2": "06", "3": "09", "4": "12"}[p[-1]]
    return pd.Timestamp(f"{y}-{end}-01") + pd.offsets.MonthEnd(0)


def load_dart():
    c = sqlite3.connect(DART)
    disc = pd.read_sql(
        "select rcept_dt, code, report_nm, material_type from disclosures "
        "where material_type in ('실적','정기보고')", c)
    c.close()
    disc["rcept"] = pd.to_datetime(disc.rcept_dt, format="%Y%m%d")
    prov = disc[(disc.material_type == "실적") & disc.report_nm.str.contains(
        r"영업.잠정.실적|영업실적|매출액또는손익구조", regex=True)].copy()
    per = disc[(disc.material_type == "정기보고") & disc.report_nm.str.contains(
        r"\(\d{4}\.\d{2}\)", regex=True)].copy()
    per["ptag"] = per.report_nm.str.extract(r"\((\d{4}\.\d{2})\)")
    return prov, per


def match_anchors(earn, prov, per):
    """earn 각 행에 prov_dt, per_dt 부여(매칭). avail<=DART_END 만."""
    earn = earn[earn.avail_date <= DART_END].copy()
    earn["pend"] = earn.period.apply(period_end)
    prov_g = {c: g.sort_values("rcept") for c, g in prov.groupby("code")}
    per_g = {c: g for c, g in per.groupby("code")}

    def row(r):
        pv = pd.NaT; pr = pd.NaT
        g = prov_g.get(r.code)
        if g is not None:
            m = g[(g.rcept > r.pend) & (g.rcept <= r.avail_date + pd.Timedelta(days=15))]
            if len(m):
                pv = m.rcept.iloc[0]
        g = per_g.get(r.code)
        if g is not None:
            tag = f"{r.pend.year}.{r.pend.strftime('%m')}"
            m = g[g.ptag == tag]
            if len(m):
                pr = m.rcept.min()
        return pd.Series({"prov_dt": pv, "per_dt": pr})

    earn = earn.join(earn.apply(row, axis=1))
    # clip: 앵커가 원avail 보다 늦으면 원avail 로 (룩어헤드 방지)
    earn["prov_anchor"] = earn[["prov_dt", "avail_date"]].min(axis=1)
    earn.loc[earn.prov_dt.isna(), "prov_anchor"] = pd.NaT
    earn["per_anchor"] = earn[["per_dt", "avail_date"]].min(axis=1)
    earn.loc[earn.per_dt.isna(), "per_anchor"] = pd.NaT
    return earn


def ls_spread(ev):
    """유니버스&non-limit, 분기내 데실, 롱숏 Q10-Q1 +40 분기 시계열 반환."""
    base = ev[ev["in_universe"] & ~ev["entry_limit_up"]].copy()

    def _q(s):
        if s.nunique() < 10:
            return pd.Series([np.nan] * len(s), index=s.index)
        return pd.qcut(s.rank(method="first"), 10, labels=range(1, 11)).astype(float)
    base["decile"] = base.groupby("period")["sue_raw"].transform(_q)
    base = base[base.decile.notna()]
    base["decile"] = base.decile.astype(int)
    d = base[base[f"ret{PRIMARY}"].notna()].copy()
    q10 = d[d.decile == 10].groupby("period")[f"ret{PRIMARY}"].mean()
    q1 = d[d.decile == 1].groupby("period")[f"ret{PRIMARY}"].mean()
    dec = d.groupby("decile")[f"ret{PRIMARY}"].mean()
    return (q10 - q1).dropna(), dec, d


def restrict(ev, keep_keys):
    ev = ev.copy()
    ev["key"] = list(zip(ev.code, ev.period))
    return ev[ev.key.isin(keep_keys)]


def main():
    Lst = []
    def w(s=""):
        Lst.append(s); print(s)

    earn, bars, raw, spac = load_raw()
    prov, per = load_dart()
    matched = match_anchors(earn, prov, per)

    w("# PEAD 앵커 정교화 측정 결과 (사전등록 27번, 1회)")
    w()
    cov = matched
    w(f"- DART 커버 컷(avail<=2024-01): {len(cov):,}이벤트, {cov.period.nunique()}분기 "
      f"({cov.period.min()}~{cov.period.max()}).")
    w(f"- 잠정실적 매칭 {cov.prov_dt.notna().mean():.1%}, 정기보고 매칭 {cov.per_dt.notna().mean():.1%}.")
    sh = (cov.avail_date - cov.prov_anchor).dt.days.dropna()
    w(f"- 잠정앵커 앞당김(avail−anchor) 중앙 {sh.median():.0f}일 (p25 {sh.quantile(.25):.0f}, "
      f"p75 {sh.quantile(.75):.0f}).")
    w()

    # === 주 비교: 잠정 매칭 부분집합, A(원avail) vs B(잠정앵커) ===
    prov_keys = set(zip(cov[cov.prov_dt.notna()].code, cov[cov.prov_dt.notna()].period))
    # variant frames (동일 부분집합)
    sub = cov[cov.prov_dt.notna()].copy()
    earn_A = sub.copy()  # avail_date = 원본
    earn_B = sub.copy()
    earn_B["avail_date"] = earn_B["prov_anchor"]

    # 교집합: 두 변형 모두 유니버스&ret40 존재하는 (code,period)
    evA_full = build_events(earn_A, bars, raw, spac)
    evB_full = build_events(earn_B, bars, raw, spac)
    def valid_keys(ev):
        v = ev[ev.in_universe & ~ev.entry_limit_up & ev[f"ret{PRIMARY}"].notna()]
        return set(zip(v.code, v.period))
    keep = valid_keys(evA_full) & valid_keys(evB_full) & prov_keys

    w(f"## 주 비교 (잠정 매칭 부분집합, 동일 이벤트 {len(keep):,}건)")
    spA, decA, _ = ls_spread(restrict(evA_full, keep))
    spB, decB, _ = ls_spread(restrict(evB_full, keep))
    mA, tA, nA = tstat(spA)
    mB, tB, nB = tstat(spB)
    netA = mA - 2 * RT - BORROW_40D
    netB = mB - 2 * RT - BORROW_40D
    spearA = np.corrcoef(decA.sort_index().index, decA.sort_index().rank())[0, 1]
    spearB = np.corrcoef(decB.sort_index().index, decB.sort_index().rank())[0, 1]
    w("| 앵커 | 롱숏 Q10−Q1 분기평균 | t | net(68bp+0.5%p) | Spearman | 분기N |")
    w("|---|---|---|---|---|---|")
    w(f"| A 원 avail_date | {mA*100:+.2f}% | {tA:+.2f} | {netA*100:+.2f}% | {spearA:+.2f} | {nA} |")
    w(f"| B 잠정실적 rcept_dt | {mB*100:+.2f}% | {tB:+.2f} | {netB*100:+.2f}% | {spearB:+.2f} | {nB} |")
    rel = (netB - netA) / abs(netA) if netA != 0 else np.nan
    w(f"- net 상대 변화 B/A: {rel*100:+.1f}%  (판정: +20% 이상이면 성공)")
    w()

    # === 진단: 정기보고 앵커(부분집합) ===
    per_sub = cov[cov.per_dt.notna()].copy()
    earn_pA = per_sub.copy()
    earn_pB = per_sub.copy(); earn_pB["avail_date"] = earn_pB["per_anchor"]
    evpA = build_events(earn_pA, bars, raw, spac)
    evpB = build_events(earn_pB, bars, raw, spac)
    keep_p = valid_keys(evpA) & valid_keys(evpB) & set(zip(per_sub.code, per_sub.period))
    spPA, _, _ = ls_spread(restrict(evpA, keep_p))
    spPB, _, _ = ls_spread(restrict(evpB, keep_p))
    mPA, tPA, _ = tstat(spPA); mPB, tPB, _ = tstat(spPB)
    w(f"## 진단: 정기보고 앵커 (부분집합 {len(keep_p):,}건, shift~0 예상)")
    w(f"- A 원avail: 롱숏 {mPA*100:+.2f}% t={tPA:+.2f}; B 정기보고앵커: {mPB*100:+.2f}% t={tPB:+.2f} "
      f"→ 차이 미미하면 정기보고=정교화 없음 확인.")
    w()

    # === 맥락: 원 PEAD 전체(23번 재현, 전 기간) ===
    ev_all = build_events(earn, bars, raw, spac)
    (sp_all, _, _) = ls_spread(ev_all)
    m_all, t_all, n_all = tstat(sp_all)
    w("## 맥락: 원 PEAD 전체(전 기간, 23번 재현)")
    w(f"- 롱숏 Q10−Q1 분기평균 {m_all*100:+.2f}%, t={t_all:+.2f}, N={n_all}분기 "
      f"(net {(m_all-2*RT-BORROW_40D)*100:+.2f}%).")
    w()

    # === 판정 ===
    c_mono = spearB > 0
    c_t = abs(tB) >= 2
    rel_ok = bool(np.isfinite(rel)) and rel >= 0.20
    success = rel_ok and c_t and c_mono and (netB > netA)
    partial = (netB > netA) and c_mono
    verdict = "정교화 성공" if success else ("부분" if partial else "정교화 불가")
    w("## 판정 (사전등록 3절)")
    w(f"- 잠정앵커 net {netB*100:+.2f}% vs 원앵커 {netA*100:+.2f}% (동일 이벤트), "
      f"상대 {rel*100:+.1f}%, tB={tB:+.2f}, 단조 {'OK' if c_mono else 'FAIL'}.")
    w("- 정기보고 앵커는 shift~0 → 그 경로는 정교화 불가(자동).")
    w(f"\n### 최종: **{verdict}**")

    with open(OUT, "w") as f:
        f.write("\n".join(Lst) + "\n")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
