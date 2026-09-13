"""뉴스 판단(news_judge) 반응 매매 — 그림자 수집·판정.

사전등록: `docs/research/manju/55-news-judge-shadow-preregistration.md`. `news_judgments`
는 quant-airflow 세션이 2026-09-06 에 막 붙인 테이블이라 이 스크립트를 처음 돌리는
시점엔 표본이 0건이다 — **백필 불가**(뉴스는 라이브만 수집됨, `docs/research/manju/17`)라
과거 검증이 아니라 지금부터 쌓이는 걸 재는 그림자 도구다. 실주문 없음, DB 읽기만.

트리거 = news_judgments 한 행이 다음을 모두 만족:
  price_impact_likely=true, is_stale_repeat=false, sentiment_direction>0(롱온리),
  confidence>=CONFIDENCE_THR, 종목 매핑 존재.

진입 시각 = judged_at 이후 최초 체결가(인과성 — 그 이전 체결은 못 산다).
라벨·비교군·판정선은 `tick_sanity.py`(28번)와 같은 척도를 쓴다 — 청산방식 비교를
나중에 하기 위함. 지평만 N∈{30,60,300}으로 넓혔다(뉴스 확산은 체결강도 트리거보다
늦게 퍼질 수 있어서).

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/news_judge_shadow.py --from 2026-09-06 --to 2026-09-06
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    _arr,
    _load_env,
    _placeholder,
    bootstrap_diff_ci,
    epoch_seconds,
    forward_labels_all,
    quintile_means,
)

CONFIDENCE_THR = 70
HORIZONS = (30, 60, 300)
COST_BP = 50.0
BASELINE_PER_CODE = 300
EXCLUDE_WINDOW_SEC = 300   # 트리거 시각 ±5분은 비교군에서 제외
SESSION_START = "09:00:00"
SESSION_END = "15:20:00"
RNG_SEED = 20260906


def _fetch_triggers(conn, date_from: str, date_to: str) -> list[dict]:
    """조건을 만족하는 news_judgments 행 + 매핑 종목 + published_at(레이턴시용)."""
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        f"""
        select j.source_id, j.ticker, j.related_codes, j.confidence, j.judged_at,
               a.published_at
        from news_judgments j
        left join news_articles a on a.id = j.source_id
        where j.price_impact_likely = true
          and j.is_stale_repeat = false
          and j.sentiment_direction > 0
          and j.confidence >= {ph}
          and j.judged_at >= {ph} and j.judged_at < {ph}
        order by j.judged_at
        """,
        # +09:00 을 명시해 postgres 세션 타임존과 무관하게 KST 거래일 경계로 고정한다 —
        # judged_at 은 timestamptz 라 오프셋 없는 문자열은 세션 타임존에 따라 달라진다.
        (CONFIDENCE_THR, f"{date_from} 00:00:00+09:00", f"{date_to} 23:59:59+09:00"),
    )
    rows = cur.fetchall()
    cur.close()
    out = []
    for source_id, ticker, related_codes, confidence, judged_at, published_at in rows:
        codes = set()
        if ticker:
            codes.add(str(ticker).strip())
        if related_codes:
            codes.update(c.strip() for c in str(related_codes).split(",") if c.strip())
        for code in codes:
            out.append({
                "source_id": source_id, "code": code, "confidence": confidence,
                "judged_at": judged_at, "published_at": published_at,
            })
    return out


def _load_code_day_ticks(conn, code: str, date: str):
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        f"select ts, price from ticks where code = {ph} and ts >= {ph} and ts < {ph} "
        f"order by ts, seq",
        (code, f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None
    import pandas as pd

    ts = pd.to_datetime([r[0] for r in rows])
    ts_sec = epoch_seconds(ts)
    hms = ts.strftime("%H:%M:%S").to_numpy()
    price = np.array([float(r[1]) if r[1] is not None else np.nan for r in rows])
    return ts_sec, hms, price


def build_report(
    trig: dict[int, list], base: dict[int, list], trig_mfe: dict[int, list],
    base_mfe: dict[int, list], qconf: list, qret: list, n_trig: int, n_skipped: int,
    latencies: list, label: str,
) -> str:
    L = []
    L.append(f"# 뉴스 판단 반응매매 — 그림자 — {label}")
    L.append("")
    L.append(
        f"> 사전등록 `docs/research/manju/55`. confidence≥{CONFIDENCE_THR}, "
        f"N∈{list(HORIZONS)}s, 비용 {COST_BP:.0f}bp. **실주문 없음(그림자)**."
    )
    L.append("")
    L.append(f"- 트리거(종목 매핑 성공) {n_trig}건 · 스킵(해당일 체결 없음) {n_skipped}건")
    if latencies:
        lat = _arr(latencies)
        L.append(f"- 레이턴시(judged_at−published_at) 평균 {lat.mean():.1f}s · 중앙 {np.median(lat):.1f}s (n={lat.size})")
    else:
        L.append("- 레이턴시: published_at 매칭 없음(0건)")
    L.append("")
    if n_trig == 0:
        L.append("## 결과")
        L.append("")
        L.append("트리거 0건 — 판정 보류. `news_judgments`에 아직 표본이 쌓이지 않았다.")
        return "\n".join(L)
    L.append("## 1. 트리거군 vs 비트리거군 (forward N초, %)")
    L.append("")
    L.append(
        "| N | 트리거n | 트리거 close 평균/중앙 | 비트리거 close 평균/중앙 | "
        "차 close(부트 95%CI) | 트리거 MFE 평균/중앙 | 비트리거 MFE 평균/중앙 | "
        "차 MFE(부트 95%CI) | net 50bp |"
    )
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    dir_ok = False
    net_ok = False
    for n in HORIZONS:
        t = _arr(trig[n])
        b = _arr(base[n])
        tm = _arr(trig_mfe[n])
        bm = _arr(base_mfe[n])
        d, lo, hi = bootstrap_diff_ci(t, b)
        dm, lom, him = bootstrap_diff_ci(tm, bm)
        tc = t.mean() if t.size else float("nan")
        bc = b.mean() if b.size else float("nan")
        net = (tc * 10000 - COST_BP) if t.size else float("nan")
        if np.isfinite(d) and lo > 0 and np.isfinite(dm) and lom > 0:
            dir_ok = True
        if np.isfinite(net) and net >= 0:
            net_ok = True
        L.append(
            f"| {n} | {t.size} | {tc*100:+.3f}/{np.median(t)*100 if t.size else float('nan'):+.3f} | "
            f"{bc*100:+.3f}/{np.median(b)*100 if b.size else float('nan'):+.3f} | "
            f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | "
            f"{tm.mean()*100 if tm.size else float('nan'):+.3f}/{np.median(tm)*100 if tm.size else float('nan'):+.3f} | "
            f"{bm.mean()*100 if bm.size else float('nan'):+.3f}/{np.median(bm)*100 if bm.size else float('nan'):+.3f} | "
            f"{dm*100:+.3f} [{lom*100:+.3f},{him*100:+.3f}] | {net:+.1f}bp |"
        )
    L.append("")
    L.append("## 2. confidence 5분위 → close 60s")
    L.append("")
    mono_ok = False
    qm = quintile_means(_arr(qconf), _arr(qret))
    if qm:
        L.append("| 분위 | n | confidence 평균 | ret_close_60 평균(%) |")
        L.append("|--:|--:|--:|--:|")
        vals = []
        for q, row in enumerate(qm):
            cnt, smean, rmean = row
            vals.append(rmean)
            L.append(f"| Q{q+1} | {cnt} | {smean:.1f} | {rmean*100:+.3f} |")
        if len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]) and vals[4] > vals[0]:
            mono_ok = True
    else:
        L.append("(표본 부족 — 5분위 불가)")
    L.append("")
    L.append("## 판정")
    L.append("")
    if n_trig < 100:
        L.append(f"**판정 보류.** 표본 {n_trig}건 < 100건(사전등록 §4 최소 표본). 부트스트랩이 아직 무의미하다.")
    elif dir_ok and net_ok and mono_ok:
        L.append("**신호 있음 → 정식 채택 검토.** 방향(+CI, close·MFE 둘 다)·비용후(net≥0)·단조성 모두 충족.")
    else:
        miss = []
        if not dir_ok:
            miss.append("방향(부트 CI가 0 초과 없음)")
        if not net_ok:
            miss.append("비용후 net<0")
        if not mono_ok:
            miss.append("단조성 미충족")
        L.append(f"**신호 약함/미확인.** 미충족: {', '.join(miss)}.")
    L.append("")
    L.append(
        f"_주의: confidence≥{CONFIDENCE_THR} 임계값은 사전 데이터 없이 정한 임의값(사전등록 §6). "
        "표본이 쌓이면 재사전등록 필요._"
    )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="뉴스 판단 반응매매 그림자 sanity (읽기 전용)")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    rng = np.random.default_rng(RNG_SEED)
    trig: dict[int, list] = {n: [] for n in HORIZONS}
    base: dict[int, list] = {n: [] for n in HORIZONS}
    trig_mfe: dict[int, list] = {n: [] for n in HORIZONS}
    base_mfe: dict[int, list] = {n: [] for n in HORIZONS}
    qconf: list = []
    qret: list = []
    latencies: list = []
    n_trig = 0
    n_skipped = 0
    try:
        candidates = _fetch_triggers(conn, args.date_from, args.date_to)
        for cand in candidates:
            judged_at = cand["judged_at"]
            t0_sec = epoch_seconds_scalar(judged_at)
            date = _kst_date(judged_at)
            data = _load_code_day_ticks(conn, cand["code"], date)
            if data is None:
                n_skipped += 1
                continue
            ts_sec, hms, price = data
            i = int(np.searchsorted(ts_sec, t0_sec, side="left"))
            if i >= ts_sec.shape[0]:
                n_skipped += 1
                continue
            n_trig += 1
            if cand["published_at"] is not None:
                pub_sec = epoch_seconds_scalar(cand["published_at"])
                latencies.append(float(t0_sec - pub_sec))
            fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS}
            for n in HORIZONS:
                mfe_i, close_i = fwd[n][0][i], fwd[n][1][i]
                if np.isfinite(close_i):
                    trig[n].append(close_i)
                    trig_mfe[n].append(mfe_i)
            qconf.append(float(cand["confidence"]))
            qret.append(float(fwd[60][1][i]) if np.isfinite(fwd[60][1][i]) else float("nan"))
            in_sess = (hms >= SESSION_START) & (hms <= SESSION_END)
            excl = np.abs(ts_sec - t0_sec) <= EXCLUDE_WINDOW_SEC
            pool = np.flatnonzero(in_sess & ~excl)
            if pool.size:
                k = min(BASELINE_PER_CODE, pool.size)
                for j in rng.choice(pool, k, replace=False):
                    for n in HORIZONS:
                        close_j = fwd[n][1][j]
                        if np.isfinite(close_j):
                            base[n].append(close_j)
                            base_mfe[n].append(fwd[n][0][j])
    finally:
        conn.close()

    label = f"{args.date_from} ~ {args.date_to}"
    report = build_report(trig, base, trig_mfe, base_mfe, qconf, qret, n_trig, n_skipped, latencies, label)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.date_from.replace('-', '')}_{args.date_to.replace('-', '')}"
    out = out_dir / f"news_judge_shadow_{tag}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[news_judge_shadow] wrote {out}")
    return 0


def _kst_date(ts) -> str:
    """timestamptz(UTC) → KST 날짜 문자열. 자정 근처 UTC/KST 날짜가 갈리는 걸 막는다."""
    import pandas as pd

    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Seoul")
    return t.date().isoformat()


def epoch_seconds_scalar(ts) -> int:
    """``ticks.ts``(naive KST) 와 같은 단위로 맞춘 초 단위 epoch int.

    ``news_judgments.judged_at``는 timestamptz(UTC 저장)라 KST wall-clock으로
    변환한 뒤 tz 를 벗겨야 ``ticks``(naive KST) 와 직접 비교할 수 있다. 그대로
    UTC로 비교하면 9시간이 밀린다.
    """
    import pandas as pd

    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Seoul").tz_localize(None)
    return int(t.value // 1_000_000_000)


if __name__ == "__main__":
    sys.exit(main())
