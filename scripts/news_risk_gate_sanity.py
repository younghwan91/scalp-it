"""뉴스·공시 리스크 게이트 sanity. 사전등록 `docs/research/manju/73`.

72번(짝꿍매매 라이브 규칙 백테스트, 이미 기각) 캐시를 재사용하고, DART
material_type='악재' 공시가 거래일 기준 [D-3, D] 안에 대장 또는 후속 종목에
있으면 그 거래를 제외한다 — 신규 진입 차단 게이트 개념 검증(§0). Toss 뉴스는
라이브 전용이라 이 오프라인 측정엔 못 쓴다.

읽기 전용, DB만 읽는다, 실주문 없음.

사용:
  uv run python scripts/news_risk_gate_sanity.py \
      --cache /path/pair_live.pkl --dart-2024 data/dart_2024.db --dart data/dart.db
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

import numpy as np
import pandas as pd

COST = 0.0064
LOOKBACK_DAYS = 3
N_FOLDS = 4

# 72번 확정 결과(비교 기준선) — 하드코딩이 아니라 그 문서 §5 조건에 쓰인 값 그대로.
BASELINE_NET64_MEAN = -0.00455
BASELINE_NEG_FOLDS = 4
BASELINE_WIN_RATE = 0.480


def load_bad_news(dart_2024_path: str, dart_path: str) -> pd.DataFrame:
    """두 DART sqlite 에서 material_type='악재' 공시만 (code, date) 로 합친다.

    겹치는 기간은 rcept_no 로 중복 제거하고, dart.db(최신) 쪽을 우선한다 — 최신
    수집이 정정·보완을 반영했을 수 있어서다.
    """
    frames = []
    for path, priority in ((dart_2024_path, 0), (dart_path, 1)):
        p = pathlib.Path(path)
        if not p.exists():
            print(f"[경고] {path} 없음 — 건너뜀", file=sys.stderr)
            continue
        conn = sqlite3.connect(path)
        df = pd.read_sql(
            "select rcept_no, rcept_dt, code from disclosures where material_type='악재'",
            conn)
        conn.close()
        df["priority"] = priority
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["code", "date"])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("priority").drop_duplicates("rcept_no", keep="last")
    all_df["date"] = pd.to_datetime(all_df["rcept_dt"], format="%Y%m%d")
    all_df["code"] = all_df["code"].astype(str).str.strip()
    return all_df[["code", "date"]].drop_duplicates()


def has_recent_bad_news(code: str, day: pd.Timestamp, bad: pd.DataFrame) -> bool:
    lo = day - pd.Timedelta(days=LOOKBACK_DAYS)
    hit = bad[(bad["code"] == code) & (bad["date"] >= lo) & (bad["date"] <= day)]
    return not hit.empty


def _fold_stats(r: pd.Series, n_folds: int) -> list[float]:
    folds = np.array_split(np.arange(len(r)), n_folds)
    return [float(r.iloc[f].mean() - COST) for f in folds if len(f)]


def build_report(tr: pd.DataFrame, bad: pd.DataFrame) -> str:
    tr = tr.sort_values("date").reset_index(drop=True)
    gated = tr["lead"].combine(tr["date"], lambda c, d: has_recent_bad_news(c, d, bad)) \
        | tr["fol"].combine(tr["date"], lambda c, d: has_recent_bad_news(c, d, bad))
    kept = tr[~gated].reset_index(drop=True)

    L = ["# 뉴스·공시 리스크 게이트 sanity 결과", ""]
    L.append("> 사전등록 `docs/research/manju/73`. 72번 캐시 615건 재사용.")
    L.append("")
    L.append(f"- 전체 615건 중 악재 공시(D-3~D, 대장 또는 후속) 게이트에 걸려 제외: "
              f"{int(gated.sum())}건")
    L.append(f"- 남은 표본: {len(kept)}건")
    L.append("")

    if len(kept) < 100:
        L.append(f"**표본 부족(<100, n={len(kept)}) — 판정 불가로 기록.**")
        return "\n".join(L)

    r = kept["ret"]
    net = r - COST
    folds = _fold_stats(net, N_FOLDS)
    neg_folds = sum(1 for f in folds if f < 0)
    win_rate = float((r > 0).mean())

    L.append("## 게이트 적용 후 vs 72번 전체(기준선)")
    L.append("")
    L.append("| 지표 | 72번 전체(기준선) | 게이트 적용 후 |")
    L.append("|---|--:|--:|")
    L.append(f"| net64 평균 | {BASELINE_NET64_MEAN:+.3%} | {net.mean():+.3%} |")
    L.append(f"| 음수 폴드(4개 중) | {BASELINE_NEG_FOLDS} | {neg_folds} |")
    L.append(f"| 승률 | {BASELINE_WIN_RATE:.1%} | {win_rate:.1%} |")
    L.append(f"| n | 615 | {len(kept)} |")
    L.append("")
    L.append("시간순 4폴드 net64: " + ", ".join(f"{f:+.3%}" for f in folds))
    L.append("")

    cond = {
        "1 net64평균 개선": net.mean() > BASELINE_NET64_MEAN,
        "2 음수폴드 감소": neg_folds < BASELINE_NEG_FOLDS,
        "3 승률 상승": win_rate > BASELINE_WIN_RATE,
        "4 최소표본100 유지": len(kept) >= 100,
    }
    L.append("## 판정 (사전등록 §5)")
    L.append("")
    L.append("  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    L.append("")
    L.append(f"**{'채택(추가 조사 가치)' if all(cond.values()) else '기각'}**"
              " — 채택이어도 실배선 아님(§6).")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="72번 pair_live_rules_sanity.py --cache 산출물")
    ap.add_argument("--dart-2024", default="data/dart_2024.db")
    ap.add_argument("--dart", default="data/dart.db")
    ap.add_argument("--out", default="data/news_risk_gate_sanity.md")
    args = ap.parse_args(argv)

    data = pd.read_pickle(args.cache)
    tr = data["trades"]
    bad = load_bad_news(args.dart_2024, args.dart)
    print(f"악재 공시 {len(bad):,}건 (종목-일 유니크)", file=sys.stderr)

    report = build_report(tr, bad)
    print(report)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n[news_risk_gate_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
