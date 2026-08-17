"""`swing-watch` — ①층(수급) 전진 기록. 오늘 후보를 남기고, 지난 기록을 채점한다.

①층은 2024+ 일관 유니버스에서 전면 통과했다
(`docs/research/2026-08-17-supply-layer-pass.md`). 그런데 같은 문서가 "아직 믿지
못할 이유" 넷을 함께 적어뒀다 — 실효 독립 표본이 14개뿐이고, 기간이 단일 레짐이며,
널리 알려진 효과이고, 포트폴리오가 아니라 시그널 지표다. **넷 다 백테스트로는
못 푼다.** 손대지 않은 미래 데이터만이 푼다. 그래서 이 CLI 의 값어치는 기능이
아니라 **오늘부터 도는 것** 에 있다.

`swing-today` 와 겹쳐 보이지만 다른 물건이다. 저쪽은 ③층(뉴스+LLM)이라 계획 2·3
산출물이 있어야 돌고, 날짜당 상위 N 을 고른다. 이쪽은 수급만 쓰므로 가격·수급 DB
하나로 오늘 돌릴 수 있고, **절대 임계** 만 쓴다 (스펙 §6) — 재료가 시원찮은 날은
0 건이 정답이다. 0 건은 실패가 아니므로 종료 코드도 0 이다.

임계는 walk-forward 폴드별 계산 결과의 ``latest_threshold`` 를 쓴다. 전역 풀링은
금지다 — 폴드마다 예측값의 절대 수준이 달라서, 전역 임계를 쓰면 스케일이 큰 폴드가
통과분을 독식한다. 실측에서 그 버그 하나로 상위 286건의 초과수익이 +2.91% 에서
−1.77% 로 뒤집혔다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .adapters.db import connect, load_prices, load_stock_master, load_supply_demand
from .dataset import FEATURE_COLUMNS
from .layers import build_layered_dataset
from .model.baseline import walk_forward_predict
from .model.layered import fit_final_model
from .report.forward_log import (
    append_forward_log,
    build_records,
    read_forward_log,
    score_forward_log,
)
from .report.threshold import concurrency_threshold

LAYER = "supply"
# ①층 판정은 2026-08-17 기준 PASS 다 (supply-layer-pass.md, 네 조건 전부).
# 기록에 판정을 박아두는 이유는, 나중에 판정이 바뀌어도 그때 무엇을 믿고 샀는지가
# 남아야 전진 기록이 증거로 쓰이기 때문이다.
GATE_VERDICT = "PASS"


def _table(candidates: pd.DataFrame, names: dict[str, str], threshold: float) -> str:
    """사람이 읽는 후보 표. 원자료는 찍지 않는다 — 판단에 쓰는 다섯 숫자만."""
    header = (
        "| 종목코드 | 종목명 | 예측 | 임계 | 스마트머니 | 거래대금배수 | adv(백만) |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    rows = [
        "| {code} | {name} | {pred:+.4f} | {thr:+.4f} | {smart:+.4f} | {tv:.1f} | {adv:,.0f} |".format(
            code=row["code"],
            name=names.get(str(row["code"]), ""),
            pred=row["prediction"],
            thr=threshold,
            smart=row["smart_money_net_5d_scaled"],
            tv=row["tv_ratio"],
            adv=10.0 ** row["log_adv"],
        )
        for _, row in candidates.iterrows()
    ]
    return "\n".join([header, *rows])


def _plan_dates(prices: pd.DataFrame, asof: pd.Timestamp, hold_end: int):
    """진입(다음 거래일 시가)과 목표 청산(그로부터 hold_end 거래일) 날짜."""
    calendar = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
    matched = calendar[calendar == asof]
    idx = int(matched.index[0]) if len(matched) else len(calendar) - 1
    entry = calendar.iloc[min(idx + 1, len(calendar) - 1)]
    exit_idx = idx + 1 + hold_end
    if exit_idx < len(calendar):
        return entry, calendar.iloc[exit_idx]
    # 달력에 아직 없는 미래는 영업일 근사로 표시한다
    return entry, entry + pd.tseries.offsets.BDay(hold_end)


def _score(train: pd.DataFrame, log_path: str) -> int:
    """전진 기록 중 라벨이 붙은 것만 채점한다. 나머지는 아직 시간이 안 됐을 뿐이다."""
    log = read_forward_log(log_path)
    if log.empty:
        print(f"전진 기록이 비어 있다: {log_path}")
        return 0

    log = log[log["layer"] == LAYER].copy()
    scored = score_forward_log(log, train[["code", "date", "excess_return"]])
    done = scored[scored["realized_excess"].notna()]
    pending = len(scored) - len(done)

    lines = [
        f"# ①층 전진 기록 채점 — {log_path}",
        "",
        f"기록 {len(scored)}건 · 채점 {len(done)}건 · 미채점 {pending}건 (아직 D+40 전)",
        "",
    ]
    if done.empty:
        lines.append("채점 가능한 기록이 아직 없다. 기록을 계속 쌓아라.")
        print("\n".join(lines))
        return 0

    hit = float((done["realized_excess"] > 0).mean())
    lines += [
        (f"평균 초과수익 **{done['realized_excess'].mean():+.2%}** · "
         f"중앙값 {done['realized_excess'].median():+.2%} · 승률 {hit:.0%}"),
        "",
        "| 기록일 | 이벤트일 | 종목코드 | 예측 | 실현 초과수익 |",
        "|---|---|---|---:|---:|",
    ]
    ordered = done.sort_values(["event_date", "prediction"], ascending=[True, False])
    lines += [
        f"| {row['logged_at']} | {row['event_date']} | {row['code']} | "
        f"{row['prediction']:+.4f} | {row['realized_excess']:+.2%} |"
        for _, row in ordered.iterrows()
    ]
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="①층(수급) 전진 기록 — 오늘 후보 기록·채점")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2024-01-01",
                        help="유니버스 일관성 때문에 2024-01 이후만 쓴다 (universe-shift.md)")
    parser.add_argument("--asof", default=None, help="기준일. 생략 시 오늘")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--target-concurrent", type=int, default=20,
                        help="목표 동시 보유 종목 수. 스펙 §6 의 15~25 중앙값이 20")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="data/forward_log.jsonl")
    parser.add_argument("--score", action="store_true",
                        help="기록하지 않고, 라벨이 붙은 과거 기록만 채점한다")
    args = parser.parse_args(argv)

    if args.target_concurrent < 15:
        parser.error(
            f"목표 동시 보유가 {args.target_concurrent} 종목뿐이다. "
            "스펙 §2 는 15 이상을 요구한다 — 집중은 이미 죽은 영역이다."
        )

    end = args.asof or str(pd.Timestamp.today().date())
    conn = connect(args.db)
    prices = load_prices(conn, args.start, end)
    supply = load_supply_demand(conn, args.start, end)

    common = {
        "adv_floor": args.adv_floor,
        "tv_multiple": args.tv_multiple,
        "hold_end": args.hold_end,
    }
    train = build_layered_dataset(prices, supply, require_label=True, **common)
    if train.empty:
        print("학습 표본이 비었다. --start 와 DB 커버리지를 확인하라.", file=sys.stderr)
        return 1

    if args.score:
        return _score(train, args.log_path)

    master = load_stock_master(conn)
    names = dict(zip(master["code"], master["name"]))

    # 임계는 폴드별 계산의 최신 폴드 값을 쓴다. 전역 풀링 금지 (모듈 docstring).
    try:
        pred = walk_forward_predict(
            train, n_folds=args.folds, min_train=args.min_train, seed=args.seed
        )
    except ValueError as exc:
        print(f"임계를 정할 만큼의 표본이 없다: {exc}", file=sys.stderr)
        return 1
    threshold = concurrency_threshold(
        pred, hold_days=args.hold_end, target_concurrent=args.target_concurrent
    )["latest_threshold"]
    if not np.isfinite(threshold):
        print("임계를 정하지 못했다 (폴드 예측이 비었다).", file=sys.stderr)
        return 1

    model, _ = fit_final_model(train, FEATURE_COLUMNS, seed=args.seed)
    pool = build_layered_dataset(prices, supply, require_label=False, **common)
    asof = pd.Timestamp(end)
    today = pool[pool["date"] == asof].copy()

    print(f"# {asof.date()} ①층 수급 후보")
    print()
    print(
        f"학습 표본 {len(train):,}건 · 오늘 풀 {len(today)}종목 · "
        f"임계 {threshold:+.4f} (최신 폴드, 목표 동시보유 {args.target_concurrent})"
    )
    print()

    if today.empty:
        print(f"오늘은 후보 없음 — {asof.date()} 에 거래대금 급증 이벤트가 없다.")
        return 0

    today["prediction"] = model.predict(today[FEATURE_COLUMNS])
    candidates = (
        today[today["prediction"] >= threshold]
        .sort_values("prediction", ascending=False)
        .reset_index(drop=True)
    )
    if candidates.empty:
        best = today["prediction"].max()
        print(
            f"오늘은 후보 없음 — 풀 {len(today)}종목 최고 예측 {best:+.4f} 이 임계 미달이다. "
            "임계 미달인 날 0 건은 정상이다 (스펙 §6)."
        )
        return 0

    print(_table(candidates, names, threshold))
    print()

    entry, exit_date = _plan_dates(prices, asof, args.hold_end)
    written = append_forward_log(
        args.log_path,
        build_records(
            candidates,
            feature_columns=FEATURE_COLUMNS,
            layer=LAYER,
            gate_verdict=GATE_VERDICT,
            logged_at=str(pd.Timestamp.today().date()),
            entry_date=str(pd.Timestamp(entry).date()),
            exit_date=str(pd.Timestamp(exit_date).date()),
            seed=args.seed,
        ),
    )
    print(
        f"진입 {pd.Timestamp(entry).date()} 시가 · 목표 청산 {pd.Timestamp(exit_date).date()} 종가 · "
        f"전진 기록 {written} 건 추가 ({args.log_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
