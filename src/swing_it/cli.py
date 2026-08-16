"""`swing-baseline` — 베이스라인 파이프라인 전체를 실 DB 로 한 번 돌린다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.db import connect, load_prices, load_supply_demand
from .dataset import build_dataset
from .model.baseline import portfolio_returns, walk_forward_predict
from .validate.gate import gate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="수급 단독 베이스라인 판정")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args(argv)

    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    print(f"가격 {len(prices):,} 행 · 수급 {len(supply):,} 행", file=sys.stderr)

    ds = build_dataset(
        prices,
        supply,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        hold_end=args.hold_end,
    )
    print(f"이벤트 {len(ds):,} 건 · 종목 {ds['code'].nunique():,}", file=sys.stderr)
    if ds.empty:
        print("이벤트가 없다. 임계값을 확인하라.", file=sys.stderr)
        return 1

    pred = walk_forward_predict(ds, n_folds=args.folds)
    report = gate_report(pred, top_n=args.top_n)
    report["n_events"] = len(ds)
    report["n_codes"] = int(ds["code"].nunique())
    report["date_range"] = [str(ds["date"].min().date()), str(ds["date"].max().date())]
    report["config"] = {
        "start": args.start,
        "end": args.end,
        "adv_floor": args.adv_floor,
        "tv_multiple": args.tv_multiple,
        "hold_end": args.hold_end,
        "top_n": args.top_n,
        "folds": args.folds,
    }
    report["per_fold"] = {
        int(fold): float(portfolio_returns(group, top_n=args.top_n)["mean_excess"].mean())
        for fold, group in pred.groupby("fold")
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
