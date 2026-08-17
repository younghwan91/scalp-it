"""`scalp-dart-layers` — ①수급 vs ②+공시 판정.

FAIL 이 나와도 exit 0 이다. 부정 결과는 오류가 아니라 결과다 (스펙 §11).
파라미터를 바꿔 재시도하는 경로는 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.db import connect, load_prices, load_supply_demand
from .dart.dataset import build_dart_dataset
from .dart.events import build_disclosure_events
from .dart.store import init_dart_db, load_disclosures
from .dataset import FEATURE_COLUMNS
from .features.disclosure import DISC_FEATURE_COLUMNS
from .model.layered import fold_dates, walk_forward_predict_features
from .report.threshold import concurrency_threshold
from .validate.layer_verdict import layer_verdict
from .validate.monotonic import decile_table, monotonicity

_LAYERS = {
    "supply": FEATURE_COLUMNS,
    "supply_disc": [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="공시 층 판정 (①수급 vs ②+공시)")
    parser.add_argument("--db", default=None, help="가격 DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--dart-db", default="data/dart.db")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--target-concurrent", type=int, default=20)
    parser.add_argument("--out", default="results/dart_layers.json")
    args = parser.parse_args(argv)

    if not Path(args.dart_db).is_file():
        print(f"\n공시 DB 가 없다: {args.dart_db}\n  → 먼저 `scalp-dart` 를 돌려야 한다.\n",
              file=sys.stderr)
        return 1

    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    disc = load_disclosures(
        init_dart_db(args.dart_db),
        args.start.replace("-", ""),
        args.end.replace("-", ""),
    )
    print(f"가격 {len(prices):,} · 수급 {len(supply):,} · 공시 {len(disc):,}", file=sys.stderr)

    # 라벨이 붙기 전 이벤트 수를 따로 센다. 둘을 구분하지 않으면 "라벨 구간이
    # 데이터 끝을 넘어갔다"를 "공시가 없다"로 오진한다.
    events = build_disclosure_events(disc, prices, adv_floor=args.adv_floor)
    ds = build_dart_dataset(prices, supply, disc,
                            adv_floor=args.adv_floor, hold_end=args.hold_end)
    print(f"이벤트 {len(events):,} → 라벨 완비 {len(ds):,}", file=sys.stderr)

    if ds.empty:
        if events.empty:
            print("\n공시 이벤트가 하나도 없다. 수집 범위나 유동성 하한을 확인하라.",
                  file=sys.stderr)
        else:
            last_event = events["date"].max().date()
            last_price = prices["date"].max().date()
            print(
                f"\n이벤트는 {len(events):,} 건 있으나 라벨이 하나도 안 만들어졌다.\n"
                f"  마지막 이벤트 {last_event} · 마지막 가격 {last_price}\n"
                f"  D+{args.hold_end} 청산 시점이 가격 데이터 끝을 넘어간다.\n"
                f"  → 공시 수집 시작일을 앞당기거나(`scalp-dart --start`), "
                f"--end 를 최소 {args.hold_end} 거래일 이전으로 당겨라.",
                file=sys.stderr,
            )
        return 1

    folds = fold_dates(ds, n_folds=args.folds)
    report: dict = {
        "n_events": len(ds),
        "n_codes": int(ds["code"].nunique()),
        "date_range": [str(ds["date"].min().date()), str(ds["date"].max().date())],
        "config": {"adv_floor": args.adv_floor, "hold_end": args.hold_end,
                   "top_n": args.top_n, "folds": args.folds,
                   "target_concurrent": args.target_concurrent},
        "layers": {},
    }

    for name, features in _LAYERS.items():
        pred = walk_forward_predict_features(ds, features, folds=folds)
        verdict = layer_verdict(pred, top_n=args.top_n)
        verdict["monotonicity"] = monotonicity(pred)
        verdict["deciles"] = decile_table(pred).round(6).to_dict("records")
        verdict["threshold"] = concurrency_threshold(
            pred, hold_days=args.hold_end, target_concurrent=args.target_concurrent
        )
        verdict["n_features"] = len(features)
        report["layers"][name] = verdict
        print(f"  {name}: {verdict['overall_verdict']} · "
              f"단조 {verdict['monotonicity']['is_monotone']}", file=sys.stderr)

    base = report["layers"]["supply"]
    disc_layer = report["layers"]["supply_disc"]
    report["disclosure_adds_value"] = bool(
        disc_layer["gate"]["strategy_mean"] > base["gate"]["strategy_mean"]
        and disc_layer["monotonicity"]["is_monotone"]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["layers"]["supply_disc"]["checks"], indent=2, ensure_ascii=False))
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
