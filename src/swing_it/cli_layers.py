"""`swing-layers` — 세 층을 동일 조건에서 비교하고 판정을 출력한다.

**FAIL 이어도 종료 코드는 0 이다.** FAIL 은 오류가 아니라 결과다(스펙 §10).
종료 코드로 FAIL 을 표현하면 래퍼나 CI 가 "실패했으니 파라미터 바꿔 재시도"
경로를 만들게 되고, 그게 정확히 이 프로젝트가 두 번 당한 실패 방식이다.
0 이 아닌 코드는 실행 자체가 불가능할 때만 낸다.

``--dry-run`` 은 성과를 계산하지 않고 표본 수와 하루 평균 풀 크기만 센다.
``tv_multiple`` 을 그 숫자만 보고 확정하기 위한 사전등록 절차다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from .adapters.db import connect, load_prices, load_supply_demand
from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
from .features.news import NEWS_FEATURE_COLUMNS, load_news_features
from .layers import build_layered_dataset, layer_features
from .report.layers_md import render_layer_report
from .validate.compare import compare_layers


def _load_llm(args) -> pd.DataFrame:
    """계획 3 의 (종목, 일자) 접기 결과를 읽고 계획 4 가 안 쓰는 두 컬럼을 뗀다.

    ``n_articles`` 는 ②층의 ``article_count`` 와 같은 축이라 중복이고,
    ``material_type`` 은 원-핫 9개가 스펙 §6 상한을 혼자 먹으므로 모델에
    넣지 않는다. ``material_type`` 은 ``swing-today`` 가 카드 표시용으로만 쓴다.
    """
    frame = load_llm_features(
        args.llm_cache,
        news_path=args.news_db,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    return frame[["code", "date", *LLM_FEATURE_COLUMNS]]


def _build(args, llm, tv_multiple: float):
    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    news = load_news_features(args.news_db, start=args.start, end=args.end)
    print(
        f"가격 {len(prices):,} · 수급 {len(supply):,} · 뉴스피처 {len(news):,} · "
        f"LLM {len(llm):,}",
        file=sys.stderr,
    )
    return build_layered_dataset(
        prices,
        supply,
        news=news,
        news_columns=NEWS_FEATURE_COLUMNS,
        llm=llm,
        llm_columns=LLM_FEATURE_COLUMNS,
        adv_floor=args.adv_floor,
        tv_multiple=tv_multiple,
        hold_end=args.hold_end,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="층별 기여도 분리 판정")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--news-db", default="data/news.db")
    parser.add_argument("--llm-cache", default="data/llm_features.db")
    parser.add_argument("--model", default=None,
                        help="생략 시 캐시에서 성공 건수가 가장 많은 모델")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--list-models", action="store_true",
                        help="캐시의 (model, prompt_version, n_ok) 을 출력하고 끝낸다")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--cost-one-way", type=float, default=0.0023)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="표본 수만 세고 끝낸다 (사전등록용)")
    parser.add_argument("--baseline-json", default="results/baseline.json")
    parser.add_argument("--prereg-json", default="results/preregistration.json")
    parser.add_argument("--out-json", default="results/layers.json")
    parser.add_argument("--out-md", default="results/layers.md")
    args = parser.parse_args(argv)

    # 스펙 §2 의 "동시 보유 15종목 이상"은 **동시 보유** 요건이지 날짜당 종목 수가
    # 아니다. 보유가 hold_end 거래일이므로 동시 보유 ≈ top_n × hold_end 다.
    # 이것을 top_n >= 15 로 구현하면 하루 후보(중앙값 8건)보다 많이 골라
    # selection_ratio 가 1.0 이 되고, 전략이 정의상 랜덤 널과 같아진다.
    # 첫 베이스라인 측정이 정확히 그렇게 망가졌다.
    # → docs/research/2026-08-17-topn-preregistration.md
    concurrent = args.top_n * args.hold_end
    if concurrent < 15:
        parser.error(
            f"동시 보유가 {concurrent} 종목뿐이다 (top_n {args.top_n} × 보유 {args.hold_end}일). "
            "스펙 §2 는 15 이상을 요구한다 — 집중은 이미 죽은 영역이다."
        )
    if args.top_n < 1:
        parser.error("top_n 은 1 이상이어야 한다.")
    if args.tv_multiple > 3.0:
        parser.error(
            "tv_multiple 을 3.0 초과로 올리지 마라. 풀을 좁히는 것은 뉴스 축의 역할이다 "
            "(docs/research/2026-08-17-baseline-verdict.md 시사점 1)."
        )

    if args.list_models:
        from .features.llm import available_models

        conn = sqlite3.connect(f"file:{Path(args.llm_cache)}?mode=ro", uri=True)
        try:
            for model, prompt_version, n_ok in available_models(conn):
                print(f"{model}\t{prompt_version}\t{n_ok:,}")
        finally:
            conn.close()
        return 0

    llm = _load_llm(args)

    if args.dry_run:
        for candidate in (3.0, 1.0):
            ds = _build(args, llm, candidate)
            pool = ds.groupby("date").size().mean() if not ds.empty else 0.0
            print(
                f"tv_multiple={candidate}: 이벤트 {len(ds):,}건 · "
                f"종목 {ds['code'].nunique() if not ds.empty else 0:,} · "
                f"하루 평균 풀 {pool:.1f}건"
            )
        print(
            "\n하루 평균 풀 ≥ 40 이고 총 이벤트 ≥ 2,000 이면 3.0, 아니면 1.0 을 골라 "
            f"{args.prereg_json} 에 기록하고 --tv-multiple 로 넘겨라.",
            file=sys.stderr,
        )
        return 0

    ds = _build(args, llm, args.tv_multiple)
    if ds.empty:
        print("세 층 피처가 모두 있는 이벤트가 없다. 계획 2·3 산출물을 확인하라.",
              file=sys.stderr)
        return 1
    print(f"비교 표본 {len(ds):,} 건 · 종목 {ds['code'].nunique():,}", file=sys.stderr)

    spec = layer_features(NEWS_FEATURE_COLUMNS, LLM_FEATURE_COLUMNS)
    comparison = compare_layers(
        ds,
        spec,
        top_n=args.top_n,
        n_folds=args.folds,
        n_trials=args.trials,
        n_boot=args.boot,
        cost_one_way=args.cost_one_way,
        hold_days=args.hold_end,
        seed=args.seed,
    )

    baseline_reference = None
    if Path(args.baseline_json).exists():
        raw = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        baseline_reference = {
            "strategy_mean": raw["strategy_mean"],
            "verdict": raw.get("overall_verdict", raw["verdict"]),
            "n_events": raw.get("n_events", 0),
        }
    preregistration = None
    if Path(args.prereg_json).exists():
        preregistration = json.loads(Path(args.prereg_json).read_text(encoding="utf-8"))

    payload = {
        **comparison,
        "baseline_reference": baseline_reference,
        "preregistration": preregistration,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    markdown = render_layer_report(
        comparison,
        baseline_reference=baseline_reference,
        preregistration=preregistration,
    )
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
