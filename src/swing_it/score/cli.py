"""`swing-score` — 뉴스 테이블을 사전 필터에 걸고 LLM 배치로 태운다.

``--provider fake`` 로 두면 네트워크 없이 배선 전체를 확인할 수 있다.
첫 실 호출은 반드시 ``--limit`` 을 작게 잡고 시작한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from ..adapters.db import connect, load_prices
from ..adapters.news_db import connect_news, load_articles
from ..events import extract_events
from .batch import score_articles
from .cache import open_cache
from .prefilter import prefilter_articles, prefilter_summary
from .prompt import PROMPT_VERSION
from .provider import provider_from_env


def price_lookback_start(start: str, event_window: int) -> str:
    """이벤트 판정용 가격을 얼마나 앞당겨 읽을지 정한다.

    ``extract_events`` 의 ``adv`` 는 직전 ``event_window`` **거래일** 평균이고
    ``min_periods`` 가 창 크기와 같아 창이 안 차면 전부 ``NaN`` 이 된다.
    달력일은 주말·공휴일 때문에 거래일보다 성기므로 두 배 + 14일을 앞당긴다
    (설날·추석 연휴를 낀 최악의 경우도 덮는다).
    """
    return (
        pd.Timestamp(start) - pd.Timedelta(days=event_window * 2 + 14)
    ).strftime("%Y-%m-%d")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="뉴스 기사 → LLM 고정 스키마 피처")
    parser.add_argument(
        "--news-db", default=None, help="생략 시 SWING_NEWS_DB 또는 data/news.db"
    )
    parser.add_argument("--cache-db", default="data/llm_features.db")
    parser.add_argument("--db", default=None, help="가격 DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument(
        "--provider", default=None, help="fake / gemini / openai / anthropic"
    )
    parser.add_argument(
        "--model", default=None, help="생략 시 SWING_LLM_MODEL 또는 기본 모델"
    )
    parser.add_argument("--limit", type=int, default=None, help="API 호출 상한")
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--event-window", type=int, default=20)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--out", default="results/score_run.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    env = dict(os.environ)
    if args.provider:
        env["SWING_LLM_PROVIDER"] = args.provider
    if args.model:
        env["SWING_LLM_MODEL"] = args.model
    provider = provider_from_env(env)

    articles = load_articles(connect_news(args.news_db), args.start, args.end)
    # 이벤트 판정은 직전 event_window 거래일 평균이 필요하다. start 부터만 읽으면
    # 롤링 창이 끝까지 안 차서 이벤트가 0 이 되고, 모든 기사가 no_volume_event 로
    # 조용히 탈락한다. 휴장일을 감안해 넉넉히 앞당겨 읽는다.
    price_start = price_lookback_start(args.start, args.event_window)
    prices = load_prices(connect(args.db), price_start, args.end)
    events = extract_events(
        prices,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        window=args.event_window,
    )
    print(f"기사 {len(articles):,} 건 · 이벤트 {len(events):,} 건", file=sys.stderr)

    filtered = prefilter_articles(articles, events, min_chars=args.min_chars)
    summary = prefilter_summary(filtered)
    kept = filtered.loc[filtered["keep"]].reset_index(drop=True)
    print(
        f"사전 필터 통과 {summary['n_kept']:,} / {summary['n_input']:,}", file=sys.stderr
    )

    cache = open_cache(args.cache_db)
    result = score_articles(
        kept,
        provider,
        cache,
        max_retries=args.max_retries,
        limit=args.limit,
        on_progress=lambda done, total: (
            print(f"  {done}/{total}", file=sys.stderr) if done % 100 == 0 else None
        ),
    )

    # results/ 는 커밋될 수 있다. 기사 텍스트·article_id 를 넣지 않는다.
    report = {
        "provider": provider.name,
        "model": provider.model,
        "prompt_version": PROMPT_VERSION,
        "date_range": [args.start, args.end],
        "prefilter": summary,
        "batch": {
            "n_input": result.n_input,
            "n_cached": result.n_cached,
            "n_called": result.n_called,
            "n_ok": result.n_ok,
            "n_failed": result.n_failed,
        },
        "failure_reasons": sorted({reason for _, reason in result.failures})[:10],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
