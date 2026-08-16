"""`swing-today` — 오늘 후보의 근거 카드를 내고 전진 기록에 남긴다 (스펙 §1·§9).

층은 ``--layer`` 로 고른다. 기본은 ③이지만, `swing-layers` 판정이
``LLM_NOT_JUSTIFIED`` 였다면 ②를 쓰는 것이 옳다. 그 선택은 사람이 한다 —
판정에 따라 자동으로 층을 바꾸면 리포트를 안 읽게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .adapters.db import connect, load_prices, load_stock_master, load_supply_demand
from .adapters.news_db import connect_news, load_article_cards
from .cli_layers import MissingPrerequisite, require_sqlite
from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
from .features.news import NEWS_FEATURE_COLUMNS, load_news_features
from .layers import LAYER_LABELS, build_layered_dataset, layer_features
from .model.layered import fit_final_model
from .report.cards import render_candidates
from .report.forward_log import append_forward_log, build_records


def _material_types(llm_full: pd.DataFrame, asof: pd.Timestamp) -> dict[str, str]:
    """기준일의 종목별 재료 유형. 카드 표시 전용이다 (모델 행렬에 없다)."""
    same_day = llm_full[llm_full["date"] == asof]
    return dict(zip(same_day["code"].astype(str), same_day["material_type"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="오늘 후보 + 근거 카드 + 전진 기록")
    parser.add_argument("--db", default=None)
    parser.add_argument("--news-db", default="data/news.db")
    parser.add_argument("--llm-cache", default="data/llm_features.db")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--asof", default=None, help="기준일. 생략 시 데이터 마지막 날")
    parser.add_argument("--layer", default="supply_news_llm", choices=list(LAYER_LABELS))
    parser.add_argument("--layers-json", default="results/layers.json")
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="data/forward_log.jsonl")
    parser.add_argument("--out-md", default=None, help="생략 시 results/today/<기준일>.md")
    args = parser.parse_args(argv)

    # 스펙 §2 는 **동시 보유** 15종목 이상을 요구한다. 날짜당 종목 수가 아니다.
    # 보유가 hold_end 거래일이므로 동시 보유 ≈ top_n × hold_end.
    # → docs/research/2026-08-17-topn-preregistration.md
    concurrent = args.top_n * args.hold_end
    if concurrent < 15:
        parser.error(
            f"동시 보유가 {concurrent} 종목뿐이다 (top_n {args.top_n} × 보유 {args.hold_end}일). "
            "스펙 §2 는 15 이상을 요구한다."
        )
    if args.top_n < 1:
        parser.error("top_n 은 1 이상이어야 한다.")

    # 선행 산출물을 **가격을 읽기 전에** 확인한다. 570만 행을 로드한 뒤에
    # "파일이 없다"를 만나면 몇 분을 버린다.
    try:
        require_sqlite(args.news_db, what="뉴스 DB", produced_by="swing-news")
        require_sqlite(args.llm_cache, what="LLM 피처 캐시", produced_by="swing-score")
    except MissingPrerequisite as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    conn = connect(args.db)
    end = args.asof or str(pd.Timestamp.today().date())
    prices = load_prices(conn, args.start, end)
    supply = load_supply_demand(conn, args.start, end)
    master = load_stock_master(conn)
    names = dict(zip(master["code"], master["name"]))

    # 근거 카드용 대표 기사. ops_only=False — 후보는 학습 유니버스에서 나오므로
    # 운영 상위 10 밖의 종목도 카드를 받아야 한다.
    news_conn = connect_news(args.news_db)
    try:
        articles = load_article_cards(news_conn, args.start, end, ops_only=False)
    finally:
        news_conn.close()

    # material_type 은 모델에 넣지 않고 카드 표시용으로만 들고 간다 (스펙 §6).
    llm_full = load_llm_features(
        args.llm_cache,
        news_path=args.news_db,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    common = {
        "news": load_news_features(args.news_db, start=args.start, end=end),
        "news_columns": NEWS_FEATURE_COLUMNS,
        "llm": llm_full[["code", "date", *LLM_FEATURE_COLUMNS]],
        "llm_columns": LLM_FEATURE_COLUMNS,
        "adv_floor": args.adv_floor,
        "tv_multiple": args.tv_multiple,
        "hold_end": args.hold_end,
    }
    train = build_layered_dataset(prices, supply, require_label=True, **common)
    pool = build_layered_dataset(prices, supply, require_label=False, **common)
    if train.empty or pool.empty:
        print("학습 표본 또는 후보 풀이 비었다.", file=sys.stderr)
        return 1

    asof = pd.Timestamp(args.asof) if args.asof else pool["date"].max()
    today = pool[pool["date"] == asof].copy()
    if today.empty:
        print(f"{asof.date()} 에 이벤트가 없다.", file=sys.stderr)
        return 1

    columns = layer_features(NEWS_FEATURE_COLUMNS, LLM_FEATURE_COLUMNS)[args.layer]
    model, baseline = fit_final_model(train, columns, seed=args.seed)
    today["prediction"] = model.predict(today[columns])
    candidates = today.nlargest(args.top_n, "prediction").reset_index(drop=True)

    verdict = "UNKNOWN"
    if Path(args.layers_json).exists():
        payload = json.loads(Path(args.layers_json).read_text(encoding="utf-8"))
        for row in payload["rows"]:
            if row["layer"] == args.layer:
                verdict = row["overall_verdict"]
    else:
        print("results/layers.json 이 없다. 먼저 swing-layers 를 돌려라.", file=sys.stderr)

    # 진입은 기준일 다음 거래일 시가, 목표 청산은 그로부터 hold_end 거래일 뒤.
    calendar = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
    idx = int(calendar[calendar == asof].index[0])
    entry = calendar.iloc[min(idx + 1, len(calendar) - 1)]
    exit_idx = idx + 1 + args.hold_end
    exit_date = (
        calendar.iloc[exit_idx]
        if exit_idx < len(calendar)
        # 달력에 아직 없는 미래는 영업일 근사로 표시한다
        else entry + pd.tseries.offsets.BDay(args.hold_end)
    )

    markdown = "\n".join(
        [
            f"# {asof.date()} 후보 — {LAYER_LABELS[args.layer]}",
            "",
            (f"종합 판정 **{verdict}** · 상위 {len(candidates)}종목 / 풀 {len(today)}종목 · "
            f"학습 표본 {len(train):,}건"),
            "",
            render_candidates(
                candidates,
                model=model,
                baseline=baseline,
                train=train,
                feature_columns=columns,
                articles=articles,
                gate_verdict=verdict,
                layer_label=LAYER_LABELS[args.layer],
                entry_date=entry,
                exit_date=exit_date,
                names=names,
                material_types=_material_types(llm_full, asof),
            ),
        ]
    )

    out_md = Path(args.out_md or f"results/today/{asof.date()}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")

    written = append_forward_log(
        args.log_path,
        build_records(
            candidates,
            feature_columns=columns,
            layer=args.layer,
            gate_verdict=verdict,
            logged_at=str(pd.Timestamp.today().date()),
            entry_date=str(pd.Timestamp(entry).date()),
            exit_date=str(pd.Timestamp(exit_date).date()),
            seed=args.seed,
        ),
    )
    print(f"{out_md} 작성 · 전진 기록 {written} 건 추가", file=sys.stderr)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
