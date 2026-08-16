"""근거 카드 — 왜 걸렸는지, 얼마나 들 건지, 과거에 비슷한 게 어땠는지.

기여도는 '피처를 학습셋 중앙값으로 되돌리면 예측이 얼마나 줄어드는가'다.
상호작용 때문에 합이 총 예측과 정확히 맞지는 않는다. 카드에 그렇게 적는다.
정확한 가법 분해가 필요하면 SHAP 이 필요하지만, 하루 10건을 사람이 훑고
'무엇이 이걸 밀어올렸나'를 보는 데는 순위면 충분하다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DISCLAIMER = (
    "이 카드는 근거 제시이지 매매 신호가 아니다. 진입·청산·비중은 사람이 정한다."
)


def feature_contributions(
    model,
    row: pd.Series,
    baseline: pd.Series,
    feature_columns: Sequence[str],
) -> pd.Series:
    """피처를 하나씩 중앙값으로 되돌렸을 때의 예측 감소량. 내림차순."""
    columns = list(feature_columns)
    x = pd.DataFrame([row[columns].astype(float).to_dict()], columns=columns)
    full = float(model.predict(x)[0])

    out = {}
    for col in columns:
        alt = x.copy()
        alt[col] = float(baseline[col])
        out[col] = full - float(model.predict(alt)[0])
    return pd.Series(out).sort_values(ascending=False)


def find_analogs(
    train: pd.DataFrame,
    row: pd.Series,
    feature_columns: Sequence[str],
    *,
    k: int = 3,
) -> pd.DataFrame:
    """표준화 피처 공간에서 가장 가까운 과거 이벤트 ``k`` 건과 그 **실현** 초과수익."""
    columns = list(feature_columns)
    scale = train[columns].std(ddof=0).replace(0.0, 1.0)
    centre = train[columns].mean()
    target = (row[columns].astype(float) - centre) / scale
    scaled = (train[columns] - centre) / scale

    out = train[["code", "date", "excess_return"]].copy()
    out["distance"] = np.sqrt(((scaled - target) ** 2).sum(axis=1)).to_numpy()
    return out.nsmallest(k, "distance").reset_index(drop=True)


def _counts_line(row: pd.Series) -> str:
    """집계 수치는 ②층 **피처**에서 읽는다 — ``load_article_cards`` 가 아니라.

    같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다. ①층처럼 뉴스 피처가
    없는 층에서는 빈 문자열을 돌려주고, 카드는 그 줄만 생략한다.
    """
    parts = []
    if "article_count" in row and pd.notna(row["article_count"]):
        parts.append(f"기사 {int(row['article_count'])}건")
    if "outlet_count" in row and pd.notna(row["outlet_count"]):
        parts.append(f"매체 {int(row['outlet_count'])}곳")
    if "article_ratio" in row and pd.notna(row["article_ratio"]):
        parts.append(f"평소 대비 {row['article_ratio']:.1f}배")
    return " · ".join(parts)


def _format_articles(articles: pd.DataFrame, row: pd.Series) -> str:
    """집계 수치(후보 행) + 대표 기사 1건(``load_article_cards``).

    ``load_article_cards`` 는 (code, date) 당 대표 기사 한 건을 준다
    (``daily_news.primary_news_id`` 조인). 발행 시각은 없다 — 빅카인즈가
    일자 단위만 주기 때문이고, 운영 스키마와 갈라놓지 않기 위해 쓰지 않는다.
    """
    code = str(row["code"])
    date = pd.Timestamp(row["date"])
    counts = _counts_line(row)

    hit = (
        articles[(articles["code"] == code) & (articles["date"] == date)]
        if articles is not None and not articles.empty
        else articles
    )
    if hit is None or hit.empty:
        head = f"**근거 기사 없음**{' — ' + counts if counts else ''}"
        return head + "\n"

    art = hit.iloc[0]
    header = f"**근거 {counts}**" if counts else "**근거 기사**"
    header += f" (핫함 {art['hotness']:.2f} · 그날 {int(art['hotness_rank'])}위)"
    if pd.isna(art.get("article_id")):
        return header + "\n\n- 대표 기사 없음 (클러스터 대표를 못 고름)\n"
    return "\n".join(
        [
            header,
            "",
            f"- [{art['press']}] [{art['title']}]({art['url']})",
            "",
            f"> {art['body200']}",
            "",
        ]
    )


def render_card(
    row: pd.Series,
    *,
    rank: int,
    contributions: pd.Series,
    analogs: pd.DataFrame,
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    names: dict[str, str] | None = None,
    material_types: dict[str, str] | None = None,
    top_features: int = 5,
) -> str:
    """후보 한 건의 근거 카드 마크다운.

    ``material_types`` 는 표시 전용이다. LLM 이 뽑은 재료 유형을 모델 행렬에
    넣으면 원-핫 9개가 스펙 §6 의 피처 상한을 혼자 먹으므로, 사람이 읽는
    문자열로만 딸려 보낸다.
    """
    code = str(row["code"])
    heading = f"{code} {(names or {}).get(code, '')}".strip()
    event_date = pd.Timestamp(row["date"])

    lines = [
        f"### {rank}. {heading} — 예측 초과수익 {row['prediction']:+.2%}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 층 | {layer_label} |",
        f"| 이벤트일 | {event_date.date()} |",
    ]
    material = (material_types or {}).get(code)
    if material:
        lines.append(f"| 재료 유형 | {material} |")
    lines += [
        f"| 진입 | {pd.Timestamp(entry_date).date()} 시가 |",
        f"| 목표 청산 | {pd.Timestamp(exit_date).date()} 종가 (D+40, 중간 청산 없음) |",
        "",
    ]

    top = contributions.head(top_features)
    lines += [
        f"**왜 걸렸나 — 예측 기여도 상위 {len(top)}**",
        "",
        "| 피처 | 값 | 기여 |",
        "|---|---|---|",
    ]
    for feature, value in top.items():
        lines.append(f"| `{feature}` | {float(row[feature]):.3f} | {value:+.4f} |")
    lines += [
        "",
        "> 기여도 = 해당 피처를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량.",
        "> 피처 간 상호작용이 있어 합이 총 예측과 정확히 일치하지는 않는다. 순위로 읽어라.",
        "",
        _format_articles(articles, row),
        f"**유사 과거 사례 (피처 공간 최근접 {len(analogs)}건, 실현 수익)**",
        "",
        "| 종목 | 이벤트일 | 실현 초과수익 | 거리 |",
        "|---|---|---|---|",
    ]
    for _, a in analogs.iterrows():
        lines.append(
            f"| {a['code']} | {pd.Timestamp(a['date']).date()} | "
            f"{a['excess_return']:+.2%} | {a['distance']:.2f} |"
        )

    lines.append("")
    if gate_verdict != "PASS":
        lines += [
            (
                "> **이 층은 음성대조 널을 이기지 못했다.** 아래 후보는 모델이 뽑았다는 "
                "사실 외에 어떤 통계적 근거도 없다."
            ),
            "",
        ]
    lines += [f"**판정: {gate_verdict}** ({layer_label} 층). {DISCLAIMER}", ""]
    return "\n".join(lines)


def render_candidates(
    candidates: pd.DataFrame,
    *,
    model,
    baseline: pd.Series,
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    names: dict[str, str] | None = None,
    material_types: dict[str, str] | None = None,
) -> str:
    """후보 전체를 예측 내림차순으로 렌더한다.

    후보는 모두 같은 이벤트일이므로 진입·청산일이 하나씩만 필요하다.
    """
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    cards = [
        render_card(
            row,
            rank=rank,
            contributions=feature_contributions(model, row, baseline, feature_columns),
            analogs=find_analogs(train, row, feature_columns, k=3),
            articles=articles,
            gate_verdict=gate_verdict,
            layer_label=layer_label,
            entry_date=entry_date,
            exit_date=exit_date,
            names=names,
            material_types=material_types,
        )
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1)
    ]
    return "\n---\n\n".join(cards)
