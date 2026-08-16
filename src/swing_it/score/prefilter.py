"""LLM 에 태울 기사를 줄이는 규칙 기반 사전 필터 — 순수 함수.

비용을 줄이는 게 목적이지만 그게 전부는 아니다. 종목이 안 붙은 기사는 라벨을
만들 수 없고, 시장이 거래대금으로 반응하지 않은 기사는 스펙 §4 가 말하는 '핫함'
정의를 못 채운다. 어느 쪽도 LLM 을 태울 가치가 없다.

행을 지우지 않고 ``keep``/``drop_reason`` 을 붙여 돌려준다. 어떤 규칙이 몇 건을
잘랐는지 봐야 임계값을 조정할 수 있기 때문이다.
"""

from __future__ import annotations

import re

import pandas as pd

DROP_REASONS: tuple[str, ...] = ("no_code", "too_short", "no_volume_event", "duplicate")

_BRACKET_PREFIX = re.compile(r"^\s*[\[\(<【][^\]\)>】]{0,12}[\]\)>】]\s*")
_NON_WORD = re.compile(r"[^0-9A-Za-z가-힣]+")


def normalize_title(title: str) -> str:
    """중복 판정용 제목 정규화.

    ``[단독]``·``[속보]`` 류 말머리를 걷어내고, 공백·문장부호를 모두 지운다.
    같은 사안을 여러 매체가 받아쓸 때 남는 차이가 대개 이 둘뿐이다.
    """
    text = str(title or "")
    while True:
        stripped = _BRACKET_PREFIX.sub("", text)
        if stripped == text:
            break
        text = stripped
    return _NON_WORD.sub("", text)


def prefilter_articles(
    articles: pd.DataFrame,
    events: pd.DataFrame,
    *,
    min_chars: int = 30,
) -> pd.DataFrame:
    """규칙 순서대로 탈락 사유를 매긴다. 입력 행 순서를 보존한다."""
    out = articles.copy()
    out["_date"] = pd.to_datetime(out["date"])
    out["_title_key"] = out["title"].map(normalize_title)

    code = out["code"].fillna("").astype(str).str.strip()
    text_len = (
        out["title"].fillna("").astype(str).str.len()
        + out["body200"].fillna("").astype(str).str.len()
    )

    event_keys = set(
        zip(
            events["code"].astype(str),
            pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d"),
        )
    )
    has_event = pd.Series(
        [
            (c, d) in event_keys
            for c, d in zip(code, out["_date"].dt.strftime("%Y-%m-%d"))
        ],
        index=out.index,
    )

    reason = pd.Series("", index=out.index, dtype=object)
    reason[code.eq("")] = "no_code"
    reason[reason.eq("") & text_len.lt(min_chars)] = "too_short"
    reason[reason.eq("") & ~has_event] = "no_volume_event"

    # 중복은 살아남은 행들 사이에서만 따진다. 이미 탈락한 행이 첫 건 자리를
    # 차지해 진짜 기사를 duplicate 로 밀어내는 일이 없어야 한다.
    group_cols = ["_date", "code", "_title_key"]
    survivors = out.loc[reason.eq("")]
    if not survivors.empty:
        dup_rank = survivors.groupby(group_cols, sort=False).cumcount()
        reason.loc[dup_rank[dup_rank > 0].index] = "duplicate"

    # dup_count 는 탈락 행 포함 전체 그룹 크기다. 뒷단의 '보도 매체 수' 참고용.
    out["dup_count"] = out.groupby(group_cols, sort=False)["article_id"].transform("size")
    out["drop_reason"] = reason
    out["keep"] = reason.eq("")
    return out.drop(columns=["_date", "_title_key"])


def prefilter_summary(filtered: pd.DataFrame) -> dict[str, int]:
    """필터 결과를 리포트용 카운트로 접는다."""
    counts = filtered["drop_reason"].value_counts()
    summary = {
        "n_input": int(len(filtered)),
        "n_kept": int(filtered["keep"].sum()),
    }
    for reason in DROP_REASONS:
        summary[reason] = int(counts.get(reason, 0))
    return summary
