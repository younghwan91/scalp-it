"""같은 사건을 다룬 중복 기사 클러스터링.

통신사 기사를 여러 매체가 받아쓰면 한 사건이 수십 건으로 부풀어 기사 건수
급증 배수가 왜곡된다. 그래서 사건 단위로 묶어 센다.

행을 지우지는 않는다. 보도 매체 수가 핫함의 독립적인 한 축이기 때문이다
(스펙 §4) — 삭제하면 그 정보가 사라진다.
"""

from __future__ import annotations

import re

import pandas as pd

_BRACKET_RE = re.compile(r"[\[\(【][^\]\)】]*[\]\)】]")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

_EXTRA_COLUMNS = ["cluster_id", "is_primary"]


def normalize_title(title: str) -> str:
    """괄호 안 머리표를 떼고 토큰만 남긴다."""
    text = _BRACKET_RE.sub(" ", str(title))
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.lower().split())


def title_tokens(title: str) -> set[str]:
    """2자 이상 토큰 집합. 1자 토큰은 조사·수치 파편이라 변별력이 없다."""
    return {token for token in normalize_title(title).split() if len(token) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    """토큰 집합의 자카드 유사도."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_duplicates(mapped: pd.DataFrame, *, threshold: float = 0.6) -> pd.DataFrame:
    """``(date, code)`` 안에서 제목이 유사한 기사를 한 클러스터로 묶는다.

    ``news_id`` 순으로 훑으며 기존 클러스터의 **대표 기사** 제목과 비교하는
    탐욕적 방식이다. 그룹 크기가 하루 수십 건이라 O(n²) 로 충분하고,
    입력 순서가 ``news_id`` 로 고정돼 결과가 재현된다.
    """
    if mapped.empty:
        out = mapped.copy()
        out["cluster_id"] = pd.Series(dtype="int64")
        out["is_primary"] = pd.Series(dtype="bool")
        return out

    blocks = []
    for _, group in mapped.groupby(["date", "code"], sort=False):
        group = group.sort_values("news_id")
        representatives: list[set[str]] = []
        assigned: list[int] = []
        for title in group["title"]:
            tokens = title_tokens(title)
            match = None
            for cluster_id, rep_tokens in enumerate(representatives):
                if jaccard(tokens, rep_tokens) >= threshold:
                    match = cluster_id
                    break
            if match is None:
                match = len(representatives)
                representatives.append(tokens)
            assigned.append(match)
        block = group.copy()
        block["cluster_id"] = assigned
        block["is_primary"] = ~block.duplicated(subset="cluster_id", keep="first")
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["date", "code", "news_id"]).reset_index(drop=True)
