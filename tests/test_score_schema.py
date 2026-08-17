import pytest

from scalp_it.score.schema import (
    LLM_FEATURE_COLUMNS,
    ArticleFeatures,
    SchemaError,
    encode_features,
    parse_features,
)

_VALID = {
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "large",
}


def test_parse_accepts_dict():
    features = parse_features(dict(_VALID))
    assert features == ArticleFeatures(
        material_type="수주",
        persistence="multi_quarter",
        specificity=3,
        freshness="new",
        sentiment_direction=1,
        sentiment_strength=2,
        scale_vs_size="large",
    )


def test_parse_strips_code_fence_and_prose():
    raw = (
        "물론입니다. 아래가 결과입니다.\n"
        "```json\n"
        '{"material_type":"실적","persistence":"one_off","specificity":1,'
        '"freshness":"repeat","sentiment_direction":-1,"sentiment_strength":1,'
        '"scale_vs_size":"unknown"}\n'
        "```\n"
    )
    features = parse_features(raw)
    assert features.material_type == "실적"
    assert features.sentiment_direction == -1


def test_parse_rejects_unknown_material_type():
    bad = dict(_VALID, material_type="어닝서프라이즈")
    with pytest.raises(SchemaError, match="material_type"):
        parse_features(bad)


def test_parse_rejects_out_of_range_specificity():
    with pytest.raises(SchemaError, match="specificity"):
        parse_features(dict(_VALID, specificity=7))


def test_parse_rejects_missing_field():
    partial = {k: v for k, v in _VALID.items() if k != "freshness"}
    with pytest.raises(SchemaError, match="freshness"):
        parse_features(partial)


def test_parse_rejects_text_without_json():
    with pytest.raises(SchemaError):
        parse_features("죄송합니다. 답변할 수 없습니다.")


def test_parse_coerces_numeric_strings():
    # LLM 이 숫자를 문자열로 내는 일이 흔하다. 값만 맞으면 받아준다.
    features = parse_features(dict(_VALID, specificity="3", sentiment_direction="1"))
    assert features.specificity == 3
    assert features.sentiment_direction == 1


def test_encode_returns_exactly_feature_columns():
    encoded = encode_features(parse_features(dict(_VALID)))
    assert list(encoded.keys()) == LLM_FEATURE_COLUMNS


def test_encode_maps_ordinals_to_numbers():
    encoded = encode_features(parse_features(dict(_VALID)))
    assert encoded["llm_persistence"] == 1.0   # multi_quarter
    assert encoded["llm_specificity"] == 3.0
    assert encoded["llm_freshness"] == 2.0     # new
    assert encoded["llm_scale"] == 3.0         # large


def test_sentiment_is_direction_times_strength():
    positive = encode_features(parse_features(dict(_VALID)))
    negative = encode_features(
        parse_features(dict(_VALID, sentiment_direction=-1, sentiment_strength=3))
    )
    neutral = encode_features(
        parse_features(dict(_VALID, sentiment_direction=0, sentiment_strength=3))
    )
    assert positive["llm_sentiment"] == 2.0
    assert negative["llm_sentiment"] == -3.0
    assert neutral["llm_sentiment"] == 0.0
