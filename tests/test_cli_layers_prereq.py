"""선행 산출물이 없을 때의 동작.

아침에 사용자가 처음 만나는 상태다 — 계획 2·3 을 아직 안 돌려서 news.db 도
llm_features.db 도 없다. 그때 원시 sqlite 오류를 던지면 무엇이 문제인지 알 수 없다.
"""

import pytest

from scalp_it.cli_layers import MissingPrerequisite, require_sqlite


def test_missing_file_raises_with_actionable_message(tmp_path):
    with pytest.raises(MissingPrerequisite) as exc:
        require_sqlite(tmp_path / "nope.db", what="LLM 피처 캐시", produced_by="scalp-score")
    message = str(exc.value)
    assert "LLM 피처 캐시" in message
    assert "scalp-score" in message
    assert "nope.db" in message


def test_existing_file_passes_through(tmp_path):
    path = tmp_path / "there.db"
    path.write_bytes(b"")
    assert require_sqlite(path, what="x", produced_by="y") == path


def test_directory_is_treated_as_missing(tmp_path):
    with pytest.raises(MissingPrerequisite):
        require_sqlite(tmp_path, what="x", produced_by="y")


def test_missing_prerequisite_is_not_a_bare_exception():
    """호출자가 잡아서 exit 1 로 바꿀 수 있어야 한다."""
    assert issubclass(MissingPrerequisite, Exception)
    assert not issubclass(MissingPrerequisite, SystemExit)
