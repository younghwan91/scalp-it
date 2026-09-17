"""전진 판정 — 연구 모듈마다 Scaler 위치가 다르다. 이름으로 분기하면 새 연구에서 조용히 터진다.

2026-09-17 실측: rl92 크론 첫 실행에서 `module 'rl_92_mod' has no attribute 'Scaler'` 로 죽었다.
크론이 한 번도 안 돌아 그때까지 드러나지 않았다.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    spec = importlib.util.spec_from_file_location("forward_gate", ROOT / "scripts" / "forward_gate.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["forward_gate"] = m
    spec.loader.exec_module(m)
    return m


def test_every_registered_study_has_a_doc_and_model_path():
    g = _gate()
    for name, spec in g.STUDIES.items():
        assert spec["doc"], name
        assert spec["model"], name
        assert (ROOT / spec["doc"]).exists(), f"{name} 사전등록 문서가 없다: {spec['doc']}"


def test_scaler_lookup_falls_back_instead_of_branching_on_name():
    """모듈에 Scaler 가 없으면 85 것으로 떨어져야 한다 — 이름 목록으로 분기하면 안 된다."""
    src = (ROOT / "scripts" / "forward_gate.py").read_text(encoding="utf-8")
    assert 'getattr(R, "Scaler", None)' in src
    assert 'study in ("rl87", "rl96")' not in src, "이름 분기가 남아 있다"
