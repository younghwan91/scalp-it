"""``scripts/_envload.py`` — 운영 체크아웃과 깊은 워크트리 둘 다에서 ``quant-airflow/.env`` 를 찾는다."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _find_env():
    spec = importlib.util.spec_from_file_location("_envload_t", ROOT / "scripts" / "_envload.py")
    mod = importlib.util.module_from_spec(spec)
    saved = dict(os.environ)
    try:
        spec.loader.exec_module(mod)  # 이 체크아웃에서도 import 가 돼야 한다
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return mod.find_env


def test_finds_env_beside_the_production_checkout(tmp_path):
    env = tmp_path / "git" / "quant-airflow" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("X=1\n")
    scripts = tmp_path / "git" / "scalp-it" / "scripts"
    scripts.mkdir(parents=True)
    assert _find_env()(scripts) == env


def test_finds_env_from_a_deep_worktree(tmp_path):
    env = tmp_path / "git" / "quant-airflow" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("X=1\n")
    scripts = tmp_path / "git" / ".harness-wt" / "scalp-it" / "some-wt" / "scripts"
    scripts.mkdir(parents=True)
    assert _find_env()(scripts) == env


def test_missing_env_is_loud(tmp_path):
    scripts = tmp_path / "nowhere" / "scalp-it" / "scripts"
    scripts.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        _find_env()(scripts)
