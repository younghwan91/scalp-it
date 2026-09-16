# ruff: noqa: DTZ001 — 합성 틱 시각은 DB 틱과 같은 naive
"""``scripts/pair_sweep.py`` 의 krx-quant-core ``run_sweep`` 경로.

``--legacy-serial``(기록 없는 옛 직렬 루프)과 코어 엔진(``n_jobs=1``)이 **같은 결과
레코드**를 내야 한다. 엔진 경로는 ``start_run`` 을 거치므로 simnode 호스트명과 깨끗한
git 레포가 필요하다 — 코어 테스트처럼 임시 git 레포를 만들고 호스트명을 바꾼다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

import pytest
from krx_quant_core.runtime.oos import RunRefused
from test_pair_sweep import ps


def _fake_run_combo(pairs, cfg, rows, guard_cfg, last_ts):
    # 조합 파라미터가 결과에 반영되는지 보이도록 설정값으로 숫자를 만든다.
    return {"alerts": int(cfg.surge_window) + len(rows), "filtered": int(cfg.cosurge_min * 1000),
            "by_basis": {"through": {"filled": len(pairs)}}, "elapsed_s": 0.0}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("KQC_RUNS_ROOT", raising=False)
    monkeypatch.setenv("KQC_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: "simnode")
    root = tmp_path / "repo"
    root.mkdir()
    g = ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "config", "user.email", "t@example.com"], check=True)
    subprocess.run([*g, "config", "user.name", "t"], check=True)
    (root / "a.py").write_text("x = 1\n")
    subprocess.run([*g, "add", "a.py"], check=True)
    subprocess.run([*g, "commit", "-qm", "init"], check=True)
    return root


@pytest.fixture
def ctx(monkeypatch):
    # 병렬 경로는 objective 를 이름으로 피클한다 — 스크립트로 돌 땐 __main__ 이지만
    # 테스트에선 파일에서 읽은 모듈이라 sys.modules 에 올려 둬야 찾는다.
    monkeypatch.setitem(sys.modules, ps.__name__, ps)
    monkeypatch.setattr(ps, "run_combo", _fake_run_combo)
    rows = [("AAA", datetime(2026, 8, 24, 9, 0, i), 1000.0, 1.0, 999.0, 1) for i in range(3)]
    ps.set_context(pairs=[ps.Pair(leader="AAA", follower="BBB")], rows=rows,
                   guard_cfg=ps.RiskGuardConfig(cost_rate=0.0), last_ts=rows[-1][1],
                   base=ps.DetectorConfig.stable(),
                   axis_names=["surge_window", "cosurge_min"])
    axes, _ = ps.parse_grid(["surge_window=10,15", "cosurge_min=0.005,0"])
    return ps.build_combos(axes)


def test_engine_n_jobs_1_matches_legacy_serial(repo, ctx):
    legacy = ps.run_serial(ctx)
    engine = ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=False)
    assert engine == legacy
    assert [r["params"] for r in engine] == [
        {"surge_window": 10.0, "cosurge_min": 0.005}, {"surge_window": 10.0, "cosurge_min": 0.0},
        {"surge_window": 15.0, "cosurge_min": 0.005}, {"surge_window": 15.0, "cosurge_min": 0.0}]
    # 결과는 JSONL 로 그대로 나갈 수 있어야 한다(원본과 같은 모양)
    json.dumps(engine, ensure_ascii=False, default=str)


def test_engine_records_every_config_in_trial_ledger(repo, ctx):
    ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=False)
    ledger = repo / "research" / "runs" / "pair-sweep-2026-08-24" / "TRIALS.jsonl"
    assert ledger.exists()
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == len(ctx) + 1


def test_engine_parallel_matches_legacy(repo, ctx):
    assert ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=2,
                         cache=False) == ps.run_serial(ctx)


def test_cache_does_not_leak_across_different_ticks_or_base(repo, ctx):
    """캐시 키에 틱·쌍·기준설정이 들어가야 한다 — 같은 날짜·그리드라도 입력이 다르면 재계산."""
    first = ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=True)
    assert ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=True) == first
    ps._CTX["rows"] = ps._CTX["rows"][:1]                      # --limit-ticks 같은 변화
    fewer = ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=True)
    assert fewer == ps.run_serial(ctx) and fewer != first


def test_engine_refuses_off_simnode(repo, ctx, monkeypatch):
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: "trader")
    with pytest.raises(RunRefused):
        ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=False)


def test_objective_failure_is_reported_not_silently_dropped(repo, ctx, monkeypatch):
    def boom(pairs, cfg, rows, guard_cfg, last_ts):
        if cfg.surge_window == 15.0:
            raise ValueError("boom")
        return _fake_run_combo(pairs, cfg, rows, guard_cfg, last_ts)

    monkeypatch.setattr(ps, "run_combo", boom)
    with pytest.raises(SystemExit, match="boom"):
        ps.run_engine(ctx, day="2026-08-24", repo_root=repo, n_jobs=1, cache=False)


def test_cli_flags():
    a = ps.build_parser().parse_args(["--replay", "2026-08-24"])
    assert a.jobs is None and a.no_cache is False and a.legacy_serial is False
    a = ps.build_parser().parse_args(["--replay", "2026-08-24", "--jobs", "4", "--no-cache",
                                      "--legacy-serial"])
    assert (a.jobs, a.no_cache, a.legacy_serial) == (4, True, True)
