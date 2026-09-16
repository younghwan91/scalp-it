"""``scalp-pair-detect --live`` 단일 인스턴스 잠금(krx-quant-core ``InstanceLock``).

같은 계좌로 감지기 두 개가 돌면 같은 신호에 주문이 두 번 나간다. 두 번째 프로세스는
감지기·실행기(인증·계좌조회 포함)를 만들기 **전에** 잠금에서 멈춰야 한다.

런처(``scripts/pair_detect_daily.sh``)는 종료코드와 무관하게 5초 뒤 재시작하므로, 잠금을
못 잡은 쪽은 ``ALREADY_RUNNING_WAIT_S`` 만큼 기다렸다가 0 으로 끝난다 — 몇 초마다 uv 를
다시 띄우는 헛바퀴를 막고, 앞 프로세스가 죽으면 다음 회차에 이어받는다.
"""

from __future__ import annotations

import pytest
from krx_quant_core.execution.oms import AlreadyRunning, InstanceLock

from scalp_it import cli_pair_detect as cli

ARGS = ["--live", "--pairs", "000001:000002", "--no-universe"]


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    p = tmp_path / "data" / "pair_detect.lock"
    monkeypatch.setattr(cli, "PAIR_DETECT_LOCK", p)
    return p


def test_default_lock_path_and_wait():
    assert str(cli.PAIR_DETECT_LOCK) == "data/pair_detect.lock"
    assert cli.ALREADY_RUNNING_WAIT_S >= 30


def test_second_instance_exits_zero_before_building_anything(lock_path, monkeypatch, capsys):
    slept: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", slept.append)

    def must_not_run(*a, **k):
        raise AssertionError("잠금을 못 잡았는데 감지기를 조립했다")

    monkeypatch.setattr(cli, "resolve_pairs", must_not_run)
    monkeypatch.setattr(cli, "run_live", must_not_run)
    with InstanceLock(lock_path):
        assert cli.main(ARGS) == 0
    assert slept == [cli.ALREADY_RUNNING_WAIT_S]
    assert "이미 실행 중" in capsys.readouterr().err


def test_lock_held_for_whole_live_run_and_released(lock_path, monkeypatch):
    seen: list[bool] = []

    def fake_run_live(args, detector, guard, executor=None):
        try:
            InstanceLock(lock_path).acquire()
        except AlreadyRunning:
            seen.append(True)
        return 7

    monkeypatch.setattr(cli, "run_live", fake_run_live)
    assert cli.main(ARGS) == 7
    assert seen == [True]
    InstanceLock(lock_path).acquire().release()        # 끝나면 풀려 있다


def test_lock_released_when_run_live_raises(lock_path, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("socket died")

    monkeypatch.setattr(cli, "run_live", crash)
    with pytest.raises(RuntimeError):
        cli.main(ARGS)
    InstanceLock(lock_path).acquire().release()


def test_replay_mode_takes_no_lock(lock_path, monkeypatch):
    monkeypatch.setattr(cli, "run_replay", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "connect", lambda *a, **k: object())
    with InstanceLock(lock_path):
        assert cli.main(["--replay", "2026-08-24", "--pairs", "000001:000002",
                         "--no-universe"]) == 0


def test_old_core_without_instance_lock_still_runs(lock_path, monkeypatch, capsys):
    """코어가 0.5.0 미만(oms 없음)이어도 --live 기동은 죽지 않는다 — 잠금만 빠지고 경고."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "krx_quant_core.execution.oms":
            raise ImportError("No module named 'krx_quant_core.execution.oms'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: 5)
    assert cli.main(ARGS) == 5
    assert "잠금 없이" in capsys.readouterr().err


def test_executor_module_does_not_need_new_core_at_import():
    """order_executor 는 실시간 프로세스가 import 한다 — 0.5.0 전용 모듈을 최상위에서 부르면
    코어 핀이 늦게 올라간 날 감지기 기동 자체가 죽는다(장부는 부가 기능)."""
    from pathlib import Path

    src = Path(cli.__file__).with_name("realtime").joinpath("order_executor.py").read_text(
        encoding="utf-8")
    top = src.split("\nclass ", 1)[0]
    assert "krx_quant_core.execution" not in top.replace("    from krx_quant_core.execution", "")


def test_lock_os_error_falls_back_to_running_without_lock(lock_path, monkeypatch, capsys):
    """잠금 파일 자체를 못 만들면(권한·경로가 디렉터리 등) 매 재기동마다 죽지 말고 잠금 없이 돈다."""

    def broken_acquire(self):
        raise PermissionError("data/pair_detect.lock: Permission denied")

    monkeypatch.setattr(InstanceLock, "acquire", broken_acquire)
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: 3)
    assert cli.main(ARGS) == 3
    assert "잠금 없이" in capsys.readouterr().err
