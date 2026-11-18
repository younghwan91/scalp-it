"""``deploy/research_run.sh`` — systemd user bus 가 거부해도 명령은 돈다.

bus 소켓 파일은 있는데 연결이 거부되면 ``exec systemd-run`` 이 실패해 래핑한 명령이
아예 실행되지 않았다(17:45 kqc nightly 로그의 "Failed to connect to user scope bus").
"""

from __future__ import annotations

import os
import socket
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "research_run.sh"


def _fake_bin(tmp: Path, *, systemd_ok: bool) -> Path:
    bindir = tmp / "bin"
    bindir.mkdir()
    fake = bindir / "systemd-run"
    if systemd_ok:
        # 성공하는 가짜: '--' 뒤 명령을 그대로 돌리고 표시를 남긴다
        fake.write_text('#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done; shift\n'
                        'echo via-systemd; exec "$@"\n')
    else:
        fake.write_text("#!/bin/sh\necho 'Failed to connect to user scope bus' >&2\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return bindir


def _run(tmp: Path, *, systemd_ok: bool) -> subprocess.CompletedProcess:
    run_dir = tmp / "run"
    run_dir.mkdir()
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(run_dir / "bus"))  # -S 검사를 통과시키는 소켓 파일(연결은 안 받는다)
    try:
        env = {**os.environ, "XDG_RUNTIME_DIR": str(run_dir),
               "PATH": f"{_fake_bin(tmp, systemd_ok=systemd_ok)}:{os.environ['PATH']}"}
        return subprocess.run(["bash", str(SCRIPT), "sh", "-c", "echo ran"],
                              env=env, capture_output=True, text=True, timeout=30, check=False)
    finally:
        sock.close()


def test_falls_back_to_nice_when_bus_refuses(tmp_path):
    r = _run(tmp_path, systemd_ok=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ran"]


def test_uses_systemd_scope_when_it_works(tmp_path):
    r = _run(tmp_path, systemd_ok=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "ran"
    assert "via-systemd" in r.stdout
