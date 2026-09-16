"""``research/nightly.toml`` + ``scripts/nightly_pair_sweep.py`` + simnode 크론 배선.

nightly job 은 ``kqc nightly`` 가 simnode 에서 돌린다. 여기서는 DB 없이 조립만 본다:
job 정의가 코어 로더로 읽히는지, 래퍼가 쌍·날짜·그리드를 제대로 만드는지, 크론 줄이
파일의 다른 줄과 같은 모양인지.
"""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import date
from pathlib import Path

from krx_quant_core.runtime.nightly import load_jobs

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "nightly_pair_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("nightly_pair_sweep", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_nightly_toml_defines_pair_sweep_job():
    jobs = {j.name: j for j in load_jobs(ROOT)}
    job = jobs["pair-sweep-yesterday"]
    assert job.timeout_min == 90
    assert job.weekdays_only is True
    assert job.cmd[0] == "/home/young/.local/bin/uv"
    assert "scripts/nightly_pair_sweep.py" in job.cmd


def test_pairs_from_alerts_for_that_day_only(tmp_path):
    log = tmp_path / "pair_alerts.log"
    rows = [
        {"notice": "[돌발재료] 무관한 줄", "day": "2026-09-09"},
        {"ts": "2026-09-09 09:13:52", "leader": "069540", "follower": "010170"},
        {"ts": "2026-09-09 09:24:43", "leader": "069540", "follower": "010170"},
        {"ts": "2026-09-09 10:00:00", "leader": "111111", "follower": "222222"},
        {"ts": "2026-09-08 10:00:00", "leader": "333333", "follower": "444444"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    nps = _load()
    assert nps.pairs_from_alerts(log, date(2026, 9, 9)) == ["069540:010170", "111111:222222"]
    assert nps.pairs_from_alerts(log, date(2026, 9, 10)) == []
    assert nps.pairs_from_alerts(tmp_path / "missing.log", date(2026, 9, 9)) == []


def test_sweep_argv_uses_documented_grid():
    nps = _load()
    argv = nps.sweep_argv(date(2026, 9, 9), ["069540:010170"], jobs=4)
    assert argv[:2] == ["--replay", "2026-09-09"]
    assert "--pairs" in argv and argv[argv.index("--pairs") + 1] == "069540:010170"
    grids = [argv[i + 1] for i, a in enumerate(argv) if a == "--grid"]
    assert grids == ["surge_window=10,15,30", "cosurge_min=0.005,0.002,0.001,0"]
    assert "--yes" in argv and argv[argv.index("--jobs") + 1] == "4"


def test_no_pairs_skips_cleanly(tmp_path, capsys):
    nps = _load()
    calls: list = []
    rc = nps.main([], latest_day=lambda: date(2026, 9, 9),
                  alerts_log=tmp_path / "none.log", run=calls.append)
    assert rc == 0 and calls == []
    assert "건너뛴다" in capsys.readouterr().out


def test_runs_sweep_with_pairs(tmp_path):
    log = tmp_path / "a.log"
    log.write_text(json.dumps({"ts": "2026-09-09 09:00:00", "leader": "A", "follower": "B"}),
                   encoding="utf-8")
    nps = _load()
    calls: list = []
    rc = nps.main([], latest_day=lambda: date(2026, 9, 9), alerts_log=log,
                  run=lambda argv: calls.append(argv) or 0)
    assert rc == 0 and len(calls) == 1 and "A:B" in calls[0]


def test_crontab_nightly_line_matches_file_style():
    text = (ROOT / "deploy" / "crontab.simnode").read_text(encoding="utf-8")
    line = next(x for x in text.splitlines() if "kqc nightly" in x and not x.startswith("#"))
    assert line == (
        '45 17 * * 1-5 cd /home/young/git/scalp-it && export $(grep -v "^#" '
        '../quant-airflow/.env | xargs) && deploy/research_run.sh /home/young/.local/bin/uv '
        'run kqc nightly . >> data/nightly.cron.log 2>&1'
    )
    times = [tuple(map(int, m.groups())) for m in
             re.finditer(r"^(\d+) (\d+) \* \* 1-5", text, flags=re.MULTILINE)]
    assert times.count((45, 17)) == 1


def test_preopen_checks_core_version_warn_only():
    src = (ROOT / "scripts" / "preopen_check.sh").read_text(encoding="utf-8")
    block = src[src.index("krx-quant-core 버전"):]
    block = block[: block.index("\n\n")]
    assert "0.5." in block
    assert "bad " not in block and "warn " in block
