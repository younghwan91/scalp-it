"""크론·런처의 ``pkill -f``/``pgrep -f`` 패턴은 첫 글자를 ``[]`` 로 감싼다.

맨 패턴(``pkill -f 'scalp-tick-collect'``)은 그 패턴이 적힌 명령줄 자체를 맞힌다. 크론은
한 줄을 ``sh -c '<줄 전체>'`` 로 돌리므로 15:40 정리 줄의 첫 pkill 이 자기 부모 셸을
SIGTERM 으로 죽였고, 뒤의 두 pkill 은 한 번도 돌지 않았다(2026-11-19 재현, rc=143).
``'[s]calp-tick-collect'`` 는 같은 프로세스를 맞히지만 자기 명령줄의 ``[s]calp`` 는 못 맞힌다.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# -f 앞뒤에 다른 플래그가 붙어도(pkill -9 -f, pgrep -af) 마지막 인자를 패턴으로 본다
PAT = re.compile(r"\bp(?:kill|grep)\s+(?:-\S+\s+)*?-\w*f\w*\s+(?:-\S+\s+)*(?P<q>['\"]?)(?P<pat>[^'\"\s;|&]+)(?P=q)")


def _targets() -> list[Path]:
    files = [ROOT / "deploy" / "crontab.simnode"]
    files += sorted((ROOT / "deploy").glob("*.sh"))
    files += sorted((ROOT / "scripts").rglob("*.sh"))
    return [f for f in files if f.is_file()]


def _bare_patterns(text: str) -> list[str]:
    bad = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for m in PAT.finditer(line):
            for alt in m.group("pat").split("|"):
                if not alt.startswith("["):
                    bad.append(f"{alt!r} in: {line.strip()[:120]}")
    return bad


def test_detects_the_old_cron_line():
    old = "40 15 * * 1-5 pkill -f 'scalp-tick-collect'; pkill -f 'scalp-pair-detect'; pkill -f pair_detect_daily"
    assert len(_bare_patterns(old)) == 3
    assert len(_bare_patterns("pkill -9 -f 'scalp-pair-detect'; pgrep -af scalp-tick-collect")) == 2
    fixed = "40 15 * * 1-5 pkill -f '[s]calp-tick-collect'; pgrep -fa '[s]calp-pair-detect|[p]air_detect_daily'"
    assert _bare_patterns(fixed) == []


def test_no_bare_process_patterns_in_cron_and_launchers():
    bad = []
    for f in _targets():
        bad += [f"{f.relative_to(ROOT)}: {b}" for b in _bare_patterns(f.read_text(encoding="utf-8"))]
    assert not bad, "맨 pkill/pgrep 패턴 — 첫 글자를 [] 로 감쌀 것:\n" + "\n".join(bad)
