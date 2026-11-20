"""공통: ../quant-airflow/.env 에서 환경변수 로드 (import 시 실행).

2026-09-13: kr-quant(현 swing-it)는 스윙 리서치 전용이라 trader에서 관리할
필요가 없어져 sparse clone을 없앴다. trader가 쓰던 유일한 값(KR_QUANT_DB)은
trader가 이미 갖고 있는 quant-airflow/.env로 옮겨졌다.
"""
import os
from pathlib import Path


def find_env(start: Path) -> Path:
    """``start`` 의 조상 폴더마다 형제 ``quant-airflow/.env`` 를 찾는다(가까운 쪽 먼저).

    운영 체크아웃(``~/git/scalp-it``)은 바로 위에서 찾는다. 워크트리는
    ``~/git/.harness-wt/scalp-it/<이름>`` 처럼 더 깊이 있어서, 예전처럼 한 단계 위만
    보면 ``FileNotFoundError`` 로 pytest 수집이 0건에서 멈췄다.
    """
    tried = []
    for parent in start.resolve().parents:
        cand = parent / "quant-airflow" / ".env"
        tried.append(cand)
        if cand.exists():
            return cand
    # 예전엔 /home/young/Documents/git/kr-quant/.env 로 폴백했는데, 2026-09-11 레포가
    # ~/git 으로 옮겨져 그 경로가 사라졌다 — 죽은 경로로 폴백하지 않고 명확히 알린다.
    raise FileNotFoundError(f"quant-airflow/.env 를 찾을 수 없다: {tried[1] if len(tried) > 1 else start}")


try:
    _env = find_env(Path(__file__).resolve().parent)
except FileNotFoundError:
    # 공개본·CI 처럼 형제 quant-airflow 가 없는 환경에서 테스트가 스크립트를 import 할 수 있게
    # 하는 명시적 선택(tests/conftest.py 가 켠다). 운영에서는 켜지 않으므로 여전히 큰 소리로 죽는다.
    if os.environ.get("SCALP_ENVLOAD_OPTIONAL") != "1":
        raise
    _env = None
for line in (_env.read_text().splitlines() if _env else []):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    v = v.strip().strip('"').strip("'")
    os.environ.setdefault(k.strip(), v)
