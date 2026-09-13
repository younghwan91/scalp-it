"""공통: ../quant-airflow/.env 에서 환경변수 로드 (import 시 실행).

2026-09-13: kr-quant(현 swing-it)는 스윙 리서치 전용이라 trader에서 관리할
필요가 없어져 sparse clone을 없앴다. trader가 쓰던 유일한 값(KR_QUANT_DB)은
trader가 이미 갖고 있는 quant-airflow/.env로 옮겨졌다.
"""
import os
from pathlib import Path

_env = Path(__file__).resolve().parents[2] / "quant-airflow" / ".env"
if not _env.exists():
    # 예전엔 여기서 /home/young/Documents/git/kr-quant/.env 로 폴백했는데,
    # 2026-09-11 레포가 ~/git 으로 옮겨져 그 경로가 사라졌다 — 죽은 경로로
    # 폴백하면 다음 줄에서 FileNotFoundError 로 터지니 차라리 명확히 알린다.
    raise FileNotFoundError(f"quant-airflow/.env 를 찾을 수 없다: {_env}")
for line in _env.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    v = v.strip().strip('"').strip("'")
    os.environ.setdefault(k.strip(), v)
