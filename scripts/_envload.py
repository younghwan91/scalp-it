"""공통: ../kr-quant/.env 에서 환경변수 로드 (import 시 실행)."""
import os
from pathlib import Path

_env = Path(__file__).resolve().parents[2] / "kr-quant" / ".env"
if not _env.exists():
    _env = Path("/home/young/Documents/git/kr-quant/.env")
for line in _env.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    v = v.strip().strip('"').strip("'")
    os.environ.setdefault(k.strip(), v)
