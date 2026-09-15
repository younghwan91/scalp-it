"""86번 실시간 그림자용 전일 문턱 — 그날 유니버스 결정 행 dstr30 99% 분위를 data/tiger86_prevq.json 에 쓴다.

simnode 평일 17:00(83번 캐시가 16:30 에 만들어진 뒤) → trader 로 rsync. 감지기가 다음 날 09:35 전 문턱으로 쓴다.
"""
import datetime as dt
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flow_83"))
import features as F83  # noqa: E402

day = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
f = F83.CACHE / f"{day}.pkl"
if not f.exists():
    import psycopg2
    F83.build_day(psycopg2.connect(os.environ["KR_QUANT_DB"]), day, forward_ok=True)
rows = pickle.loads(f.read_bytes())["rows"]
v = rows.dstr30.to_numpy(np.float64)
v = v[np.isfinite(v)]
out = Path(__file__).resolve().parents[2] / "data" / "tiger86_prevq.json"
out.write_text(json.dumps({"day": day, "q99_dstr30": float(np.quantile(v, 0.99)), "rows": int(len(v))}))
print(out.read_text())
