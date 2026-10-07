"""86번 실시간 그림자용 전일 문턱 — 그날 유니버스 결정 행 dstr30 99% 분위를 data/tiger86_prevq.json 에 쓴다.

simnode 평일 17:00(83번 캐시가 16:30 에 만들어진 뒤). 감지기가 같은 data/ 에서 읽어 다음 날 09:35 전 문턱으로
쓴다(2026-10-07 부터 감지기도 simnode 라 rsync 가 없다).

휴장일(평일 크론이 그대로 돈다)엔 그날 데이터가 없다 — 파일을 건드리지 않고 끝낸다. 그래야 다음 거래일
감지기가 직전 거래일 문턱을 그대로 읽는다. 예전엔 build_day 가 빈 배열로 죽었다(2026-10-05 대체공휴일).
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
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    with conn.cursor() as cur:
        cur.execute("select exists(select 1 from daily_bars where date = %s)", (day,))
        trading_day = cur.fetchone()[0]
    if not trading_day:
        print(f"[prevq] {day} 일봉 없음(휴장일) — 문턱 파일을 그대로 둔다")
        sys.exit(0)
    F83.build_day(conn, day, forward_ok=True)
rows = pickle.loads(f.read_bytes())["rows"]
v = rows.dstr30.to_numpy(np.float64)
v = v[np.isfinite(v)]
out = Path(__file__).resolve().parents[2] / "data" / "tiger86_prevq.json"
out.write_text(json.dumps({"day": day, "q99_dstr30": float(np.quantile(v, 0.99)), "rows": int(len(v))}))
print(out.read_text())
