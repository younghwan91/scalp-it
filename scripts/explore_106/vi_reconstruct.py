"""106번 — 틱·호가에서 VI(변동성완화장치) 발동·해제를 복원한다. `vi_events` 는 비어 있다.

규칙(문서 106 §1 · tests/test_explore_106.py 로 고정):

1. 연속 세션(09:00:00 ~ 15:19:59) 안에서 같은 종목의 **체결 공백 ≥ 110초, ≤ 600초**.
   (VI 는 2분 단일가 + 최대 30초 무작위 연장. 10분 넘는 공백은 저유동 종목의 잠잠함이다.)
2. 가격 조건: 재개 첫 체결가 / 공백 직전 체결가 − 1 의 절댓값 ≥ 1%, **또는** 공백 직전 60초 변동 ≥ ±3%.
   (정적 VI 10% · 동적 VI 2~3% 조건의 근사. 35번이 보였듯 공백 길이만으로는 유동성 공백과 못 가른다.)
3. 호가 조건: 공백 중 호가 갱신이 멈췄어야 한다(단일가 접수 중엔 체결 호가가 안 바뀐다).
   공백 시작 2초 안의 잔여 갱신은 봐준다.
4. 시장 조건: 공백 구간의 **전 종목** 초당 틱 수가 그날 중앙값의 20% 이상이어야 한다.
   (모든 종목이 같이 멈췄으면 수집 중단이지 VI 가 아니다 — bad_days 에 없는 부분 수집일도 걸린다.)

모든 후보(규칙 1 통과)를 돌려주고 `keep`·`reason` 열로 2~4 를 기록한다 — 왜 뺐는지 보고하기 위해서다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_MIN, GAP_MAX = 110, 600
JUMP_MIN = 0.01
PRE_WIN, PRE_MOVE_MIN = 60, 0.03
QUOTE_GRACE, QUOTE_MAX = 2, 0           # 공백 시작 뒤 2초를 지나서 온 호가 갱신 허용 개수
MARKET_MIN_RATE = 0.2
SESSION_LO, SESSION_HI = 9 * 3600, 15 * 3600 + 20 * 60     # [09:00:00, 15:20:00)

COLUMNS = ["code", "sec_prev", "sec_resume", "gap", "price_prev", "price_resume", "jump",
           "pre_move", "n_quote_gap", "market_rate", "direction", "keep", "reason"]


def _market_rate(tk: pd.DataFrame) -> tuple[np.ndarray, float]:
    """초당 전 종목 틱 수(세션 안)와 그 중앙값(틱이 1건 이상인 초 기준)."""
    s = tk.sec.to_numpy(np.int64)
    m = (s >= SESSION_LO) & (s < SESSION_HI)
    cnt = np.bincount(s[m] - SESSION_LO, minlength=SESSION_HI - SESSION_LO).astype(float)
    med = float(np.median(cnt[cnt > 0])) if (cnt > 0).any() else np.nan
    return cnt, med


def reconstruct(tk: pd.DataFrame, qt: pd.DataFrame) -> pd.DataFrame:
    """tk: (code, sec, seq, price) · qt: (code, sec). sec 는 자정 기준 초. → COLUMNS 의 DataFrame."""
    cnt, med = _market_rate(tk)
    q_by_code = {c: np.sort(g.sec.to_numpy(np.int64)) for c, g in qt.groupby("code")}
    rows: list[dict] = []
    for code, g in tk.groupby("code", sort=True):
        g = g.sort_values(["sec", "seq"], kind="stable")
        sec = g.sec.to_numpy(np.int64)
        px = g.price.to_numpy(np.float64)
        d = np.diff(sec)
        qs = q_by_code.get(code, np.empty(0, np.int64))
        for i in np.nonzero(d >= GAP_MIN)[0]:
            s0, s1 = int(sec[i]), int(sec[i + 1])
            if s0 < SESSION_LO or s1 >= SESSION_HI or s1 - s0 > GAP_MAX:
                continue
            p0, p1 = px[i], px[i + 1]
            jump = p1 / p0 - 1.0
            j = np.searchsorted(sec[: i + 1], s0 - PRE_WIN, side="right") - 1
            pre_move = p0 / px[j] - 1.0 if j >= 0 else np.nan
            nq = int(((qs >= s0 + QUOTE_GRACE) & (qs < s1)).sum())
            rate = float(cnt[s0 + 1 - SESSION_LO: s1 - SESSION_LO].mean() / med) if med > 0 else np.nan
            reasons = []
            if not (abs(jump) >= JUMP_MIN or (np.isfinite(pre_move) and abs(pre_move) >= PRE_MOVE_MIN)):
                reasons.append("가격 조건 미달")
            if nq > QUOTE_MAX:
                reasons.append("공백 중 호가 갱신")
            if not (rate >= MARKET_MIN_RATE):
                reasons.append("시장 전체 틱 중단")
            direction = int(np.sign(jump)) if abs(jump) >= JUMP_MIN / 2 else (
                int(np.sign(pre_move)) if np.isfinite(pre_move) else 0)
            rows.append({"code": code, "sec_prev": s0, "sec_resume": s1, "gap": s1 - s0,
                         "price_prev": p0, "price_resume": p1, "jump": jump, "pre_move": pre_move,
                         "n_quote_gap": nq, "market_rate": rate, "direction": direction,
                         "keep": not reasons, "reason": "; ".join(reasons)})
    return pd.DataFrame(rows, columns=COLUMNS)
