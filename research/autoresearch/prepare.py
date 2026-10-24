"""autoresearch 고정 평가기 v2 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

대상: **야간 보유 종목 선택** — t 일 종가 단일가(15:30)에 사서 다음 거래 가능일 시가 단일가(09:00)에 판다. 매일 다시 고른다.
같은 다리를 04·06·14(종가베팅)가 이미 시험해 기각·철회했다 — PREREG.md "이미 본 것". 이 루프는 정보 시점(t−1 + 당일 시가),
지표(노출 맞춘 선택 알파), 잠금(솔트 종목 절반 + 개봉 3회)이 다르다.

karpathy/autoresearch 의 prepare.py 자리 + dietmarwo/autoresearch-trading 의 "구조는 LLM, 숫자는 최적화기" 분리.
``train.py`` 의 ``get_strategy()`` 가 후보·순위·파라미터 범위·슬롯 수를 내면, 폴드마다 학습 구간에서 파라미터를 고르고
(라틴 하이퍼큐브 48점 → 이웃 5점 중앙값 최대 = 고원) 시험 구간에서 한 번 잰다.

v2 = 정정 1(quant-reviewer 심사 C-1·M-1~M-7·MINOR 반영, PREREG.md "정정 1"):
- C1 train.py 는 **이 프로세스에서 실행하지 않는다.** ``ar_sandbox`` 의 spawn 프로세스(읽기 전용 mmap V 만 가짐)에서 돌고,
  신호 조각(ndarray)·슬롯(int)·파라미터 범위(JSON 왕복)만 돌아온다. 허용 목록: 단일 밑줄·프레임 속성(gi_/f_/tb_…)·속성 대입·class·
  하위 모듈 import 금지. 태그가 없으면 원장이 빈 때만 실행, 태그 커밋은 ~/.cache 에 고정, 원장 행마다 MANIFEST sha.
- C2 미래정보: 절단 3곳 · C3 날짜: 앞 열 잘라내기 2곳 + 앞 덧대기 1곳(이동 불변, 최근 300열 밖을 보면 void — 재검증 J-1) · 종목 식별: 행 치환 1회 — 전부 새 프로세스.
  고른 파라미터 중 빈도 상위 4개로. V 에 지수 수준·시장 구분이 없다(수익률만 — 날짜·이전상장 대리 차단).
- C4·M-1 점수 = 폴드 **노출 단위** 선택 알파(Σα/Σ노출, bp) 평균 − 0.5σ. α_t = 포트 순수익 − 노출 × 체결 가능 유니버스 동일가중 총수익.
  덜 사거나 쉬는 날을 골라도 점수가 안 오른다(테스트). 평균 노출 < 0.5 면 void.
- M-2 출구가 없으면: 재개된 정지는 재개일 시가(상한 없음). 다시 열리지 않으면 KIND 폐지 사유로 — 합병·지주·이전·완전자회사·신청 폐지는
  표본 제외, 그 밖(감사의견·자본잠식·부도 등)은 −100%, 사유 미상·자료 끝은 표본 제외(전략·벤치마크 모두).
- M-6 졸업 심사는 탐색 쪽 바(b0·b1·b2·b3·b5·b6·b7)를 **모두** 넘을 때만 잠금 개봉을 기록하고 연다. 잠금 출력은 요약만.
- M-7 b6 팩터 회귀 = [유니버스 야간, KOSPI 야간, KOSDAQ 야간, 모멘텀·사이즈·저변동·단기반전 롱숏(같은 유니버스 3분위)]. 널 300회.
- MINOR 하한가 잠김 = 고가 = 저가 ∧ 갭이 제한폭 3%p 안(±15% 시대 수정가 비율 왜곡). keep 에 짝지은 일별 차이 NW t ≥ 1 추가.
  널 void 는 분위수에서 뺀다.
그대로 둔 것: 정지일 체결 없음(M3), 매도 비용 포함 진입일 귀속(M4), 같은 봉 금지(M5 — V 의 키), 원장 이중 기록(M1), 폴드 값 비공개(M8).

사용(작업 트리 루트):
  uv run --extra pg python research/autoresearch/prepare.py --build       # 탐색 절반 캐시(한 번, KR_QUANT_DB) → MANIFEST.json
  uv run python research/autoresearch/prepare.py --smoke                  # 체결 수·소요 시간만(점수는 출력하지 않는다)
  uv run python research/autoresearch/prepare.py > run.log 2>&1           # 실험 한 번
  uv run python research/autoresearch/prepare.py --accept                 # 직전 실행이 beats_best 면 BEST 갱신(keep 때)
  AR_FINAL=main-session uv run --extra pg python research/autoresearch/prepare.py --final   # 졸업 심사 — 메인 세션만
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ar_sandbox as SB  # noqa: E402

TRAIN = os.path.join(HERE, "train.py")
MANIFEST = os.path.join(HERE, "MANIFEST.json")
LEDGER = os.path.join(HERE, "ledger")                                       # git 밖(.gitignore) — 되돌리기에 안 지워진다
CACHE = os.path.expanduser("~/.cache/scalp-it/autoresearch")
SHADOW = os.path.expanduser("~/.cache/scalp-it/autoresearch_ledger")       # 원장 사본·태그 고정(작업 트리를 지워도 남는다)
PRIVATE = os.path.expanduser("~/.cache/scalp-it/autoresearch_private")     # 폴드별 값·BEST — 루프 에이전트는 읽지 않는다
CACHE_FILE = "search_s0.pkl"
PREREG_TAG = "autoresearch-overnight-prereg-2"
GUARDED_FILES = ("research/autoresearch/prepare.py", "research/autoresearch/ar_sandbox.py", "research/autoresearch/PREREG.md",
                 "research/autoresearch/program.md", "tests/test_autoresearch_guards.py", "pyproject.toml", "uv.lock")
CORE_FILES = ("costs/tax.py", "stats/sharpe.py")                            # 설치된 krx-quant-core 중 평가기가 부르는 것

SALT = "scalp-it/autoresearch-overnight:"                                   # swing-it 잠금 절반과 독립
DATA = ("2007-11-22", "2026-09-30")                                         # 09-14~ adj_v2 는 KRX 분봉 정규장 값(애프터마켓 제외)
LIMIT_CHANGE = pd.Timestamp("2015-06-15")                                   # 가격제한폭 ±15% → ±30%
TICK_CHANGE = pd.Timestamp("2023-01-25")                                    # 호가단위 개편(코스피·코스닥 통일)
LIMIT_EPS = 0.005                                                           # 제한폭 0.5%p 이내 종가 = 상한가(못 산다)
LOCK_TOL = 0.03                                                             # 고가 = 저가 ∧ 제한폭 3%p 이내 = 잠김
FEE = 0.00015                                                               # 편도 수수료 정가(키움 온라인 기본) — 할인·이벤트 없음
SLIP_TICKS = 1.0                                                            # 단일가 편도 슬리피지 1틱(보수 — 1주 주문은 단일가에 그대로 체결)
MIN_ADV = 5e8                                                               # 전일까지 20일 평균 거래대금 5억원
MIN_PRICE = 1000.0                                                          # 전일 원주가 1,000원
MIN_HIST = 55                                                               # 전일까지 60거래일 중 거래일 55 이상
ANOMALY = 0.01                                                              # 하룻밤 |수익| > 제한폭 + 1%p = 자료 이상 → 표본 제외
EVENT_DELIST = re.compile(r"합병|지주회사|이전상장|시장 ?상장|완전자회사|주식교환|주식이전|신청에 의한|상장폐지 ?신청|존속기간|"
                          r"해산|증권거래소 ?상장|증권투자회사")                  # 정정 1 재검증 J-2: 해산(삼성물산·제일모직 합병이 이 사유)·재상장·펀드 = 모호 → 표본 제외
SLOTS = (5, 10, 20, 40)
SEED = 20261024
N_EVAL = 48
K_PLATEAU = 5
MIN_TRADES_PER_YEAR = 200
MIN_FOLD_TRADES_PER_YEAR = 50
MAX_THIN_FOLDS = 2
MIN_EXPOSURE = 0.5
DISPERSION = 0.5
KEEP_DSCORE, KEEP_FOLD_SHARE, KEEP_TOP5_SLACK, KEEP_T = 0.5, 0.6, 0.5, 1.0  # bp · 폴드 비율 · bp · 짝 차이 NW t
CUTS = (0.37, 0.71, 0.93)
SHIFTS = (500, 1500)
PAD = 777                                                                   # 앞 덧대기 — 초기 열(2008~2011)까지 이동 불변 검사(재검증 J-1)
WARMUP = 300                                                                # 신호는 최근 300열 안의 값만 봐야 한다(이동 불변)
CHECK_PARAMS = 4
MISMATCH_CELLS = 2                                                          # 검사당 불일치 허용 칸(절대 — 유니버스 크기에 비례하지 않게)
N_NULL = 300
LOCKBOX_MAX = 3
STRESS = 1.5
BP = 1e4


def _folds() -> list:
    """(era, 학습 시작, 시험 시작, 시험 끝) — 학습 2년 → 시험. A = 2010~2016-08(scalp-it 가 일봉 야간을 보지 않은 구간),
    B = 2016-09~2026-08(04·06·14·97·108 이 본 구간)."""
    out = [("A", f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2010, 2016)]
    out.append(("A", "2014-01-01", "2016-01-01", "2016-09-01"))
    out.append(("B", "2014-09-01", "2016-09-01", "2017-01-01"))
    out += [("B", f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2017, 2026)]
    out.append(("B", "2024-01-01", "2026-01-01", "2026-09-01"))
    return out


FOLDS = _folds()
_G: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def half_of(code: str) -> int:
    """종목 절반 잠금 — 솔트 붙인 코드 sha256 첫 바이트 짝홀. 0 = 탐색, 1 = 잠금."""
    return hashlib.sha256((SALT + str(code)).encode()).digest()[0] & 1


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 캐시

def _codes_meta(con) -> "tuple[dict, set, dict]":
    """보통주 시장 구분(0 = 유가증권, 1 = 코스닥 — 평가기 세율·호가표용, 신호에는 안 준다), 뺄 코드(스팩·코넥스),
    폐지 분류(0 = 상장, 1 = 사건 폐지 → 표본 제외, 2 = 손실 폐지 → −100%, 3 = 폐지·사유 미상 → 표본 제외)."""
    cur = con.cursor()
    cur.execute("select code, name, market from stocks")
    mkt, drop, dl = {}, set(), {}
    for code, name, market in cur.fetchall():
        mkt[code], dl[code] = (1 if market == "코스닥" else 0), 0
        if name and "스팩" in name:
            drop.add(code)
    cur.execute("select code, name, market from delisted_stocks")
    for code, name, market in cur.fetchall():
        if market == "코넥스" or (name and "스팩" in name):
            drop.add(code)
        mkt.setdefault(code, 1 if market == "코스닥" else 0)
        dl.setdefault(code, 3)
    cur.execute("select code, reason from kind_delistings order by delist_date")
    for code, reason in cur.fetchall():
        if dl.get(code) in (None, 3):
            dl[code] = 1 if EVENT_DELIST.search(reason or "") else 2
    return mkt, drop, dl


def build_half(con, half: int) -> dict:
    """한 절반의 종목 × 날짜 배열. 잠금 절반(half=1)은 ``--final`` 때만 만들고 디스크에 쓰지 않는다."""
    cur = con.cursor()
    cur.execute("set statement_timeout = '1800s'")
    mkt, drop, dl = _codes_meta(con)
    cur.execute("select distinct code from daily_bars_adj_v2 where code ~ '^[0-9]{5}0$' and date between %s and %s", DATA)
    allc = sorted(r[0] for r in cur.fetchall())
    codes = [c for c in allc if c in mkt and c not in drop and half_of(c) == half]
    diag = {"codes_all": len(allc), "unknown_market": sum(c not in mkt for c in allc), "dropped": sum(c in drop for c in allc),
            "codes_half": len(codes), "dl_counts": {k: sum(dl.get(c, 3) == k for c in codes) for k in range(4)}}
    cur.execute("select date, open, close from sector_index where code = '001' and date between %s and %s order by date", DATA)
    kx = pd.DataFrame(cur.fetchall(), columns=["date", "o", "c"]).set_index("date")
    cur.execute("select date, open, close from sector_index where code = '101' and date between %s and %s order by date", DATA)
    kq = pd.DataFrame(cur.fetchall(), columns=["date", "o", "c"]).set_index("date")
    days = pd.DatetimeIndex(pd.to_datetime(kx.index))
    cur.execute("""select code, date, open, high, low, close, volume, trade_value, raw_close, close_src
                   from daily_bars_adj_v2 where code = any(%s) and date between %s and %s""", (codes, *DATA))
    df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "o", "h", "l", "c", "v", "tv", "raw", "src"])
    df["date"] = pd.to_datetime(df["date"])
    for k in ("o", "h", "l", "c", "v", "tv", "raw"):
        df[k] = pd.to_numeric(df[k], errors="coerce").astype(float)
    df.loc[df["src"] == "halted", "v"] = 0.0
    df.loc[~(df["o"] > 0), "o"] = np.nan
    df["raw"] = df["raw"].fillna(df["c"])                                    # 폐지 종목은 원주가가 없다 — 수정가로 근사(공개)
    df["tv"] = (df["tv"] * 1e6).where(df["tv"] > 0, df["raw"] * df["v"])        # 표는 백만원 단위(원천 셋 모두 확인) → 원
    ci = pd.Index(codes).get_indexer(df["code"])
    di = days.get_indexer(df["date"])
    ok = (ci >= 0) & (di >= 0)
    diag["rows"], diag["rows_off_calendar"] = int(len(df)), int((~ok).sum())
    out = {"days": days, "codes": codes, "mkt": np.array([mkt[c] for c in codes], np.int8),
           "dl": np.array([dl.get(c, 3) for c in codes], np.int8), "half_id": half}
    for k, col in (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c"), ("V", "v"), ("TV", "tv"), ("RAW", "raw")):
        a = np.full((len(codes), len(days)), np.nan, np.float32)
        a[ci[ok], di[ok]] = df[col].to_numpy()[ok]
        out[k] = a
    for k, s in (("kxo", kx["o"]), ("kxc", kx["c"]), ("kqo", kq["o"]), ("kqc", kq["c"])):
        out[k] = pd.Series(pd.to_numeric(s).to_numpy(float), index=pd.to_datetime(s.index)).reindex(days).ffill().to_numpy(float)
    out["diag"] = diag
    return out


def build() -> None:
    import psycopg2
    os.makedirs(CACHE, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    D = build_half(con, 0)                                                   # 탐색 절반만 디스크에(C1)
    with open(os.path.join(CACHE, CACHE_FILE), "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"탐색 절반 {len(D['codes'])} codes × {len(D['days'])} days · {D['diag']}")
    write_manifest()


def _core_dir() -> str:
    import krx_quant_core
    return os.path.dirname(krx_quant_core.__file__)


def write_manifest() -> None:
    man = {p: sha_file(os.path.join(ROOT, p)) for p in GUARDED_FILES}
    man.update({f"core/{p}": sha_file(os.path.join(_core_dir(), p)) for p in CORE_FILES})
    man[f"cache/{CACHE_FILE}"] = sha_file(os.path.join(CACHE, CACHE_FILE))
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def manifest_sha() -> str:
    return sha_file(MANIFEST)[:16] if os.path.exists(MANIFEST) else "none"


def load_search() -> dict:
    with open(os.path.join(CACHE, CACHE_FILE), "rb") as f:
        return derive(pickle.load(f))


# ---------------------------------------------------------------- 무결성(C1·M1·M-5)

def _git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)


def integrity(check_cache: bool = True) -> None:
    r = _git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}^{{commit}}")
    if r.returncode != 0:
        if _read_jsonl("RUNS.jsonl"):
            raise SystemExit(f"태그 {PREREG_TAG} 가 없는데 원장에 실행 기록이 있다 — 평가 거부(태그 삭제 우회 차단)")
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행(원장 빈 때만 허용)", file=sys.stderr)
        return
    commit = r.stdout.strip()
    pin_path = os.path.join(SHADOW, "TAG_PIN.json")
    pin = json.load(open(pin_path)) if os.path.exists(pin_path) else {}
    if pin.get(PREREG_TAG, commit) != commit:
        raise SystemExit(f"태그 {PREREG_TAG} 가 {pin[PREREG_TAG][:10]} 에서 {commit[:10]} 로 옮겨졌다 — 평가 거부")
    if PREREG_TAG not in pin:
        os.makedirs(SHADOW, exist_ok=True)
        pin[PREREG_TAG] = commit
        with open(pin_path, "w") as f:
            json.dump(pin, f, indent=1)
    guarded = (*GUARDED_FILES, "research/autoresearch/MANIFEST.json")
    changed = _git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", *guarded).stdout.strip()
    dirty = _git("status", "--porcelain", "--", *guarded).stdout.strip()
    if changed or dirty:
        raise SystemExit(f"평가기·의존 파일이 사전등록({PREREG_TAG}) 뒤 바뀌었다 — 평가 거부.\n{changed}\n{dirty}")
    check_manifest(check_cache)


def check_manifest(check_cache: bool = True) -> None:
    with open(MANIFEST) as f:
        man = json.load(f)
    for p, h in man.items():
        if p.startswith("cache/"):
            if check_cache and sha_file(os.path.join(CACHE, p[6:])) != h:
                raise SystemExit(f"캐시 {p} sha 불일치 — 평가 거부")
        elif p.startswith("core/"):
            if sha_file(os.path.join(_core_dir(), p[5:])) != h:
                raise SystemExit(f"krx-quant-core {p[5:]} sha 불일치 — 평가 거부")
        elif sha_file(os.path.join(ROOT, p)) != h:
            raise SystemExit(f"{p} sha 불일치 — 평가 거부")


def _paths(name: str) -> tuple:
    return os.path.join(LEDGER, name), os.path.join(SHADOW, name)


def _append(name: str, rec: dict) -> None:
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    for p in _paths(name):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


def _read_jsonl(name: str) -> list:
    out = []
    for p in _paths(name):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append([json.loads(x) for x in f if x.strip()])
    return max(out, key=len) if out else []


def n_trials() -> int:
    return len({r["train_sha"] for r in _read_jsonl("RUNS.jsonl") if "train_sha" in r})


def _private(name: str) -> str:
    os.makedirs(PRIVATE, exist_ok=True)
    return os.path.join(PRIVATE, name)


# ---------------------------------------------------------------- 체결·비용(평가기만 아는 것)

def _lag(a: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(a.shape, np.nan if a.dtype.kind == "f" else False, a.dtype)
    out[:, k:] = a[:, :-k]
    return out


def _roll(a: np.ndarray, w: int, how: str) -> np.ndarray:
    r = pd.DataFrame(a.T).rolling(w, min_periods=w)
    return getattr(r, how)().to_numpy().T


def tick_hist(price: np.ndarray, new: np.ndarray, kosdaq: np.ndarray) -> np.ndarray:
    """호가단위 — 2023-01-25 개편 전후 표. 개편 전 코스닥은 5만원 이상 100원."""
    p = np.asarray(price, float)
    old = np.select([p < 1000, p < 5000, p < 10000, p < 50000, p < 100000, p < 500000], [1, 5, 10, 50, 100, 500], 1000)
    old = np.where(kosdaq & (p >= 50000), 100, old)
    nw = np.select([p < 2000, p < 5000, p < 20000, p < 50000, p < 200000, p < 500000], [1, 5, 10, 50, 100, 500], 1000)
    return np.where(new, nw, old).astype(float)


def derive(D: dict) -> dict:
    """원 배열 → 유니버스(t−1 정보)·체결 가능·거래 수익·비용·벤치마크. 신호에는 uni 만 넘어간다."""
    from krx_quant_core.costs.tax import sell_tax_rate
    O, H, L, C, V = (np.asarray(D[k], np.float64) for k in ("O", "H", "L", "C", "V"))
    N, T = C.shape
    days = D["days"]
    lim = np.where(days < LIMIT_CHANGE, 0.15, 0.30)
    live = np.isfinite(C) & (V > 0)
    Cp = _lag(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        adv1 = _lag(_roll(np.asarray(D["TV"], np.float64), 20, "mean"))
        hist1 = _lag(_roll(live.astype(np.float64), 60, "sum"))
        raw1 = _lag(np.asarray(D["RAW"], np.float64))
        uni = _lag(live) & (adv1 >= MIN_ADV) & (hist1 >= MIN_HIST) & (raw1 >= MIN_PRICE)
        flat = H <= L
        up = C / Cp - 1
        locked_up = (up >= lim[None, :] - LIMIT_EPS) | (flat & (up >= lim[None, :] - LOCK_TOL))
        fill = live & ~locked_up                                             # 상한가(근처·잠김) 종가는 못 산다 · 정지일 없음
        gap = O / Cp - 1
        open_dn = gap <= -(lim[None, :] - LIMIT_EPS)
        locked_dn = flat & (gap <= -(lim[None, :] - LOCK_TOL))
        vex = np.isfinite(O) & (V > 0) & ~locked_dn                          # 하한가 잠김·정지일엔 못 판다
        xpx = np.where(open_dn, L, O)                                        # 하한가 시가(안 잠김) = 그날 저가로 판다(보수)
    nxt = np.full((N, T + 1), T, np.int32)
    for s in range(T - 1, -1, -1):
        nxt[:, s] = np.where(vex[:, s], s, nxt[:, s + 1])
    t_idx = np.arange(T)
    sx = np.full((N, T), T, np.int64)
    sx[:, :-1] = nxt[:, 1:T]
    ok = sx < T
    hold = sx - t_idx[None, :]
    dl = np.asarray(D.get("dl", np.zeros(N, np.int8)))[:, None]
    lost = ~ok & (dl == 2)                                                   # 손실 폐지: 다시 안 열림 → −100%
    sxc = np.minimum(sx, T - 1)
    rows = np.arange(N)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, xpx[rows, sxc] / C - 1, np.where(lost, -1.0, np.nan))   # 사건 폐지·미상·자료 끝 = NaN(표본 밖)
        nxt_o = np.full((N, T), np.nan)
        nxt_o[:, :-1] = O[:, 1:] / C[:, :-1] - 1
        lim1 = np.r_[lim[1:], lim[-1]]
        anom = (hold == 1) & (np.abs(nxt_o) > lim1[None, :] + ANOMALY)
    kos = D["mkt"].astype(bool)[:, None]
    tax_k = np.array([sell_tax_rate(d.date(), "KOSPI") for d in days])
    tax_q = np.array([sell_tax_rate(d.date(), "KOSDAQ") for d in days])
    tax = np.where(kos, tax_q[sxc], tax_k[sxc])
    new = (days >= TICK_CHANGE)[None, :]
    raw = np.asarray(D["RAW"], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_x = np.where(ok & (r > -1), raw * (1 + r), raw)
        slip = SLIP_TICKS * (tick_hist(raw, new, kos) / raw + tick_hist(raw_x, new, kos) / raw_x)
    cost = 2 * FEE + tax + slip
    elig = uni & fill & ~anom & np.isfinite(r) & np.isfinite(cost)
    n_el = elig.sum(0)
    bench = np.where(elig, r, 0.0).sum(0) / np.maximum(n_el, 1)              # 체결 가능 유니버스 동일가중 야간 총수익
    kx_on = np.r_[D["kxo"][1:] / D["kxc"][:-1] - 1, 0.0]
    kq_on = np.r_[D["kqo"][1:] / D["kqc"][:-1] - 1, 0.0]
    key = np.random.default_rng(SEED).permutation(N).astype(float)          # 동점 처리(코드 순서 = 상장 순서 편향 차단)
    D2 = dict(D)
    D2.update(lim=lim, uni=uni, elig=elig, r=np.where(elig, r, 0.0).astype(np.float32), cost=np.where(elig, cost, 0.0).astype(np.float32),
              bench=bench, n_elig=n_el, kx_on=kx_on, kq_on=kq_on, key=key, adv1=adv1)
    for k in ("r", "cost", "elig", "bench", "uni"):
        D2[k].setflags(write=False)
    D2["diag"] = {**D.get("diag", {}), "anom": int((anom & uni).sum()), "carry": int((ok & (hold > 1) & uni).sum()),
                  "lost": int((lost & uni).sum()), "excluded_noexit": int((~ok & ~lost & uni).sum()), "elig_cells": int(elig.sum())}
    return D2


# ---------------------------------------------------------------- 신호 보기(파일) · 샌드박스 풀

def view_arrays(D: dict) -> dict:
    """train.py 가 보는 묶음 — t 열에는 t 일 시가와 t−1 일까지의 값만(M5). 날짜·종목코드·지수 수준·시장 구분 없음(C3)."""
    V = {"O": np.asarray(D["O"], np.float64)}
    for k, src in (("C1", "C"), ("H1", "H"), ("L1", "L"), ("O1", "O"), ("V1", "V"), ("TV1", "TV")):
        V[k] = _lag(np.asarray(D[src], np.float64))
    V["uni"] = np.asarray(D["uni"], bool)
    V["lim"] = np.asarray(D["lim"], np.float64)
    kx1, kq1 = np.r_[np.nan, D["kxc"][:-1]], np.r_[np.nan, D["kqc"][:-1]]
    kx2, kq2 = np.r_[np.nan, np.nan, D["kxc"][:-2]], np.r_[np.nan, np.nan, D["kqc"][:-2]]
    V.update(kx_gap=D["kxo"] / kx1 - 1, kq_gap=D["kqo"] / kq1 - 1, kx_r1=kx1 / kx2 - 1, kq_r1=kq1 / kq2 - 1)
    assert set(V) == set(SB.VIEW_KEYS)
    return V


def write_view(D: dict, vdir: str) -> str:
    os.makedirs(vdir, exist_ok=True)
    for k, a in view_arrays(D).items():
        np.save(os.path.join(vdir, k + ".npy"), a)
    return vdir


def ensure_view(D: dict) -> str:
    """탐색 절반 보기 파일(캐시 sha 별). 잠금 절반은 /dev/shm 임시 폴더에만(``final``)."""
    tag = sha_file(os.path.join(CACHE, CACHE_FILE))[:12]
    vdir = os.path.join(CACHE, f"view_{tag}")
    if not os.path.exists(os.path.join(vdir, "DONE")):
        write_view(D, vdir)
        open(os.path.join(vdir, "DONE"), "w").close()
    return vdir


def _pool(workers: int, fresh: bool = False):
    """spawn 풀 — 자식은 V(mmap)와 train 모듈만 가진다. fresh = 작업마다 새 프로세스(검사용)."""
    return mp.get_context("spawn").Pool(workers, maxtasksperchild=1 if fresh else None)


def _t(kind, vdir, path, x=None, lo=0, hi=0, mode="full", arg=None) -> tuple:
    return (kind, vdir, path, mode, arg, dict(x or {}), lo, hi)


def clamp_cfg(cfg: dict) -> dict:
    k = cfg.get("slots", 10)
    if k not in SLOTS:
        raise ValueError(f"cfg['slots']={k!r} 는 메뉴 밖: {SLOTS}")
    extra = set(cfg) - {"slots"}
    if extra:
        raise ValueError(f"cfg 에 모르는 키 {sorted(extra)} — 청산·손절·비중 규칙은 설계 공간에 없다")
    return {"slots": int(k)}


# ---------------------------------------------------------------- 시뮬레이션(부모, 신뢰 영역)

def simulate(D: dict, cand: np.ndarray, score: np.ndarray, slots: int, lo: int, hi: int, exclude=None,
             cost_mult: float = 1.0) -> dict:
    """[lo, hi) 열(cand·score 는 그 창의 조각)마다 후보 ∧ 체결 가능 중 순위 상위 ``slots`` 를 종가에 1/slots 씩 사서 다음 출구에 판다."""
    E = np.asarray(cand, bool) & D["elig"][:, lo:hi]
    if exclude is not None:
        E &= ~exclude[:, None]
    sc = np.where(E, np.asarray(score, float), -np.inf)
    E &= np.isfinite(sc)
    r, cost, key = D["r"][:, lo:hi], D["cost"][:, lo:hi], D["key"]
    port, expo = np.zeros(hi - lo), np.zeros(hi - lo)
    ti, ii, rn, rg = [], [], [], []
    for j in range(hi - lo):
        idx = np.flatnonzero(E[:, j])
        if not len(idx):
            continue
        if len(idx) > slots:
            idx = idx[np.lexsort((key[idx], -sc[idx, j]))[:slots]]
        net = np.maximum(r[idx, j] - cost_mult * cost[idx, j], -1.0)
        port[j], expo[j] = net.sum() / slots, len(idx) / slots
        ti.append(np.full(len(idx), lo + j))
        ii.append(idx)
        rn.append(net)
        rg.append(r[idx, j])
    cat = (lambda a: np.concatenate(a) if a else np.zeros(0))
    return {"port": port, "expo": expo, "lo": lo, "hi": hi, "t": cat(ti).astype(int), "i": cat(ii).astype(int),
            "net": cat(rn), "gross": cat(rg)}


def fold_stats(D: dict, sim: dict) -> dict:
    """노출 단위 선택 알파(bp): Σ(포트 − 노출 × 벤치) / Σ노출. 일별 정규화 α(= α_t / 폴드 평균 노출)는 keep 의 짝 비교용."""
    lo, hi = sim["lo"], sim["hi"]
    a = sim["port"] - sim["expo"] * D["bench"][lo:hi]
    yrs = max((D["days"][hi - 1] - D["days"][lo]).days / 365.25, 1e-9) if hi > lo else 1e-9
    ex = float(sim["expo"].sum())
    em = float(sim["expo"].mean()) if hi > lo else 0.0
    return {"alpha": float(a.sum() / ex * BP) if ex > 0 else float("nan"), "net": float(sim["port"].mean() * BP) if hi > lo else float("nan"),
            "expo": em, "n": int(len(sim["net"])), "years": yrs, "daily_alpha": a, "daily_norm": a / em if em > 0 else a * np.nan}


def _window(D: dict, lo: str, hi: str) -> "tuple[int, int]":
    return int(D["days"].searchsorted(pd.Timestamp(lo))), int(D["days"].searchsorted(pd.Timestamp(hi)))


def sample_params(spec: dict, n: int, seed: int) -> "tuple[list[dict], np.ndarray]":
    names = list(spec)
    if not names:
        return [{}], np.zeros((1, 0))
    rng = np.random.default_rng(seed)                                        # 라틴 하이퍼큐브
    U = (np.argsort(rng.random((len(names), n)), axis=1).T + rng.random((n, len(names)))) / n
    pts = []
    for u in U:
        x = {}
        for k, name in enumerate(names):
            s = spec[name]
            if s[0] == "choice":
                x[name] = s[1][min(int(u[k] * len(s[1])), len(s[1]) - 1)]
            else:
                v = s[1] + u[k] * (s[2] - s[1])
                x[name] = int(round(v)) if s[0] == "int" else float(v)
        pts.append(x)
    return pts, U


def plateau_pick(U: np.ndarray, y: np.ndarray, k: int = K_PLATEAU) -> int:
    """이웃 k 개(자기 포함) **중앙값**이 가장 높은 점 — 고원(외딴 극값 하나가 이웃 손실을 덮지 못한다)."""
    if U.shape[1] == 0 or len(y) <= k:
        return int(np.argmax(y))
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d, axis=1, kind="stable")[:, :k]
    return int(np.argmax(np.median(y[nb], axis=1)))


def _objective(D: dict, out, lo: int, hi: int) -> float:
    cand, score, cfg = out
    st = fold_stats(D, simulate(D, cand, score, clamp_cfg(cfg)["slots"], lo, hi))
    ok = st["n"] >= MIN_TRADES_PER_YEAR * st["years"] and st["expo"] >= MIN_EXPOSURE and np.isfinite(st["alpha"])
    return st["alpha"] if ok else -1e9


def _safe_task(t):
    try:
        return SB.task(t)
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다(SystemExit 는 위로)
        print(f"샌드박스 작업 실패: {e!r}", file=sys.stderr, flush=True)
        return None


def choose_params(D: dict, pool, vdir: str, spec: dict, train_path: str = TRAIN, n_eval: int = N_EVAL, folds=None) -> list:
    folds = list(range(len(FOLDS))) if folds is None else list(folds)
    meta, tasks, pts_all = [], [], {}
    for fi in folds:
        pts, U = sample_params(spec, n_eval, SEED + 101 * fi)
        pts_all[fi] = (pts, U)
        a, b = _window(D, FOLDS[fi][1], FOLDS[fi][2])
        for j, x in enumerate(pts):
            meta.append((fi, j, a, b))
            tasks.append(_t("signal", vdir, train_path, x, a, b))
    y = {fi: np.full(len(pts_all[fi][0]), -1e9) for fi in folds}
    for (fi, j, a, b), out in zip(meta, pool.imap(_safe_task, tasks, chunksize=1), strict=True):
        if out is not None:
            y[fi][j] = _objective(D, out, a, b)
    return [pts_all[fi][0][plateau_pick(pts_all[fi][1], y[fi])] for fi in folds]


def test_signals(D: dict, pool, vdir: str, params: list, train_path: str = TRAIN, folds=None) -> list:
    folds = list(range(len(FOLDS))) if folds is None else list(folds)
    tasks = []
    for fi, x in zip(folds, params, strict=True):
        a, b = _window(D, FOLDS[fi][2], FOLDS[fi][3])
        tasks.append(_t("signal", vdir, train_path, x, a, b))
    outs = pool.map(SB.task, tasks, chunksize=1)
    return [{"fold": fi, "x": x, "cand": o[0], "score": o[1], "slots": clamp_cfg(o[2])["slots"]}
            for fi, x, o in zip(folds, params, outs, strict=True)]


def run_tests(D: dict, sigs: list, null: "tuple | None" = None, cost_mult: float = 1.0, exclude=None) -> dict:
    """시험 구간 한 번. ``null`` = ("rank"|"thin", seed) — 같은 신호 조각에서 순위·후보만 무작위로."""
    rng = np.random.default_rng(None if null is None else null[1] * 7919 + 17)
    res = {"folds": [], "sims": []}
    for sg in sigs:
        era, _, lo, hi = FOLDS[sg["fold"]]
        a, b = _window(D, lo, hi)
        cand, score = sg["cand"], sg["score"]
        if null and null[0] == "rank":
            score = rng.random(score.shape)
        elif null and null[0] == "thin":
            base = D["elig"][:, a:b]
            p = (cand & base).sum() / max(base.sum(), 1)
            cand, score = base & (rng.random(cand.shape) < p), rng.random(score.shape)
        sim = simulate(D, cand, score, sg["slots"], a, b, exclude=exclude, cost_mult=cost_mult)
        st = fold_stats(D, sim)
        res["folds"].append({"fold": sg["fold"], "era": era, "test": [lo, hi], "params": sg["x"], "slots": sg["slots"],
                             **{k: v for k, v in st.items() if k not in ("daily_alpha", "daily_norm")}})
        res["sims"].append({**sim, "daily_alpha": st["daily_alpha"], "daily_norm": st["daily_norm"]})
    return res


def summarize(res: dict) -> dict:
    al = np.array([f["alpha"] for f in res["folds"]], float)
    n = np.array([f["n"] for f in res["folds"]])
    yrs = np.array([f["years"] for f in res["folds"]])
    days = np.array([s["hi"] - s["lo"] for s in res["sims"]]) if res["sims"] else np.ones(len(al))
    expo = float(np.average([f["expo"] for f in res["folds"]], weights=days)) if len(al) else 0.0
    net = np.concatenate([s["net"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    gross = np.concatenate([s["gross"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    thin = int((n < MIN_FOLD_TRADES_PER_YEAR * yrs).sum())
    void = bool(thin > MAX_THIN_FOLDS or n.sum() < MIN_TRADES_PER_YEAR * yrs.sum() or expo < MIN_EXPOSURE
                or not np.isfinite(al).all() or len(al) < 2)
    win, loss = net[net > 0], net[net <= 0]
    era = np.array([f["era"] for f in res["folds"]])
    daily = np.concatenate([s["port"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    return {"score": float(al.mean() - DISPERSION * al.std(ddof=1)) if not void else -1e9, "void": void,
            "alpha_mean": float(np.nanmean(al)) if len(al) else float("nan"),
            "alpha_std": float(np.nanstd(al, ddof=1)) if len(al) > 1 else float("nan"), "folds_pos": int((al > 0).sum()),
            "n_folds": int(len(al)), "eraA_alpha": float(np.nanmean(al[era == "A"])) if (era == "A").any() else float("nan"),
            "eraB_alpha": float(np.nanmean(al[era == "B"])) if (era == "B").any() else float("nan"),
            "net_bp_day": float(daily.mean() * BP) if len(daily) else float("nan"), "exposure": expo, "oos_trades": int(n.sum()),
            "oos_trades_per_year": float(n.sum() / max(yrs.sum(), 1e-9)), "thin_folds": thin,
            "trade_net_bp": float(net.mean() * BP) if len(net) else float("nan"),
            "trade_gross_bp": float(gross.mean() * BP) if len(gross) else float("nan"),
            "payoff_ratio": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "win_rate": float((net > 0).mean()) if len(net) else float("nan")}


def top_codes(res: dict, k: int = 5) -> np.ndarray:
    i = np.concatenate([s["i"] for s in res["sims"]]) if res["sims"] else np.zeros(0, int)
    net = np.concatenate([s["net"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    if not len(i):
        return np.zeros(0, int)
    return pd.Series(net).groupby(i).sum().sort_values(ascending=False).index[:k].to_numpy()


def top_removed_score(D: dict, sigs: list, res: dict, k: int = 5) -> float:
    ex = np.zeros(len(D["codes"]), bool)
    ex[top_codes(res, k)] = True
    s = summarize(run_tests(D, sigs, exclude=ex))
    return s["score"] if not s["void"] else -1e9


def _same(a_c, a_s, b_c, b_s) -> "tuple[bool, str]":
    if a_c.shape != b_c.shape:
        return False, f"모양 {a_c.shape} ≠ {b_c.shape}"
    tol = MISMATCH_CELLS
    nc = int((a_c != b_c).sum())
    both = a_c & b_c
    ns = int((~np.isclose(a_s[both], b_s[both], rtol=1e-6, atol=1e-9, equal_nan=True)).sum())
    return (nc <= tol and ns <= tol), f"cand {nc}칸 · score {ns}칸 불일치(허용 {tol})"


def checks(D: dict, vdir: str, params: list, workers: int, train_path: str = TRAIN) -> "tuple[bool, str]":
    """C2 절단 · C3 이동 불변(앞 열 잘라내기 2 + 앞 덧대기 1 — 300열 이후 전 구간) · 행 치환 — 작업마다 새 프로세스, 파라미터 빈도 상위 CHECK_PARAMS 개."""
    T = D["C"].shape[1]
    N = len(D["codes"])
    uniq = pd.Series([json.dumps(x, sort_keys=True) for x in params]).value_counts().index[:CHECK_PARAMS]
    with _pool(workers, fresh=True) as pool:
        for xs in uniq:
            x = json.loads(xs)
            cuts = [int(T * f) for f in CUTS]
            tasks = [_t("signal", vdir, train_path, x, 0, T)]
            tasks += [_t("signal", vdir, train_path, x, 0, c + 1, "cut", c) for c in cuts]
            tasks += [_t("signal", vdir, train_path, x, 0, T - m, "shift", m) for m in SHIFTS]
            tasks += [_t("signal", vdir, train_path, x, 0, T, "perm", SEED)]
            tasks += [_t("signal", vdir, train_path, x, 0, T + PAD, "pad", PAD)]
            outs = pool.map(SB.task, tasks, chunksize=1)
            fc, fs = outs[0][0], outs[0][1]
            for c, o in zip(cuts, outs[1:4], strict=True):
                ok, why = _same(fc[:, :c + 1], fs[:, :c + 1], o[0], o[1])
                if not ok:
                    return False, f"미래정보(절단 {c}/{T}): {why}"
            for m, o in zip(SHIFTS, outs[4:6], strict=True):
                ok, why = _same(fc[:, m + WARMUP:], fs[:, m + WARMUP:], o[0][:, WARMUP:], o[1][:, WARMUP:])
                if not ok:
                    return False, f"이동 불변(앞 {m}열 제거, {WARMUP}열 뒤부터): {why}"
            ok, why = _same(fc[:, WARMUP:], fs[:, WARMUP:], outs[7][0][:, PAD + WARMUP:], outs[7][1][:, PAD + WARMUP:])
            if not ok:
                return False, f"이동 불변(앞 {PAD}열 덧대기, {WARMUP}열부터): {why}"
            p = np.random.default_rng(SEED).permutation(N)
            ok, why = _same(fc[p], fs[p], outs[6][0], outs[6][1])
            if not ok:
                return False, f"행 치환(종목 식별): {why}"
    return True, "ok"


def nw_t(x: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    e = x - x.mean()
    s = (e * e).sum()
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] * e[:-k]).sum()
    return float(x.mean() / np.sqrt(max(s, 1e-300) / len(x) ** 2))


def beats(s: dict, res: dict, top5: float) -> "tuple[bool, str]":
    if s["void"]:
        return False, "void"
    best = _private("BEST.json")
    if not os.path.exists(best):
        return True, "첫 유효 실행"
    with open(best) as f:
        b = json.load(f)
    folds = res["folds"]
    better = sum(f["alpha"] > g["alpha"] for f, g in zip(folds, b["folds"], strict=True))
    need = int(np.ceil(KEEP_FOLD_SHARE * len(folds)))
    d = np.concatenate([s_["daily_norm"] for s_ in res["sims"]]) - np.load(_private("BEST_daily.npy"))
    t = nw_t(d)
    ok = s["score"] >= b["score"] + KEEP_DSCORE and better >= need and top5 >= b["top5"] - KEEP_TOP5_SLACK and t >= KEEP_T
    return bool(ok), (f"dscore {s['score'] - b['score']:+.3f}bp · better_folds {better}/{len(folds)} (need {need}) · "
                      f"dtop5 {top5 - b['top5']:+.3f}bp · paired_t {t:+.2f} (need {KEEP_T})")


def _lint_or_exit(path: str) -> "tuple[str, str]":
    sha, src = SB.train_sha(path)
    bad = SB.lint_train(src)
    if bad:
        raise SystemExit("train.py 허용 목록 위반 — 평가 거부:\n  " + "\n  ".join(bad))
    return sha, src


def search_eval(D: dict, vdir: str, workers: int, train_path: str = TRAIN) -> dict:
    with _pool(workers) as pool:
        spec = pool.apply(SB.task, (_t("meta", vdir, train_path),))
        params = choose_params(D, pool, vdir, spec, train_path)
        sigs = test_signals(D, pool, vdir, params, train_path)
    res = run_tests(D, sigs)
    return {"spec": spec, "params": params, "sigs": sigs, "res": res}


def evaluate(workers: int) -> dict:
    integrity()
    _G["t0"] = time.time()
    sha, src = _lint_or_exit(TRAIN)
    D = load_search()
    vdir = ensure_view(D)
    E = search_eval(D, vdir, workers)
    res = E["res"]
    s = summarize(res)
    ok, why = checks(D, vdir, E["params"], workers)
    if not ok:
        s.update(void=True, score=-1e9)
    s["lookahead_check"] = why
    s["top5_removed_score"] = top_removed_score(D, E["sigs"], res) if not s["void"] else float("nan")
    s["beats_best"], s["beats_detail"] = beats(s, res, s["top5_removed_score"])
    s["n_trials"] = len({r["train_sha"] for r in _read_jsonl("RUNS.jsonl") if "train_sha" in r} | {sha})
    s["train_sha"] = sha
    s["elapsed_s"] = round(time.time() - _G["t0"], 1)
    _append("RUNS.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "score": s["score"], "void": s["void"],
                           "beats_best": s["beats_best"], "lookahead": why, "manifest": manifest_sha(), "tag": PREREG_TAG})
    with open(_private(f"train_{sha}.py"), "w", encoding="utf-8") as f:
        f.write(src)
    with open(_private("last_run.json"), "w") as f:
        json.dump({"summary": s, "folds": res["folds"]}, f, ensure_ascii=False, indent=1, default=str)
    np.save(_private("last_daily.npy"), np.concatenate([x["daily_norm"] for x in res["sims"]]))
    shown = ("score", "void", "beats_best", "beats_detail", "lookahead_check", "folds_pos", "n_folds", "oos_trades", "oos_trades_per_year",
             "thin_folds", "exposure", "top5_removed_score", "trade_net_bp", "trade_gross_bp", "payoff_ratio", "win_rate", "n_trials",
             "train_sha", "elapsed_s")                                       # 폴드별·시대별 값은 숨긴다(M8)
    print("---")
    for k in shown:
        v = s[k]
        print(f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}")
    return s


def smoke(workers: int) -> None:
    """런타임·체결 수만 본다. 점수·알파·수익은 출력하지 않는다(사전등록 전 공개 규칙)."""
    _G["t0"] = time.time()
    _lint_or_exit(TRAIN)
    D = load_search()
    log(f"derive 끝 · {len(D['codes'])} codes × {len(D['days'])} days · diag {D['diag']}")
    vdir = ensure_view(D)
    fl = [0, len(FOLDS) - 1]
    with _pool(workers) as pool:
        spec = pool.apply(SB.task, (_t("meta", vdir, TRAIN),))
        params = choose_params(D, pool, vdir, spec, n_eval=4, folds=fl)
        sigs = test_signals(D, pool, vdir, params, folds=fl)
    log("학습 2폴드 × 4점 · 시험 끝")
    for sg in sigs:
        a, b = _window(D, FOLDS[sg["fold"]][2], FOLDS[sg["fold"]][3])
        sim = simulate(D, sg["cand"], sg["score"], sg["slots"], a, b)
        print(f"fold {sg['fold']} {FOLDS[sg['fold']][2]}~{FOLDS[sg['fold']][3]}: 체결 {len(sim['net'])} · 날 {b - a} · "
              f"노출 {sim['expo'].mean():.2f} · 유니버스 중앙 {int(np.median(D['n_elig'][a:b]))}")
    ok, why = checks(D, vdir, params, workers)
    print(f"lookahead_check: {why} · elapsed {time.time() - _G['t0']:.0f}s")


def accept() -> None:
    integrity(check_cache=False)
    with open(_private("last_run.json")) as f:
        last = json.load(f)
    cur, _ = SB.train_sha(TRAIN)
    if last["summary"]["train_sha"] != cur or not last["summary"]["beats_best"]:
        raise SystemExit("직전 실행이 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
    with open(_private("BEST.json"), "w") as f:
        json.dump({"score": last["summary"]["score"], "top5": last["summary"]["top5_removed_score"], "train_sha": cur,
                   "folds": [{"fold": f["fold"], "alpha": f["alpha"]} for f in last["folds"]]}, f, indent=1)
    shutil.copy(_private("last_daily.npy"), _private("BEST_daily.npy"))
    _append("RUNS.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "accept": True, "score": last["summary"]["score"],
                           "manifest": manifest_sha(), "tag": PREREG_TAG})
    print(f"BEST 갱신: score {last['summary']['score']:.4f} bp")


# ---------------------------------------------------------------- 졸업 심사(--final, 메인 세션만)

def factor_returns(D: dict) -> np.ndarray:
    """같은 유니버스 야간 롱숏(상위 3분위 − 하위 3분위, 동일가중 총수익) — 모멘텀(12−1)·사이즈(20일 거래대금)·저변동(20일)·단기반전. t−1 정보로 정렬."""
    C = np.asarray(D["C"], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = _lag(C, 21) / _lag(C, 252) - 1
        size = np.log(D["adv1"])
        ret = C / _lag(C) - 1
        vol = -_lag(_roll(ret, 20, "std"))
        rev = -(_lag(C) / _lag(C, 2) - 1)
    E, r = D["elig"], np.asarray(D["r"], np.float64)
    out = []
    for f in (mom, size, vol, rev):
        rk = pd.DataFrame(np.where(E, f, np.nan)).rank(axis=0, pct=True).to_numpy()
        hi, lo = E & (rk > 2 / 3), E & (rk <= 1 / 3)
        out.append(np.where(hi, r, 0).sum(0) / np.maximum(hi.sum(0), 1) - np.where(lo, r, 0).sum(0) / np.maximum(lo.sum(0), 1))
    return np.column_stack(out)


def factor_t(D: dict, res: dict, lag: int = 5) -> "tuple[float, float]":
    """일 포트 순수익 ~ [유니버스 야간, KOSPI·KOSDAQ 야간, 팩터 롱숏 4] — 절편(bp/일)과 NW t."""
    F = factor_returns(D)
    ys, Xs = [], []
    for s in res["sims"]:
        lo, hi = s["lo"], s["hi"]
        ys.append(s["port"])
        Xs.append(np.column_stack([np.ones(hi - lo), D["bench"][lo:hi], D["kx_on"][lo:hi], D["kq_on"][lo:hi], F[lo:hi]]))
    y, X = np.concatenate(ys), np.vstack(Xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    Sg = (X * e[:, None]).T @ (X * e[:, None])
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None])
        Sg += (1 - k / (lag + 1)) * (G + G.T)
    Vc = XtX_inv @ Sg @ XtX_inv
    return float(b[0] * BP), float(b[0] / np.sqrt(Vc[0, 0]))


def lockbox_used() -> int:
    return len([r for r in _read_jsonl("LOCKBOX.jsonl") if r.get("event") == "open"])


def _null_scores(D: dict, vdir: str, E: dict, workers: int, train_path: str) -> dict:
    out = {"rank": [], "thin": [], "params": []}
    for kind in ("rank", "thin"):
        for r in range(N_NULL):
            s = summarize(run_tests(D, E["sigs"], null=(kind, r + 1)))
            if not s["void"]:
                out[kind].append(s["score"])
    rng = np.random.default_rng(SEED + 9)
    with _pool(workers) as pool:
        for _ in range(N_NULL):
            px = [sample_params(E["spec"], 1, int(rng.integers(1 << 30)))[0][0] for _ in E["params"]]
            try:
                s = summarize(run_tests(D, test_signals(D, pool, vdir, px, train_path)))
            except Exception:  # noqa: BLE001
                continue
            if not s["void"]:
                out["params"].append(s["score"])
    return out


def final(workers: int, train_path: str = TRAIN) -> None:
    import psycopg2
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    if os.environ.get("AR_FINAL") != "main-session" or os.environ.get("AR_LOOP") == "1":
        raise SystemExit("--final 은 메인 세션만(AR_FINAL=main-session, 루프 에이전트 셸이 아님) — PREREG.md")
    integrity()
    used = lockbox_used()
    if used >= LOCKBOX_MAX:
        raise SystemExit(f"잠금 절반은 이미 {used}회 열렸다(최대 {LOCKBOX_MAX}) — PREREG.md")
    runs = [r for r in _read_jsonl("RUNS.jsonl") if "train_sha" in r]
    if not runs:
        raise SystemExit("RUNS.jsonl 이 비었다 — 루프를 돈 작업 트리에서 실행해야 시행 수 N 이 맞다")
    if {r.get("manifest") for r in runs} != {manifest_sha()}:
        raise SystemExit("원장 실행들의 MANIFEST sha 가 지금과 다르다 — 평가기가 루프 중간에 바뀌었다, 심사 거부")
    N = n_trials()
    _G["t0"] = time.time()
    sha, _ = _lint_or_exit(train_path)
    Ds = load_search()
    vdir = ensure_view(Ds)
    E = search_eval(Ds, vdir, workers, train_path)
    s0 = summarize(E["res"])
    chk, why = checks(Ds, vdir, E["params"], workers, train_path)
    top5_s = top_removed_score(Ds, E["sigs"], E["res"])
    stress = summarize(run_tests(Ds, E["sigs"], cost_mult=STRESS))
    nulls = _null_scores(Ds, vdir, E, workers, train_path)
    log("널 완료")
    a_s, t_s = factor_t(Ds, E["res"])
    dsr = deflated_sharpe_from_sample(np.concatenate([s["daily_alpha"] for s in E["res"]["sims"]]), N)
    q95 = {k: float(np.quantile(v, 0.95)) if v else float("inf") for k, v in nulls.items()}
    bars = {"b0_checks": chk and not s0["void"],
            "b1_search": s0["score"] > 0 and s0["eraA_alpha"] > 0 and s0["eraB_alpha"] > 0 and s0["trade_net_bp"] > 0 and s0["net_bp_day"] > 0,
            "b2_beats_nulls_p95": all(s0["score"] > v for v in q95.values()),
            "b3_top5_removed_positive": top5_s > 0,
            "b5_dsr_ge_095": float(dsr.get("prob_deflated_sharpe") or 0) >= 0.95,
            "b6_factor_t_gt2": t_s > 2.0,
            "b7_cost_x1.5": (not stress["void"]) and stress["score"] > 0 and stress["trade_net_bp"] > 0}
    rec = {"event": "search", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "n_trials": N, "manifest": manifest_sha(),
           "checks": why, "search": s0, "search_top5_removed": top5_s, "search_cost_x1.5": stress, "nulls_p95": q95,
           "nulls_n": {k: len(v) for k, v in nulls.items()}, "factor": {"alpha_bp": a_s, "t_nw": t_s}, "dsr_daily_alpha": dsr, "bars": bars}
    _append("LOCKBOX.jsonl", rec)
    with open(_private(f"final_{sha}_search_folds.json"), "w") as f:
        json.dump(E["res"]["folds"], f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    if not all(bars.values()):
        print("탐색 쪽 바 실패 — 잠금은 열지 않는다(M-6). 이 후보는 다시 심사하지 않는다.")
        return
    _append("LOCKBOX.jsonl", {"event": "open", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "n_trials": N,
                              "nth": used + 1})                              # 개봉 기록을 먼저 — 중간에 죽어도 횟수는 쓴 것이다
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    Dl = derive(build_half(con, 1))                                          # 잠금 절반 — 메모리·tmpfs 에서만
    tmp = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None, prefix="ar_lock_")
    try:
        El = search_eval(Dl, write_view(Dl, tmp), workers, train_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    s1 = summarize(El["res"])
    nf = s1["n_folds"]
    b4 = (not s1["void"]) and s1["score"] > 0 and s1["folds_pos"] >= int(np.ceil(0.6 * nf)) and s1["trade_net_bp"] > 0
    with open(_private(f"final_{sha}_lock_folds.json"), "w") as f:
        json.dump(El["res"]["folds"], f, ensure_ascii=False, indent=1, default=str)
    out = {"event": "result", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "b4_lockbox": b4,
           "lockbox": {k: s1[k] for k in ("score", "void", "folds_pos", "n_folds", "trade_net_bp", "payoff_ratio", "oos_trades")},
           "pass": bool(b4 and all(bars.values()))}
    _append("LOCKBOX.jsonl", out)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    _G["t0"] = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.build:
        build()
    elif a.smoke:
        smoke(a.workers)
    elif a.final:
        final(a.workers)
    elif a.accept:
        accept()
    else:
        evaluate(a.workers)


if __name__ == "__main__":
    main()
