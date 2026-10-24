"""autoresearch 고정 평가기 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

대상: **야간 보유 종목 선택** — t 일 종가 단일가(15:30)에 사서 t+1 일 시가 단일가(09:00)에 판다.
97번이 "수익은 전부 야간, 일중은 잃는 구간"이라 했고 108번이 야간 다리를 "비용과 같은 자릿수, 단일가 비용 측정 없이는
못 정한다"로 남겼다. 이 루프가 묻는 것: **종목 선택이 정가 비용(세금·수수료·단일가 슬리피지)을 넘는 알파를 내는가.**

karpathy/autoresearch 의 prepare.py 자리 + dietmarwo/autoresearch-trading 의 "구조는 LLM, 숫자는 최적화기" 분리.
``train.py`` 의 ``get_strategy()`` 가 후보·순위·파라미터 범위·슬롯 수를 내면, 여기서 폴드마다 학습 구간에서 파라미터를
고르고(라틴 하이퍼큐브 → 이웃 5점 중앙값 최대 = 고원) 시험 구간에서 한 번 잰다.

가드(swing-it 평가기 v1 리뷰 C1~C4·M1~M8 을 처음부터 반영):
- C1 진입점은 이 파일. train.py 는 AST 허용 목록(numpy·pandas·math 등만, 파일·OS·동적 실행·더블 언더스코어·난수·
  종목코드·연도 리터럴 금지)을 통과해야 새 모듈로 로드된다. 태그 뒤 평가기·의존 파일 변경 금지 + MANIFEST sha(코드·캐시).
- C2 미래정보: 평가마다 절단일 3개 × 파라미터 2개에서 V[:, :cut+1] 로 만든 신호 == 전체 신호 앞부분, 다르면 void.
- C3 V 에는 날짜·종목코드가 없다. 배열은 읽기 전용 복사본(M7).
- C4 점수 = **노출을 맞춘 선택 알파**: 그날 포트 순수익 − 그날 노출 × 체결 가능 유니버스 동일가중 야간 총수익. 쉬는 날·
  노출 조절(타이밍)은 점수에 0 이다. 종목 절반 잠금(솔트 sha256 짝홀)은 시간 홀드아웃이 아니므로 같은 날의 시장 움직임을
  공유한다 — 그래서 지표가 타이밍을 지운다. 잠금 개봉은 ``--final`` 최대 3회, 개봉 기록을 **먼저** 남긴 뒤 연다.
- M1 시행 원장 RUNS.jsonl · 잠금 원장 LOCKBOX.jsonl 은 고정 두 경로(작업 트리 ledger/ + ~/.cache 사본)에 덧붙이고,
  읽을 때는 긴 쪽을 쓴다. DSR 의 N = 고유 train sha 수.
- M2 청산은 하나뿐(다음 거래 가능 시가). 손절·익절·시간 손절은 설계 공간에 없다.
- M3 거래정지(거래량 0)일에는 사지도 팔지도 않는다. 출구가 막히면(정지·하한가 잠김) 다음 거래 가능일 시가로 미루고,
  60 거래일 안에 출구가 없으면 −100%.
- M4 모든 거래는 매도 비용까지 뺀 뒤 진입일에 귀속한다(폴드 끝 보유분 포함).
- M5 같은 봉 금지: 신호는 t−1 까지의 OHLCV 와 t 일 시가만 본다(V 의 키가 그렇게 생겼다). t 일 고가·저가·종가·거래량은
  V 에 없다. 상한가 종가(살 수 없다)·정지일 판정만 평가기가 t 일 값으로 한다.
- M6 졸업 널 = 순위 무작위 · 후보 솎기 무작위 · 파라미터 무작위(각 100회) 95 백분위 + 팩터 회귀 절편 NW t > 2.
- M8 keep 은 평가기가 쌍 비교로 판정(Δscore · 폴드 개선 수 · top5 제거 점수). run.log 에는 요약만, 폴드별 값은
  ~/.cache 비공개 경로에만 둔다(숨김은 지시 + 경로 분리이지 보안 경계가 아니다 — program.md).

사용(작업 트리 루트):
  uv run --extra pg python research/autoresearch/prepare.py --build       # 탐색 절반 캐시(한 번, KR_QUANT_DB) → MANIFEST.json
  uv run python research/autoresearch/prepare.py --smoke                  # 체결 수·소요 시간만(점수는 계산·출력하지 않는다)
  uv run python research/autoresearch/prepare.py > run.log 2>&1           # 실험 한 번
  uv run python research/autoresearch/prepare.py --accept                 # 직전 실행이 beats_best 면 BEST 갱신(keep 때)
  AR_FINAL=main-session uv run --extra pg python research/autoresearch/prepare.py --final   # 졸업 심사 — 메인 세션만
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")
MANIFEST = os.path.join(HERE, "MANIFEST.json")
LEDGER = os.path.join(HERE, "ledger")                                       # git 밖(.gitignore) — 되돌리기에 안 지워진다
CACHE = os.path.expanduser("~/.cache/scalp-it/autoresearch")
SHADOW = os.path.expanduser("~/.cache/scalp-it/autoresearch_ledger")       # 원장 사본(작업 트리를 지워도 남는다)
PRIVATE = os.path.expanduser("~/.cache/scalp-it/autoresearch_private")     # 폴드별 값·BEST — 루프 에이전트는 읽지 않는다
CACHE_FILE = "search_s0.pkl"
PREREG_TAG = "autoresearch-overnight-prereg-1"
GUARDED_FILES = ("research/autoresearch/prepare.py", "research/autoresearch/PREREG.md", "research/autoresearch/program.md",
                 "tests/test_autoresearch_guards.py", "pyproject.toml", "uv.lock")
CORE_FILES = ("costs/tax.py", "stats/sharpe.py")                            # 설치된 krx-quant-core 중 평가기가 부르는 것

SALT = "scalp-it/autoresearch-overnight:"                                   # swing-it 잠금 절반과 독립
DATA = ("2007-11-22", "2026-09-30")                                         # 09-14~ adj_v2 는 KRX 분봉 정규장 값(애프터마켓 제외)
LIMIT_CHANGE = pd.Timestamp("2015-06-15")                                   # 가격제한폭 ±15% → ±30%
TICK_CHANGE = pd.Timestamp("2023-01-25")                                    # 호가단위 개편(코스피·코스닥 통일)
LIMIT_EPS = 0.005                                                           # 제한폭 0.5%p 이내 종가는 상한가로 보고 못 산다
FEE = 0.00015                                                               # 편도 수수료 정가(키움 온라인 기본) — 할인·이벤트 없음
SLIP_TICKS = 1.0                                                            # 단일가 편도 슬리피지 1틱(보수 — 1주 주문은 단일가에 그대로 체결)
MIN_ADV = 5e8                                                               # 전일까지 20일 평균 거래대금 5억원
MIN_PRICE = 1000.0                                                          # 전일 원주가 1,000원
MIN_HIST = 55                                                               # 전일까지 60거래일 중 거래일 55 이상
MAX_CARRY = 60                                                              # 출구 미룸 최대 거래일 — 넘으면 −100%
ANOMALY = 0.01                                                              # 하룻밤 |수익| > 제한폭 + 1%p = 자료 이상 → 표본 제외
SLOTS = (5, 10, 20, 40)
SEED = 20261024
N_EVAL = 48
K_PLATEAU = 5
MAX_PARAMS = 6
MIN_TRADES_PER_YEAR = 200
MIN_FOLD_TRADES_PER_YEAR = 50
MAX_THIN_FOLDS = 2
DISPERSION = 0.5
KEEP_DSCORE, KEEP_FOLD_SHARE, KEEP_TOP5_SLACK = 0.5, 0.6, 0.5              # bp/일
N_NULL = 100
LOCKBOX_MAX = 3
STRESS = 1.5
BP = 1e4


def _folds() -> list:
    """(era, 학습 시작, 시험 시작, 시험 끝) — 학습 2년 → 시험. A = 2010~2016-08(scalp-it 가 야간을 보지 않은 구간),
    B = 2016-09~2026-08(97·108 이 야간을 순위별로 본 구간)."""
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


# ---------------------------------------------------------------- train.py 검사(C1·C3)

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "functools", "itertools", "collections", "dataclasses", "typing", "__future__"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input",
                "breakpoint", "memoryview", "help", "exit", "quit", "hasattr", "type", "object", "super", "classmethod", "staticmethod"}
BANNED_ATTR = re.compile(r"^(load|loads|save|savez|savez_compressed|savetxt|loadtxt|genfromtxt|fromfile|tofile|memmap|fromregex|DataSource|"
                         r"ctypeslib|f2py|testing|lib|os|sys|system|eval|query|setflags|flags|base|ctypes|random|default_rng|seed|"
                         r"read_\w+|to_(pickle|csv|parquet|sql|json|excel|hdf|feather|clipboard|html|latex|markdown|xml|stata|orc)|"
                         r"year|month|day|dayofweek|weekday|quarter|date|dates|days|codes|code|now|today)$")
YEAR_LIKE = range(1990, 2031)


def lint_train(src: str) -> "list[str]":
    """train.py 허용 목록 — 위반 목록(비면 통과). 완전한 샌드박스는 아니다: 우회 시도는 program.md 위반 = 결과 무효."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"line {node.lineno}: import {m}")
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name.startswith("_") and node.module != "__future__" or BANNED_ATTR.match(a.name):
                        bad.append(f"line {node.lineno}: from-import {a.name}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            bad.append(f"line {node.lineno}: 이름 {node.id}")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            bad.append(f"line {node.lineno}: 이름 {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or BANNED_ATTR.match(node.attr)):
            bad.append(f"line {node.lineno}: 속성 .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.append(f"line {node.lineno}: global/nonlocal")
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, (str, bytes)):
                s = v.decode("latin-1") if isinstance(v, bytes) else v
                if re.search(r"\d{6}", s) or re.search(r"(19|20)\d{2}", s):
                    bad.append(f"line {node.lineno}: 종목코드·날짜 같은 문자열 {v!r}")
            if isinstance(v, int) and not isinstance(v, bool) and v in YEAR_LIKE:
                bad.append(f"line {node.lineno}: 연도 같은 정수 {v}")
    return bad


def check_params(spec: dict) -> None:
    """숫자 맞추기 가드: 범위를 한 점 근처로 좁혀 손으로 맞추는 것을 막는다."""
    if len(spec) > MAX_PARAMS:
        raise SystemExit(f"params 는 최대 {MAX_PARAMS} 개(지금 {len(spec)})")
    for name, s in spec.items():
        if s[0] == "choice":
            if len(set(s[1])) < 2:
                raise SystemExit(f"params[{name!r}] choice 는 값 2개 이상")
            continue
        if s[0] not in ("int", "float") or len(s) != 3:
            raise SystemExit(f"params[{name!r}] 형식은 ('int'|'float', lo, hi) 또는 ('choice', (…))")
        lo, hi = float(s[1]), float(s[2])
        wide = hi - lo >= max(abs(lo), abs(hi)) / 3 if lo * hi > 0 else hi > lo
        if not wide or (s[0] == "int" and hi - lo < 3):
            raise SystemExit(f"params[{name!r}] 범위 [{lo}, {hi}] 가 너무 좁다 — 폭 ≥ 큰 쪽 절댓값의 1/3(int 는 ≥ 3)")


def load_strategy(path: str = TRAIN) -> dict:
    """train.py 를 **새 모듈 인스턴스**로 로드(캐시 공유 없음). 허용 목록 위반이면 거부."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    bad = lint_train(src)
    if bad:
        raise SystemExit("train.py 허용 목록 위반 — 평가 거부:\n  " + "\n  ".join(bad))
    spec = importlib.util.spec_from_file_location(f"ar_train_{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    S = mod.get_strategy()
    for k in ("params", "signals", "cfg"):
        if k not in S:
            raise SystemExit(f"get_strategy() 에 {k!r} 가 없다")
    check_params(S["params"])
    S["sha"] = hashlib.sha256(src.encode()).hexdigest()[:16]
    S["src"] = src
    return S


# ---------------------------------------------------------------- 캐시

def _codes_meta(con) -> "tuple[dict, set]":
    """보통주 시장 구분(0 = 유가증권, 1 = 코스닥)과 뺄 코드(스팩·코넥스)."""
    cur = con.cursor()
    cur.execute("select code, name, market from stocks")
    mkt, drop = {}, set()
    for code, name, market in cur.fetchall():
        mkt[code] = 1 if market == "코스닥" else 0
        if name and "스팩" in name:
            drop.add(code)
    cur.execute("select code, name, market from delisted_stocks")
    for code, name, market in cur.fetchall():
        if market == "코넥스" or (name and "스팩" in name):
            drop.add(code)
        mkt.setdefault(code, 1 if market == "코스닥" else 0)
    return mkt, drop


def build_half(con, half: int) -> dict:
    """한 절반의 종목 × 날짜 배열. 잠금 절반(half=1)은 ``--final`` 때만 만들고 디스크에 쓰지 않는다."""
    cur = con.cursor()
    cur.execute("set statement_timeout = '1800s'")
    mkt, drop = _codes_meta(con)
    cur.execute("select distinct code from daily_bars_adj_v2 where code ~ '^[0-9]{5}0$' and date between %s and %s", DATA)
    allc = sorted(r[0] for r in cur.fetchall())
    codes = [c for c in allc if c in mkt and c not in drop and half_of(c) == half]
    diag = {"codes_all": len(allc), "unknown_market": sum(c not in mkt for c in allc), "dropped": sum(c in drop for c in allc),
            "codes_half": len(codes)}
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
    out = {"days": days, "codes": codes, "mkt": np.array([mkt[c] for c in codes], np.int8), "half_id": half}
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
    man = {p: sha_file(os.path.join(ROOT, p)) for p in GUARDED_FILES if p != "research/autoresearch/MANIFEST.json"}
    man.update({f"core/{p}": sha_file(os.path.join(_core_dir(), p)) for p in CORE_FILES})
    man[f"cache/{CACHE_FILE}"] = sha_file(os.path.join(CACHE, CACHE_FILE))
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def load_search() -> dict:
    with open(os.path.join(CACHE, CACHE_FILE), "rb") as f:
        return derive(pickle.load(f))


# ---------------------------------------------------------------- 무결성(C1·M1)

def integrity(check_cache: bool = True) -> None:
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode != 0:
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행", file=sys.stderr)
        return
    guarded = (*GUARDED_FILES, "research/autoresearch/MANIFEST.json")
    changed = git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", *guarded).stdout.strip()
    dirty = git("status", "--porcelain", "--", *guarded).stdout.strip()
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
    df = pd.DataFrame(a.T)
    r = df.rolling(w, min_periods=w)
    return (r.mean() if how == "mean" else r.sum()).to_numpy().T


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
        fill = live & (C / Cp - 1 < lim[None, :] - LIMIT_EPS)                 # 상한가(근처) 종가는 못 산다 · 정지일 없음
        gap = O / Cp - 1
        open_dn = gap <= -(lim[None, :] - LIMIT_EPS)
        vex = np.isfinite(O) & (V > 0) & ~(open_dn & (H <= L))               # 하한가 잠김·정지일엔 못 판다
        xpx = np.where(open_dn, L, O)                                        # 하한가 시가(안 잠김) = 그날 저가로 판다(보수)
    nxt = np.full((N, T + 1), T, np.int32)
    for s in range(T - 1, -1, -1):
        nxt[:, s] = np.where(vex[:, s], s, nxt[:, s + 1])
    t_idx = np.arange(T)
    sx = np.full((N, T), T, np.int64)
    sx[:, :-1] = nxt[:, 1:T]
    hold = sx - t_idx[None, :]
    ok = (sx < T) & (hold <= MAX_CARRY)
    unknown_end = ~ok & (t_idx[None, :] + MAX_CARRY >= T - 1)                # 자료 끝에서 잘림 = 미상, 표본 밖
    sxc = np.minimum(sx, T - 1)
    rows = np.arange(N)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, xpx[rows, sxc] / C - 1, -1.0)
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
    elig = uni & fill & ~anom & ~unknown_end & np.isfinite(r) & np.isfinite(cost)
    n_el = elig.sum(0)
    bench = np.where(elig, r, 0.0).sum(0) / np.maximum(n_el, 1)              # 체결 가능 유니버스 동일가중 야간 총수익
    kx_on = np.r_[D["kxo"][1:] / D["kxc"][:-1] - 1, 0.0]
    kq_on = np.r_[D["kqo"][1:] / D["kqc"][:-1] - 1, 0.0]
    key = np.random.default_rng(SEED).permutation(N).astype(float)          # 동점 처리(코드 순서 = 상장 순서 편향 차단)
    D2 = dict(D)
    D2.update(lim=lim, uni=uni, elig=elig, r=np.where(elig, r, 0.0).astype(np.float32), cost=np.where(elig, cost, 0.0).astype(np.float32),
              bench=bench, n_elig=n_el, kx_on=kx_on, kq_on=kq_on, key=key)
    D2["diag"] = {**D.get("diag", {}), "anom": int(anom.sum()), "carry": int((ok & (hold > 1) & uni).sum()),
                  "lost": int((~ok & ~unknown_end & uni).sum()), "elig_cells": int(elig.sum())}
    return D2


# ---------------------------------------------------------------- 시뮬레이션

VIEW_2D = {"O": "O", "C1": "C", "H1": "H", "L1": "L", "O1": "O", "V1": "V", "TV1": "TV"}


def view(D: dict) -> dict:
    """train.py 에 넘기는 묶음 — 읽기 전용 복사본, 날짜·종목코드 없음(C3·M7). t 열에는 t 일 시가와 t−1 일까지의 값만(M5)."""
    V = {}
    for k, src in VIEW_2D.items():
        a = np.asarray(D[src], np.float64)
        V[k] = a.copy() if k == "O" else _lag(a)
    V["uni"] = np.array(D["uni"], copy=True)
    V["mkt"] = np.array(D["mkt"], copy=True)
    V["lim"] = np.array(D["lim"], copy=True)
    V["kxo"], V["kqo"] = np.array(D["kxo"], float), np.array(D["kqo"], float)
    V["kx1"], V["kq1"] = np.r_[np.nan, D["kxc"][:-1]], np.r_[np.nan, D["kqc"][:-1]]
    for a in V.values():
        a.setflags(write=False)
    return V


def _cut(V: dict, cut: int) -> dict:
    out = {}
    for k, v in V.items():
        if k == "mkt":
            out[k] = v
        else:
            out[k] = np.array(v[:, :cut + 1] if v.ndim == 2 else v[:cut + 1])
            out[k].setflags(write=False)
    return out


def clamp_cfg(cfg: dict) -> dict:
    k = cfg.get("slots", 10)
    if k not in SLOTS:
        raise ValueError(f"cfg['slots']={k!r} 는 메뉴 밖: {SLOTS}")
    extra = set(cfg) - {"slots"}
    if extra:
        raise ValueError(f"cfg 에 모르는 키 {sorted(extra)} — 청산·손절·비중 규칙은 설계 공간에 없다")
    return {"slots": int(k)}


def _signal(S: dict, V: dict, x: dict, shape: tuple) -> "tuple[np.ndarray, np.ndarray]":
    sig = S["signals"](V, x)
    cand, score = np.asarray(sig["cand"]), np.asarray(sig["score"], float)
    if cand.shape != shape or score.shape != shape or cand.dtype != bool:
        raise ValueError(f"cand(bool)·score 는 {shape} 모양이어야 한다 — {cand.shape}/{cand.dtype}, {score.shape}")
    return cand, score


def simulate(D: dict, cand: np.ndarray, score: np.ndarray, slots: int, lo: int, hi: int, exclude=None,
             cost_mult: float = 1.0) -> dict:
    """[lo, hi) 열마다 후보 ∧ 체결 가능 중 순위 상위 ``slots`` 를 종가에 1/slots 씩 사서 다음 출구 시가에 판다."""
    E = cand[:, lo:hi] & D["elig"][:, lo:hi]
    if exclude is not None:
        E &= ~exclude[:, None]
    sc = np.where(E & np.isfinite(score[:, lo:hi]), score[:, lo:hi], -np.inf)
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
    lo, hi = sim["lo"], sim["hi"]
    a = sim["port"] - sim["expo"] * D["bench"][lo:hi]
    yrs = max((D["days"][hi - 1] - D["days"][lo]).days / 365.25, 1e-9) if hi > lo else 1e-9
    return {"alpha": float(a.mean() * BP) if len(a) else float("nan"), "net": float(sim["port"].mean() * BP) if len(a) else float("nan"),
            "expo": float(sim["expo"].mean()) if len(a) else 0.0, "n": int(len(sim["net"])), "years": yrs, "daily_alpha": a}


def _window(D: dict, lo: str, hi: str) -> "tuple[int, int]":
    return int(D["days"].searchsorted(pd.Timestamp(lo))), int(D["days"].searchsorted(pd.Timestamp(hi)))


def _train_point(args):
    fi, j, x = args
    D, S, V = _G["D"], _G["S"], _G["V"]
    _, tlo, lo, _ = FOLDS[fi]
    try:
        cand, score = _signal(S, V, x, D["C"].shape)
        a, b = _window(D, tlo, lo)
        st = fold_stats(D, simulate(D, cand, score, clamp_cfg(S["cfg"](x))["slots"], a, b))
        return fi, j, (st["alpha"] if st["n"] >= MIN_TRADES_PER_YEAR * st["years"] else -1e9)
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다
        log(f"학습 점 fold {fi} #{j} 실패: {e!r}")
        return fi, j, -1e9


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


def choose_params(D: dict, workers: int, train_path: str = TRAIN, n_eval: int = N_EVAL, folds=None) -> list:
    S = load_strategy(train_path)
    _G.update(D=D, S=S, V=view(D))
    folds = range(len(FOLDS)) if folds is None else folds
    tasks, pts_all = [], {}
    for fi in folds:
        pts, U = sample_params(S["params"], n_eval, SEED + 101 * fi)
        pts_all[fi] = (pts, U)
        tasks += [(fi, j, x) for j, x in enumerate(pts)]
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(_train_point, tasks, chunksize=1)
    chosen = {}
    for fi in folds:
        pts, U = pts_all[fi]
        y = np.array([v for f, _, v in sorted(out, key=lambda r: (r[0], r[1])) if f == fi])
        chosen[fi] = pts[plateau_pick(U, y)]
    return [chosen[fi] for fi in folds]


def test_folds(D: dict, params: list, train_path: str = TRAIN, null: "tuple | None" = None, cost_mult: float = 1.0,
               exclude=None, folds=None) -> dict:
    """폴드별 고른 파라미터로 시험 구간 한 번. ``null`` = ("rank"|"thin"|"params", seed)."""
    S = load_strategy(train_path)
    V = view(D)
    folds = list(range(len(FOLDS))) if folds is None else list(folds)
    rng = np.random.default_rng(None if null is None else null[1] * 7919 + 17)
    res = {"folds": [], "sims": []}
    for fi, x in zip(folds, params, strict=True):
        era, _, lo, hi = FOLDS[fi]
        if null and null[0] == "params":
            x = sample_params(S["params"], 1, int(rng.integers(1 << 30)))[0][0]
        cand, score = _signal(S, V, x, D["C"].shape)
        if null and null[0] == "rank":
            score = rng.random(score.shape)
        elif null and null[0] == "thin":
            a, b = _window(D, lo, hi)
            base = D["elig"][:, a:b]
            p = (cand[:, a:b] & base).sum() / max(base.sum(), 1)
            cand = D["elig"] & (rng.random(cand.shape) < p)
            score = rng.random(score.shape)
        a, b = _window(D, lo, hi)
        sim = simulate(D, cand, score, clamp_cfg(S["cfg"](x))["slots"], a, b, exclude=exclude, cost_mult=cost_mult)
        st = fold_stats(D, sim)
        res["folds"].append({"fold": fi, "era": era, "test": [lo, hi], "params": x, **{k: v for k, v in st.items() if k != "daily_alpha"}})
        res["sims"].append({**sim, "daily_alpha": st["daily_alpha"]})
    return res


def summarize(D: dict, res: dict) -> dict:
    al = np.array([f["alpha"] for f in res["folds"]])
    n = np.array([f["n"] for f in res["folds"]])
    yrs = np.array([f["years"] for f in res["folds"]])
    net = np.concatenate([s["net"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    gross = np.concatenate([s["gross"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    thin = int((n < MIN_FOLD_TRADES_PER_YEAR * yrs).sum())
    void = bool(thin > MAX_THIN_FOLDS or n.sum() < MIN_TRADES_PER_YEAR * yrs.sum() or not np.isfinite(al).all())
    win, loss = net[net > 0], net[net <= 0]
    era = np.array([f["era"] for f in res["folds"]])
    daily = np.concatenate([s["port"] for s in res["sims"]]) if res["sims"] else np.zeros(0)
    return {"score": float(al.mean() - DISPERSION * al.std(ddof=1)) if not void and len(al) > 1 else -1e9, "void": void,
            "alpha_mean": float(np.nanmean(al)), "alpha_std": float(np.nanstd(al, ddof=1)), "folds_pos": int((al > 0).sum()),
            "n_folds": int(len(al)), "eraA_alpha": float(np.nanmean(al[era == "A"])) if (era == "A").any() else float("nan"),
            "eraB_alpha": float(np.nanmean(al[era == "B"])) if (era == "B").any() else float("nan"),
            "net_bp_day": float(daily.mean() * BP) if len(daily) else float("nan"),
            "exposure": float(np.mean([f["expo"] for f in res["folds"]])), "oos_trades": int(n.sum()),
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


def top_removed_score(D: dict, res: dict, params: list, k: int = 5, train_path: str = TRAIN) -> float:
    top = top_codes(res, k)
    ex = np.zeros(len(D["codes"]), bool)
    ex[top] = True
    s = summarize(D, test_folds(D, params, train_path, exclude=ex))
    return s["score"] if not s["void"] else -1e9


def causality(D: dict, params: list, train_path: str = TRAIN) -> "tuple[bool, str]":
    """미래정보 검사(C2): 절단일 3개 × 파라미터 2개에서 V[:, :cut+1] 로 새 모듈이 만든 신호 == 전체 신호 앞부분."""
    V = view(D)
    T = D["C"].shape[1]
    for x in (params[0], params[-1]):
        cf, sf = _signal(load_strategy(train_path), V, x, D["C"].shape)
        for frac in (0.37, 0.71, 0.93):
            cut = int(T * frac)
            cp, sp = _signal(load_strategy(train_path), _cut(V, cut), x, (D["C"].shape[0], cut + 1))
            a = cf[:, :cut + 1]
            if (a != cp).mean() > 0:
                return False, f"cut {frac}: cand 불일치 {(a != cp).sum()} 칸"
            if not np.allclose(sf[:, :cut + 1][a], sp[a], rtol=1e-7, atol=1e-12, equal_nan=True):
                return False, f"cut {frac}: score 불일치"
    return True, "ok"


def beats(s: dict, folds: list, top5: float) -> "tuple[bool, str]":
    if s["void"]:
        return False, "void"
    best = _private("BEST.json")
    if not os.path.exists(best):
        return True, "첫 유효 실행"
    with open(best) as f:
        b = json.load(f)
    better = sum(f["alpha"] > g["alpha"] for f, g in zip(folds, b["folds"], strict=True))
    need = int(np.ceil(KEEP_FOLD_SHARE * len(folds)))
    ok = s["score"] >= b["score"] + KEEP_DSCORE and better >= need and top5 >= b["top5"] - KEEP_TOP5_SLACK
    return bool(ok), f"dscore {s['score'] - b['score']:+.3f}bp · better_folds {better}/{len(folds)} (need {need}) · dtop5 {top5 - b['top5']:+.3f}bp"


def evaluate(workers: int) -> dict:
    integrity()
    _G["t0"] = time.time()
    D = load_search()
    S = load_strategy()
    params = choose_params(D, workers)
    res = test_folds(D, params)
    s = summarize(D, res)
    ok, why = causality(D, params)
    if not ok:
        s.update(void=True, score=-1e9)
    s["lookahead_check"] = why
    s["top5_removed_score"] = top_removed_score(D, res, params) if not s["void"] else float("nan")
    s["beats_best"], s["beats_detail"] = beats(s, res["folds"], s["top5_removed_score"])
    s["n_trials"] = len({r["train_sha"] for r in _read_jsonl("RUNS.jsonl") if "train_sha" in r} | {S["sha"]})
    s["train_sha"] = S["sha"]
    s["elapsed_s"] = round(time.time() - _G["t0"], 1)
    _append("RUNS.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": S["sha"], "score": s["score"], "void": s["void"],
                           "beats_best": s["beats_best"], "lookahead": why})
    with open(_private(f"train_{S['sha']}.py"), "w", encoding="utf-8") as f:
        f.write(S["src"])
    with open(_private("last_run.json"), "w") as f:
        json.dump({"summary": s, "folds": res["folds"]}, f, ensure_ascii=False, indent=1, default=str)
    shown = ("score", "void", "beats_best", "beats_detail", "lookahead_check", "folds_pos", "n_folds", "oos_trades", "oos_trades_per_year",
             "thin_folds", "exposure", "top5_removed_score", "trade_net_bp", "trade_gross_bp", "payoff_ratio", "win_rate", "n_trials",
             "train_sha", "elapsed_s")                                       # 폴드별·시대별 값은 숨긴다(M8)
    print("---")
    for k in shown:
        v = s[k]
        print(f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}")
    return s


def smoke(workers: int) -> None:
    """런타임·체결 수만 본다. 점수·알파·수익은 계산하지 않는다(사전등록 전 공개 규칙)."""
    _G["t0"] = time.time()
    D = load_search()
    log(f"derive 끝 · {len(D['codes'])} codes × {len(D['days'])} days · diag {D['diag']}")
    params = choose_params(D, workers, n_eval=4, folds=[0, len(FOLDS) - 1])
    log("학습 2폴드 × 4점 끝")
    S = load_strategy()
    V = view(D)
    for fi, x in zip([0, len(FOLDS) - 1], params, strict=True):
        cand, score = _signal(S, V, x, D["C"].shape)
        a, b = _window(D, FOLDS[fi][2], FOLDS[fi][3])
        sim = simulate(D, cand, score, clamp_cfg(S["cfg"](x))["slots"], a, b)
        print(f"fold {fi} {FOLDS[fi][2]}~{FOLDS[fi][3]}: 체결 {len(sim['net'])} · 날 {b - a} · 유니버스 중앙 {int(np.median(D['n_elig'][a:b]))}")
    ok, why = causality(D, params)
    print(f"lookahead_check: {why} · elapsed {time.time() - _G['t0']:.0f}s")


def accept() -> None:
    integrity(check_cache=False)
    with open(_private("last_run.json")) as f:
        last = json.load(f)
    cur = load_strategy()["sha"]
    if last["summary"]["train_sha"] != cur or not last["summary"]["beats_best"]:
        raise SystemExit("직전 실행이 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
    with open(_private("BEST.json"), "w") as f:
        json.dump({"score": last["summary"]["score"], "top5": last["summary"]["top5_removed_score"], "train_sha": cur,
                   "folds": [{"fold": f["fold"], "alpha": f["alpha"]} for f in last["folds"]]}, f, indent=1)
    _append("RUNS.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "accept": True, "score": last["summary"]["score"]})
    print(f"BEST 갱신: score {last['summary']['score']:.4f} bp/일")


# ---------------------------------------------------------------- 졸업 심사(--final, 메인 세션만)

def factor_t(D: dict, res: dict, lag: int = 5) -> "tuple[float, float]":
    """일별 포트 순수익을 [유니버스 동일가중 야간, KOSPI 야간, KOSDAQ 야간] 에 회귀 — 절편(bp/일)과 NW t."""
    ys, Xs = [], []
    for s in res["sims"]:
        lo, hi = s["lo"], s["hi"]
        ys.append(s["port"])
        Xs.append(np.column_stack([np.ones(hi - lo), D["bench"][lo:hi], D["kx_on"][lo:hi], D["kq_on"][lo:hi]]))
    y, X = np.concatenate(ys), np.vstack(Xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    Sg = (X * e[:, None]).T @ (X * e[:, None])
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None])
        Sg += (1 - k / (lag + 1)) * (G + G.T)
    Vc = XtX_inv @ Sg @ XtX_inv
    return float(b[0] * BP), float(b[0] / np.sqrt(Vc[0, 0]))


def lockbox_used() -> int:
    return len([r for r in _read_jsonl("LOCKBOX.jsonl") if r.get("event") == "open"])


def final(workers: int) -> None:
    import psycopg2
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    if os.environ.get("AR_FINAL") != "main-session" or os.environ.get("AR_LOOP") == "1":
        raise SystemExit("--final 은 메인 세션만(AR_FINAL=main-session, 루프 에이전트 셸이 아님) — PREREG.md")
    integrity()
    used = lockbox_used()
    if used >= LOCKBOX_MAX:
        raise SystemExit(f"잠금 절반은 이미 {used}회 열렸다(최대 {LOCKBOX_MAX}) — PREREG.md")
    N = n_trials()
    if N == 0:
        raise SystemExit("RUNS.jsonl 이 비었다 — 루프를 돈 작업 트리에서 실행해야 시행 수 N 이 맞다")
    _G["t0"] = time.time()
    S = load_strategy()
    _append("LOCKBOX.jsonl", {"event": "open", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": S["sha"], "n_trials": N,
                              "nth": used + 1})                              # 개봉 기록을 먼저 — 중간에 죽어도 횟수는 쓴 것이다
    Ds = load_search()
    params = choose_params(Ds, workers)
    search = test_folds(Ds, params)
    s0 = summarize(Ds, search)
    top5_s = top_removed_score(Ds, search, params)
    stress = summarize(Ds, test_folds(Ds, params, cost_mult=STRESS))
    nulls = {kind: [summarize(Ds, test_folds(Ds, params, null=(kind, r + 1)))["score"] for r in range(N_NULL)]
             for kind in ("rank", "thin", "params")}
    log("널 완료")
    a_s, t_s = factor_t(Ds, search)
    dsr = deflated_sharpe_from_sample(np.concatenate([s["daily_alpha"] for s in search["sims"]]), N)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    Dl = derive(build_half(con, 1))                                          # 잠금 절반 — 메모리에서만
    params_l = choose_params(Dl, workers)
    lock = test_folds(Dl, params_l)
    s1 = summarize(Dl, lock)
    a_l, t_l = factor_t(Dl, lock)
    nf = s0["n_folds"]
    bars = {"b1_search": s0["score"] > 0 and s0["eraA_alpha"] > 0 and s0["eraB_alpha"] > 0 and s0["trade_net_bp"] > 0 and s0["net_bp_day"] > 0,
            "b2_beats_nulls_p95": all(s0["score"] > float(np.quantile(v, 0.95)) for v in nulls.values()),
            "b3_top5_removed_positive": top5_s > 0,
            "b4_lockbox": (not s1["void"]) and s1["score"] > 0 and s1["folds_pos"] >= int(np.ceil(0.6 * nf)) and s1["trade_net_bp"] > 0,
            "b5_dsr_ge_095": float(dsr.get("prob_deflated_sharpe") or 0) >= 0.95,
            "b6_factor_t_gt2": t_s > 2.0,
            "b7_cost_x1.5": (not stress["void"]) and stress["score"] > 0 and stress["trade_net_bp"] > 0}
    out = {"event": "result", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": S["sha"], "n_trials": N, "search": s0,
           "search_top5_removed": top5_s, "search_cost_x1.5": stress,
           "nulls_p95": {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()},
           "nulls_median": {k: float(np.median(v)) for k, v in nulls.items()}, "lockbox": s1,
           "factor": {"search_alpha_bp": a_s, "search_t_nw": t_s, "lockbox_alpha_bp": a_l, "lockbox_t_nw": t_l}, "dsr_daily_alpha": dsr,
           "bars": bars, "pass": all(bars.values()), "folds_search": search["folds"], "folds_lockbox": lock["folds"]}
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
