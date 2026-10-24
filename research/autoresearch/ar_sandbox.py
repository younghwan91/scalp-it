"""train.py 격리 실행(정정 1, 리뷰 C-1) — 평가기 데이터(미래 수익·체결 표)가 **없는** spawn 프로세스에서만 train.py 를 돈다.

부모(prepare.py)는 train.py 를 실행하지 않는다. 여기서 도는 프로세스가 가진 것은 읽기 전용 mmap V 와 train 모듈뿐이다.
프레임을 거슬러 올라가도 D 가 없고, 검사용 계산은 작업마다 새 프로세스(maxtasksperchild=1)라 모듈 속성 캐시가 남지 않는다.
돌려주는 것은 ndarray·int·JSON 왕복한 평범한 값뿐이다(피클로 객체를 넘겨 부모에서 코드를 돌리는 길을 막는다).
허용 목록 검사는 이 격리의 **보조**다 — 둘 다 보안 경계는 아니고, 우회 시도는 program.md 위반 = 결과 무효.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import time

import numpy as np

VIEW_KEYS = ("O", "C1", "H1", "L1", "O1", "V1", "TV1", "uni", "lim", "kx_gap", "kq_gap", "kx_r1", "kq_r1")
ALLOWED_IMPORTS = {"numpy", "pandas", "math", "functools", "itertools", "collections", "dataclasses", "typing", "__future__"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input",
                "breakpoint", "memoryview", "help", "exit", "quit", "hasattr", "type", "object", "super", "classmethod", "staticmethod",
                "dir", "id", "print"}
BANNED_ATTR = re.compile(r"^(_.*|(gi|cr|ag|f|tb|co)_\w+|load|loads|save|savez|savez_compressed|savetxt|loadtxt|genfromtxt|fromfile|tofile|"
                         r"memmap|open_memmap|fromregex|DataSource|ctypeslib|ctypes|f2py|distutils|testing|lib|core|io|util|utils|compat|"
                         r"format|os|sys|system|eval|query|setflags|flags|base|random|default_rng|seed|dump|dumps|pickle|HDFStore|"
                         r"ExcelFile|ExcelWriter|get_handle|read_\w+|to_(pickle|csv|parquet|sql|json|excel|hdf|feather|clipboard|html|latex|markdown|xml|stata|orc)|"
                         r"year|month|day|dayofweek|weekday|quarter|date|dates|days|codes|code|now|today)$")
YEAR_LIKE = range(1990, 2031)
MAX_PARAMS = 6


def lint_train(src: str) -> "list[str]":
    """train.py 허용 목록 — 위반 목록(비면 통과)."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        ln = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            bad += [f"line {ln}: import {a.name}" for a in node.names if a.name not in ALLOWED_IMPORTS]   # 하위 모듈도 금지
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS or node.level:
                bad.append(f"line {ln}: from {node.module} import")
            bad += [f"line {ln}: from-import {a.name}" for a in node.names
                    if node.module != "__future__" and (a.name == "*" or BANNED_ATTR.match(a.name))]
        elif isinstance(node, ast.ClassDef):
            bad.append(f"line {ln}: class 정의")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("__"):
            bad.append(f"line {ln}: 함수 이름 {node.name}")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or (node.id.startswith("__") and node.id != "__name__")):
            bad.append(f"line {ln}: 이름 {node.id}")
        elif isinstance(node, ast.Attribute):
            if BANNED_ATTR.match(node.attr):
                bad.append(f"line {ln}: 속성 .{node.attr}")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bad.append(f"line {ln}: 속성 대입 .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            bad.append(f"line {ln}: {type(node).__name__}")
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, (str, bytes)):
                s = v.decode("latin-1") if isinstance(v, bytes) else v
                if re.search(r"\d{6}", s) or re.search(r"(19|20)\d{2}", s):
                    bad.append(f"line {ln}: 종목코드·날짜 같은 문자열 {v!r}")
            if isinstance(v, int) and not isinstance(v, bool) and v in YEAR_LIKE:
                bad.append(f"line {ln}: 연도 같은 정수 {v}")
    return bad


def check_params(spec: dict) -> None:
    """숫자 맞추기 가드: 범위를 한 점 근처로 좁혀 손으로 맞추는 것을 막는다(상수를 코드 안에서 변환하는 우회는 막지 못한다 — 공개)."""
    if not isinstance(spec, dict) or len(spec) > MAX_PARAMS:
        raise SystemExit(f"params 는 dict, 최대 {MAX_PARAMS} 개")
    for name, s in spec.items():
        if not isinstance(name, str) or not isinstance(s, (list, tuple)) or not s:
            raise SystemExit(f"params[{name!r}] 형식 오류")
        if s[0] == "choice":
            if len(s) != 2 or len(set(map(str, s[1]))) < 2 or not all(isinstance(v, (int, float, str)) for v in s[1]):
                raise SystemExit(f"params[{name!r}] choice 는 값(int·float·str) 2개 이상")
            continue
        if s[0] not in ("int", "float") or len(s) != 3:
            raise SystemExit(f"params[{name!r}] 형식은 ('int'|'float', lo, hi) 또는 ('choice', (…))")
        lo, hi = float(s[1]), float(s[2])
        big = max(abs(lo), abs(hi))
        if not (hi > lo and hi - lo >= big / 3 and big >= 1e-6) or (s[0] == "int" and hi - lo < 3):
            raise SystemExit(f"params[{name!r}] 범위 [{lo}, {hi}] 가 너무 좁다 — 폭 ≥ 큰 쪽 절댓값의 1/3(int 는 ≥ 3)")


def train_sha(path: str) -> "tuple[str, str]":
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return hashlib.sha256(src.encode()).hexdigest()[:16], src


def _load(path: str) -> dict:
    sha, src = train_sha(path)
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
    return S


def _view(vdir: str, mode: str, arg) -> dict:
    V = {k: np.load(os.path.join(vdir, k + ".npy"), mmap_mode="r") for k in VIEW_KEYS}
    if mode == "cut":
        V = {k: v[:, :arg + 1] if v.ndim == 2 else v[:arg + 1] for k, v in V.items()}
    elif mode == "shift":
        V = {k: v[:, arg:] if v.ndim == 2 else v[arg:] for k, v in V.items()}
    elif mode == "pad":                                                      # 앞에 arg 열 빈칸을 덧댄다(초기 열까지 이동 불변 검사)
        V = {k: np.concatenate([np.full((v.shape[0], arg), False if v.dtype == np.bool_ else np.nan, v.dtype), v], axis=1) if v.ndim == 2
             else np.concatenate([np.full(arg, np.nan), v]) for k, v in V.items()}
        for v in V.values():
            v.setflags(write=False)
    elif mode == "perm":
        p = np.random.default_rng(arg).permutation(V["O"].shape[0])
        V = {k: np.ascontiguousarray(v[p]) if v.ndim == 2 else np.array(v) for k, v in V.items()}
        for v in V.values():
            v.setflags(write=False)
    return V


_W: dict = {}


def _get(vdir: str, path: str, mode: str, arg) -> "tuple[dict, dict]":
    key = (vdir, path, mode, arg)
    if _W.get("key") != key:
        _W.update(key=key, V=_view(vdir, mode, arg), S=_load(path))
    return _W["V"], _W["S"]


def task(t: tuple):
    """한 작업. t = (kind, vdir, train_path, mode, arg, x, lo, hi). kind: meta | signal."""
    kind, vdir, path, mode, arg, x, lo, hi = t
    V, S = _get(vdir, path, mode, arg)
    if kind == "meta":
        spec = json.loads(json.dumps(S["params"]))                     # 평범한 값만 부모로
        check_params(spec)
        return spec
    cfg = json.loads(json.dumps(S["cfg"](dict(x))))
    sig = S["signals"](V, dict(x))
    shape = V["O"].shape
    cand = np.asarray(sig["cand"])
    score = np.asarray(sig["score"], dtype=np.float64)
    if cand.shape != shape or score.shape != shape or cand.dtype != np.bool_:
        raise ValueError(f"cand(bool)·score 는 {shape} 모양이어야 한다 — {cand.shape}/{cand.dtype}, {score.shape}")
    sl = slice(lo, hi)
    return np.array(cand[:, sl], dtype=bool), np.array(score[:, sl], dtype=np.float64), cfg
