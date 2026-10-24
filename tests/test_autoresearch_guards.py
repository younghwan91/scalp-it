"""autoresearch(야간 보유) 평가기 가드 — 위반을 주입해 실제로 막히는지 본다(통과만 보는 테스트는 규칙이 죽어도 초록이다)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "research" / "autoresearch" / "prepare.py"
_spec = importlib.util.spec_from_file_location("ar_prepare", _PATH)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def _raw(n: int = 4, t: int = 140, start: str = "2012-01-02") -> dict:
    days = pd.bdate_range(start, periods=t)
    rng = np.random.default_rng(0)
    C = (10000 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, t)), axis=1))).astype(np.float32)
    O = (C * (1 + rng.normal(0, 0.003, (n, t)))).astype(np.float32)
    D = {"days": days, "codes": [f"A{i:05d}" for i in range(n)], "mkt": np.array([0, 1] * (n // 2), np.int8), "half_id": 0,
         "O": O, "H": np.maximum(O, C) * np.float32(1.01), "L": np.minimum(O, C) * np.float32(0.99), "C": C,
         "V": np.full((n, t), 1e5, np.float32), "TV": np.full((n, t), 1e9, np.float32), "RAW": C.copy(),
         "kxo": np.linspace(1000, 1100, t), "kxc": np.linspace(1000, 1100, t), "kqo": np.linspace(500, 550, t), "kqc": np.linspace(500, 550, t),
         "diag": {}}
    return D


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "train.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


BASE = Path(__file__).resolve().parents[1] / "research" / "autoresearch" / "train.py"


# ---------------------------------------------------------------- C1 허용 목록·숫자 맞추기

@pytest.mark.parametrize("bad", [
    "import os", "from pathlib import Path", "x = open('f')", "x = getattr(1, 'real')", "x = (1).__class__",
    "import numpy as np\nx = np.load('f.npy')", "import numpy as np\nx = np.random.rand(3)", "y = 2015", "c = '005930'",
    "d = '2020-03-01'", "import pandas as pd\npd.read_csv('x')", "exec('1')", "global g", "import numpy as np\nnp.zeros(1).setflags(write=True)",
    "from numpy import load", "import numpy as np\nx = np.lib",
])
def test_lint_rejects(bad):
    assert P.lint_train(bad)


def test_lint_passes_baseline():
    assert P.lint_train(BASE.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("spec", [
    {"q": ("float", 0.9, 0.99)}, {"w": ("int", 10, 11)}, {"c": ("choice", (5,))}, {f"p{i}": ("int", 1, 20) for i in range(7)},
])
def test_params_narrow_or_too_many_rejected(spec):
    with pytest.raises(SystemExit):
        P.check_params(spec)


def test_params_reasonable_ranges_pass():
    P.check_params({"w": ("int", 1, 20), "q": ("float", 0.5, 0.95), "s": ("choice", (5, 10)), "z": ("float", -1.0, 1.0)})


def test_cfg_menu_only():
    with pytest.raises(ValueError):
        P.clamp_cfg({"slots": 7})
    with pytest.raises(ValueError):
        P.clamp_cfg({"slots": 10, "stop": 0.02})
    assert P.clamp_cfg({"slots": 20}) == {"slots": 20}


# ---------------------------------------------------------------- C3·M5·M7 보기

def test_view_has_no_same_day_close_and_is_readonly_copy():
    D = P.derive(_raw())
    V = P.view(D)
    assert not {"C", "H", "L", "V", "TV", "days", "codes"} & set(V)
    assert np.allclose(V["C1"][:, 5], D["C"][:, 4]) and np.allclose(V["O"][:, 5], D["O"][:, 5])
    with pytest.raises(ValueError):
        V["C1"][0, 5] = 1.0
    c = np.array(V["C1"])
    c[0, 5] = -1.0
    assert D["C"][0, 4] != -1.0


# ---------------------------------------------------------------- C2 미래정보

FUTURE = '''
import numpy as np
def signals(V, x):
    c1 = np.asarray(V["C1"])
    nxt = np.full(c1.shape, np.nan)
    nxt[:, :-1] = c1[:, 1:]                      # t+1 열의 C1 = t 일 종가 = 미래
    return {"cand": np.asarray(V["uni"]).copy(), "score": nxt / c1}
def cfg(x):
    return {"slots": 5}
def get_strategy():
    return {"params": {}, "signals": signals, "cfg": cfg}
'''

GLOBAL_NORM = '''
import numpy as np
def signals(V, x):
    o = np.asarray(V["O"])
    return {"cand": np.asarray(V["uni"]).copy(), "score": (o - np.nanmean(o)) / np.nanstd(o)}
def cfg(x):
    return {"slots": 5}
def get_strategy():
    return {"params": {}, "signals": signals, "cfg": cfg}
'''


@pytest.mark.parametrize("body", [FUTURE, GLOBAL_NORM])
def test_causality_catches_leaks(tmp_path, body):
    D = P.derive(_raw())
    ok, why = P.causality(D, [{}], _write(tmp_path, body))
    assert not ok, why


def test_causality_passes_baseline():
    D = P.derive(_raw())
    ok, why = P.causality(D, [{"w": 5, "slots": 5}], str(BASE))
    assert ok, why


# ---------------------------------------------------------------- 체결(M3·M4)

def test_limit_up_close_and_halt_cannot_be_bought():
    R = _raw()
    t = 80
    R["C"][0, t] = R["C"][0, t - 1] * np.float32(1.148)           # 상한가(2012 년 ±15%) 근처 종가
    R["V"][1, t] = 0.0                                           # 정지일
    D = P.derive(R)
    assert D["uni"][0, t] and D["uni"][1, t]
    assert not D["elig"][0, t] and not D["elig"][1, t]
    assert D["elig"][2, t]


def test_halted_exit_is_carried_to_next_open():
    R = _raw()
    t = 80
    R["V"][0, t + 1] = 0.0
    D = P.derive(R)
    assert D["elig"][0, t]
    assert np.isclose(D["r"][0, t], R["O"][0, t + 2] / R["C"][0, t] - 1, rtol=1e-5)


def test_no_exit_within_carry_is_total_loss():
    R = _raw(t=200)
    t = 80
    R["V"][0, t + 1:] = 0.0                                      # 다시는 안 열림
    R["O"][0, t + 1:] = np.nan
    D = P.derive(R)
    assert D["elig"][0, t] and D["r"][0, t] == -1.0


def test_cost_includes_tax_fee_and_two_ticks():
    D = P.derive(_raw())
    t = 80
    raw = float(D["RAW"][0, t])
    tick = P.tick_hist(np.array([raw]), np.array([False]), np.array([False]))[0]
    lo_bound = 2 * P.FEE + 0.0025 + tick / raw                   # 2012 년 매도세 0.25%, 진입 틱
    assert D["cost"][0, t] > lo_bound


def test_tick_table_change():
    p = np.array([1500.0, 15000.0, 70000.0])
    assert list(P.tick_hist(p, np.array([False] * 3), np.array([False] * 3))) == [5, 50, 100]
    assert list(P.tick_hist(p, np.array([False] * 3), np.array([True] * 3))) == [5, 50, 100]
    assert list(P.tick_hist(p, np.array([True] * 3), np.array([False] * 3))) == [1, 10, 100]


def test_uni_guard_cannot_be_bypassed_by_cand():
    R = _raw()
    R["TV"][0] = 1e6                                             # 거래대금 미달 → 유니버스 밖
    D = P.derive(R)
    cand = np.ones_like(D["elig"])
    score = np.tile(np.array([[9.0], [1.0], [1.0], [1.0]]), (1, D["elig"].shape[1]))
    sim = P.simulate(D, cand, score, 5, 70, 130)
    assert len(sim["net"]) and 0 not in set(sim["i"])


# ---------------------------------------------------------------- C4 노출 맞춘 점수

def test_holding_everything_scores_minus_cost_and_timing_does_not_help():
    D = P.derive(_raw(n=6))
    lo, hi = 70, 130
    allc = np.ones_like(D["elig"])
    score = np.zeros(D["elig"].shape)
    sim = P.simulate(D, allc, score, 40, lo, hi)
    a = sim["port"] - sim["expo"] * D["bench"][lo:hi]
    E = D["elig"][:, lo:hi]
    mean_cost = np.where(E, D["cost"][:, lo:hi], 0).sum(0) / 40
    assert np.allclose(a, -mean_cost, atol=1e-9)
    good = D["bench"] > 0                                        # 사후에 시장이 오른 밤만 노출(타이밍 오라클)
    sim2 = P.simulate(D, allc & good[None, :], score, 40, lo, hi)
    st = P.fold_stats(D, sim2)
    assert D["bench"][lo:hi][good[lo:hi]].mean() > 0 and st["alpha"] <= 0    # 오른 밤만 골라도 점수는 못 올린다


def test_summarize_voids_thin_trading():
    folds = [{"era": "A", "alpha": 5.0, "years": 1.0, "n": 10, "expo": 0.1} for _ in range(18)]
    sims = [{"net": np.zeros(10), "gross": np.zeros(10), "port": np.zeros(250)} for _ in range(18)]
    s = P.summarize({}, {"folds": folds, "sims": sims})
    assert s["void"] and s["score"] < -1e8


# ---------------------------------------------------------------- 고원·절반

def test_plateau_pick_prefers_plateau_over_spike():
    rng = np.random.default_rng(0)
    U = rng.random((60, 2))
    y = np.where(U[:, 0] > 0.6, 0.10, -0.10)
    U = np.vstack([U, [[0.05, 0.05]]])
    y = np.append(y, 1.0)
    assert U[P.plateau_pick(U, y), 0] > 0.6


def test_half_split_is_salted_stable_and_balanced():
    import hashlib
    codes = [f"{i:05d}0" for i in range(4000)]
    h = np.array([P.half_of(c) for c in codes])
    assert 0.45 < h.mean() < 0.55
    plain = np.array([hashlib.sha256(c.encode()).digest()[0] & 1 for c in codes])
    assert 0.4 < (h == plain).mean() < 0.6                       # swing-it 절반과 독립


# ---------------------------------------------------------------- M1 원장·잠금 횟수

@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LEDGER", str(tmp_path / "wt"))
    monkeypatch.setattr(P, "SHADOW", str(tmp_path / "shadow"))
    monkeypatch.setattr(P, "PRIVATE", str(tmp_path / "private"))
    return tmp_path


def test_ledger_double_write_and_longer_wins(ledgers):
    P._append("RUNS.jsonl", {"train_sha": "a"})
    P._append("RUNS.jsonl", {"train_sha": "b"})
    P._append("RUNS.jsonl", {"train_sha": "a"})
    assert P.n_trials() == 2
    (ledgers / "wt" / "RUNS.jsonl").write_text("")               # 작업 트리 쪽을 지워도
    assert P.n_trials() == 2                                     # 사본이 센다


def test_final_refuses_outside_main_session(ledgers, monkeypatch):
    monkeypatch.delenv("AR_FINAL", raising=False)
    with pytest.raises(SystemExit):
        P.final(1)
    monkeypatch.setenv("AR_FINAL", "main-session")
    monkeypatch.setenv("AR_LOOP", "1")
    with pytest.raises(SystemExit):
        P.final(1)


def test_final_refuses_after_three_openings(ledgers, monkeypatch):
    monkeypatch.setenv("AR_FINAL", "main-session")
    monkeypatch.delenv("AR_LOOP", raising=False)
    monkeypatch.setattr(P, "integrity", lambda check_cache=True: None)
    P._append("RUNS.jsonl", {"train_sha": "a"})
    for _ in range(3):
        P._append("LOCKBOX.jsonl", {"event": "open"})
    with pytest.raises(SystemExit, match="이미 3회"):
        P.final(1)


def test_manifest_mismatch_refuses(monkeypatch, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("a")
    man = tmp_path / "MANIFEST.json"
    man.write_text(json.dumps({"x.txt": "0" * 64}))
    monkeypatch.setattr(P, "ROOT", str(tmp_path))
    monkeypatch.setattr(P, "MANIFEST", str(man))
    with pytest.raises(SystemExit, match="sha 불일치"):
        P.check_manifest(check_cache=False)
    man.write_text(json.dumps({"x.txt": P.sha_file(str(f))}))
    P.check_manifest(check_cache=False)
