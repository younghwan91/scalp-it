"""autoresearch(야간 보유) 평가기 v2 가드 — 위반을 주입해 실제로 막히는지 본다(통과만 보는 테스트는 규칙이 죽어도 초록이다).

정정 1(quant-reviewer C-1·M-1~M-7)에서 재현된 우회 경로마다 주입 테스트가 하나씩 있다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DIR = Path(__file__).resolve().parents[1] / "research" / "autoresearch"
_spec = importlib.util.spec_from_file_location("ar_prepare", _DIR / "prepare.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
SB = P.SB
BASE = _DIR / "train.py"


def _raw(n: int = 4, t: int = 160, start: str = "2012-01-02", dl=None) -> dict:
    days = pd.bdate_range(start, periods=t)
    rng = np.random.default_rng(0)
    C = (9000 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, t)), axis=1))).astype(np.float32)
    O = (C * (1 + rng.normal(0, 0.003, (n, t)))).astype(np.float32)
    return {"days": days, "codes": [f"A{i:05d}" for i in range(n)], "mkt": np.array([0, 1] * (n // 2), np.int8),
            "dl": np.zeros(n, np.int8) if dl is None else np.asarray(dl, np.int8), "half_id": 0,
            "O": O, "H": np.maximum(O, C) * np.float32(1.01), "L": np.minimum(O, C) * np.float32(0.99), "C": C,
            "V": np.full((n, t), 1e5, np.float32), "TV": np.full((n, t), 1e9, np.float32), "RAW": C.copy(),
            "kxo": np.linspace(1000, 1100, t), "kxc": np.linspace(1000, 1100, t), "kqo": np.linspace(500, 550, t),
            "kqc": np.linspace(500, 550, t), "diag": {}}


def _write(tmp_path: Path, body: str, name: str = "train.py") -> str:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------- C1 허용 목록(리뷰 C-1 재현 경로 포함)

@pytest.mark.parametrize("bad", [
    "import os", "from pathlib import Path", "x = open('f')", "x = getattr(1, 'real')", "x = (1).__class__",
    "import numpy as np\nx = np.load('f.npy')", "import numpy as np\nx = np.random.rand(3)", "y = 2015", "c = '005930'",
    "d = '2020-03-01'", "import pandas as pd\npd.read_csv('x')", "exec('1')", "global g", "from numpy import load",
    "import numpy as np\nx = np.lib",
    # 정정 1 — 리뷰가 재현한 우회
    "def g():\n    yield 1\nfr = g().gi_frame.f_back.f_locals",           # 프레임 탈출
    "import numpy as np\nnp.ar_cache = {}",                                  # 공유 모듈 속성 대입(캐시로 절단 검사 무력화)
    "import numpy.ctypeslib as cl",                                         # 하위 모듈 import
    "from numpy.lib.format import open_memmap", "from pandas.io.common import get_handle",
    "import pandas as pd\npd.HDFStore", "import pandas as pd\npd.ExcelFile", "import numpy as np\nnp._core",
    "class X:\n    pass", "def __reduce__():\n    return 1", "def f():\n    yield 1", "print(1)",
])
def test_lint_rejects(bad):
    assert SB.lint_train(bad)


def test_reviewer_frame_exploit_is_rejected():
    body = ("import numpy as np\ndef _peek():\n    yield None\ndef signals(V, x):\n    g = _peek()\n    fr = g.gi_frame.f_back\n"
            "    return {'cand': V['uni'], 'score': fr.f_locals['D']['r']}\n")
    bad = SB.lint_train(body)
    assert any("gi_frame" in b for b in bad) and any("f_back" in b for b in bad)


def test_lint_passes_baseline():
    assert SB.lint_train(BASE.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("spec", [
    {"q": ("float", 0.9, 0.99)}, {"w": ("int", 10, 11)}, {"c": ("choice", (5,))}, {f"p{i}": ("int", 1, 20) for i in range(7)},
    {"z": ("float", 0.0, 1e-9)},
])
def test_params_narrow_or_too_many_rejected(spec):
    with pytest.raises(SystemExit):
        SB.check_params(spec)


def test_params_reasonable_ranges_pass():
    SB.check_params({"w": ("int", 1, 20), "q": ("float", 0.5, 0.95), "s": ("choice", (5, 10)), "z": ("float", -1.0, 1.0)})


def test_cfg_menu_only():
    with pytest.raises(ValueError):
        P.clamp_cfg({"slots": 7})
    with pytest.raises(ValueError):
        P.clamp_cfg({"slots": 10, "stop": 0.02})
    assert P.clamp_cfg({"slots": 20}) == {"slots": 20}


# ---------------------------------------------------------------- C3·M5·M7 보기

def test_view_has_no_same_day_close_levels_or_market(tmp_path):
    D = P.derive(_raw())
    V = P.view_arrays(D)
    assert not {"C", "H", "L", "V", "TV", "days", "codes", "mkt", "kx1", "kxo", "kq1", "kqo"} & set(V)
    assert np.allclose(V["C1"][:, 5], D["C"][:, 4]) and np.allclose(V["O"][:, 5], D["O"][:, 5])
    vdir = P.write_view(D, str(tmp_path / "v"))
    Vm = SB._view(vdir, "full", None)
    with pytest.raises(ValueError):
        Vm["C1"][0, 5] = 1.0                                                 # 자식이 받는 V 는 읽기 전용 mmap
    with pytest.raises(ValueError):
        D["r"][0, 5] = 1.0                                                   # 부모의 수익 배열도 잠겨 있다


# ---------------------------------------------------------------- C2·C3 검사(절단·이동·치환, 새 프로세스)

_HDR = "import numpy as np\n"
_TAIL = "\ndef cfg(x):\n    return {'slots': 5}\ndef get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': cfg}\n"
LEAKS = {
    "future": _HDR + "def signals(V, x):\n    c1 = np.asarray(V['C1'])\n    nxt = np.full(c1.shape, np.nan)\n"
                     "    nxt[:, :-1] = c1[:, 1:]\n    return {'cand': np.asarray(V['uni']).copy(), 'score': nxt / c1}" + _TAIL,
    "global_norm": _HDR + "def signals(V, x):\n    o = np.asarray(V['O'])\n"
                          "    return {'cand': np.asarray(V['uni']).copy(), 'score': (o - np.nanmean(o)) / np.nanstd(o)}" + _TAIL,
    "column_index": _HDR + "def signals(V, x):\n    u = np.asarray(V['uni'])\n    col = np.arange(u.shape[1])[None, :] + np.zeros(u.shape)\n"
                           "    return {'cand': u & (col > 700), 'score': col}" + _TAIL,
    "early_column": _HDR + "def signals(V, x):\n    u = np.asarray(V['uni'])\n    col = np.arange(u.shape[1])[None, :] + np.zeros(u.shape)\n"
                           "    return {'cand': u & (col > 350) & (col < 420), 'score': np.zeros(u.shape)}" + _TAIL,   # 재검증 J-1: 초기 열 겨냥
    "row_identity": _HDR + "def signals(V, x):\n    u = np.asarray(V['uni'])\n    row = np.arange(u.shape[0])[:, None] + np.zeros(u.shape)\n"
                           "    return {'cand': u.copy(), 'score': row}" + _TAIL,
}


@pytest.fixture(scope="module")
def toy_long(tmp_path_factory):
    D = P.derive(_raw(n=6, t=2000, start="2008-01-02"))
    vdir = P.write_view(D, str(tmp_path_factory.mktemp("view")))
    return D, vdir


@pytest.mark.parametrize("kind", list(LEAKS))
def test_checks_catch_leaks(tmp_path, toy_long, kind):
    D, vdir = toy_long
    ok, why = P.checks(D, vdir, [{}], 2, _write(tmp_path, LEAKS[kind]))
    assert not ok, f"{kind}: {why}"


def test_checks_pass_baseline(toy_long):
    D, vdir = toy_long
    ok, why = P.checks(D, vdir, [{"w": 5, "slots": 5}], 2, str(BASE))
    assert ok, why


def test_sandbox_returns_plain_values_only(tmp_path, toy_long):
    D, vdir = toy_long
    with P._pool(1) as pool:
        spec = pool.apply(SB.task, (P._t("meta", vdir, str(BASE)),))
        c, s, cfg = pool.apply(SB.task, (P._t("signal", vdir, str(BASE), {"w": 5, "slots": 10}, 100, 200),))
    assert spec == {"w": ["int", 1, 20], "slots": ["choice", [5, 10, 20, 40]]}
    assert c.dtype == bool and c.shape == (6, 100) and s.shape == (6, 100) and cfg == {"slots": 10}


# ---------------------------------------------------------------- 체결(M3·M4·M-2)

def test_limit_up_close_and_halt_cannot_be_bought():
    R = _raw()
    t = 80
    R["C"][0, t] = R["C"][0, t - 1] * np.float32(1.148)                       # 상한가(2012 년 ±15%) 근처 종가
    R["V"][1, t] = 0.0                                                        # 정지일
    D = P.derive(R)
    assert D["uni"][0, t] and D["uni"][1, t]
    assert not D["elig"][0, t] and not D["elig"][1, t] and D["elig"][2, t]


def test_halted_exit_is_carried_to_resume_open_without_cap():
    R = _raw(t=260)
    t = 80
    R["V"][0, t + 1:t + 101] = 0.0                                            # 100 거래일 정지 뒤 재개
    D = P.derive(R)
    assert D["elig"][0, t]
    assert np.isclose(D["r"][0, t], R["O"][0, t + 101] / R["C"][0, t] - 1, rtol=1e-5)


@pytest.mark.parametrize("dl,expect", [(2, "loss"), (1, "excluded"), (3, "excluded"), (0, "excluded")])
def test_no_exit_by_delisting_class(dl, expect):
    R = _raw(t=200, dl=[dl, 0, 0, 0])
    t = 80
    R["V"][0, t + 1:] = 0.0
    R["O"][0, t + 1:] = np.nan
    D = P.derive(R)
    if expect == "loss":
        assert D["elig"][0, t] and D["r"][0, t] == -1.0                      # 감사의견·자본잠식 등 → −100%
    else:
        assert not D["elig"][0, t]                                            # 합병·지주·미상·자료 끝 → 표본 밖


def test_locked_limit_down_pre2015_with_tolerance():
    R = _raw()
    t = 80
    px = R["C"][0, t] * np.float32(0.87)                                     # −13%(±15% 시대, 수정가 비율 왜곡)
    R["O"][0, t + 1] = R["H"][0, t + 1] = R["L"][0, t + 1] = R["C"][0, t + 1] = px
    D = P.derive(R)
    assert np.isclose(D["r"][0, t], R["O"][0, t + 2] / R["C"][0, t] - 1, rtol=1e-5)   # 잠김 → 다음 날 시가로 미룸


def test_cost_includes_tax_fee_and_two_ticks():
    D = P.derive(_raw())
    t = 80
    raw = float(D["RAW"][0, t])
    tick = P.tick_hist(np.array([raw]), np.array([False]), np.array([False]))[0]
    assert D["cost"][0, t] > 2 * P.FEE + 0.0025 + tick / raw


def test_tick_table_change():
    p = np.array([1500.0, 15000.0, 70000.0])
    f, t = np.array([False] * 3), np.array([True] * 3)
    assert list(P.tick_hist(p, f, f)) == [5, 50, 100] and list(P.tick_hist(p, f, t)) == [5, 50, 100]
    assert list(P.tick_hist(p, t, f)) == [1, 10, 100]


def test_uni_guard_cannot_be_bypassed_by_cand():
    R = _raw()
    R["TV"][0] = 1e6
    D = P.derive(R)
    lo, hi = 70, 130
    cand = np.ones((4, hi - lo), bool)
    score = np.tile(np.array([[9.0], [1.0], [1.0], [1.0]]), (1, hi - lo))
    sim = P.simulate(D, cand, score, 5, lo, hi)
    assert len(sim["net"]) and 0 not in set(sim["i"])


# ---------------------------------------------------------------- C4·M-1 노출 단위 점수

def test_score_is_exposure_invariant_and_timing_cannot_help():
    D = P.derive(_raw(n=6))
    lo, hi = 70, 130
    allc, score = np.ones((6, hi - lo), bool), np.zeros((6, hi - lo))
    a10 = P.fold_stats(D, P.simulate(D, allc, score, 10, lo, hi))
    a40 = P.fold_stats(D, P.simulate(D, allc, score, 40, lo, hi))
    E = D["elig"][:, lo:hi]
    mean_cost = (np.where(E, D["cost"][:, lo:hi], 0).sum() / E.sum()) * P.BP
    assert np.isclose(a10["alpha"], -mean_cost) and np.isclose(a40["alpha"], -mean_cost)   # 노출 0.6 이든 0.15 든 같다
    good = (D["bench"] > 0)[lo:hi]                                           # 사후에 시장이 오른 밤만(타이밍 오라클)
    st = P.fold_stats(D, P.simulate(D, allc & good[None, :], score, 10, lo, hi))
    assert D["bench"][lo:hi][good].mean() > 0 and st["alpha"] <= 0


def test_summarize_voids_thin_trading_and_low_exposure():
    sims = [{"net": np.zeros(400), "gross": np.zeros(400), "port": np.zeros(250), "lo": 0, "hi": 250} for _ in range(18)]
    thin = [{"era": "A", "alpha": 5.0, "years": 1.0, "n": 10, "expo": 0.9} for _ in range(18)]
    low = [{"era": "A", "alpha": 5.0, "years": 1.0, "n": 400, "expo": 0.2} for _ in range(18)]
    for folds in (thin, low):
        s = P.summarize({"folds": folds, "sims": sims})
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
    assert 0.4 < (h == plain).mean() < 0.6


def test_delisting_reason_classifier():
    assert P.EVENT_DELIST.search("피흡수합병") and P.EVENT_DELIST.search("지주회사") and P.EVENT_DELIST.search("코스닥시장 이전상장")
    assert P.EVENT_DELIST.search("유가증권시장 상장") and P.EVENT_DELIST.search("신청에 의한 상장폐지")
    assert P.EVENT_DELIST.search("해산 사유 발생") and P.EVENT_DELIST.search("증권거래소 상장")   # 재검증 J-2(삼성물산·제일모직)
    assert not P.EVENT_DELIST.search("감사의견 거절") and not P.EVENT_DELIST.search("자본전액잠식") and not P.EVENT_DELIST.search("최종부도 발생")


# ---------------------------------------------------------------- M1·M-5·M-6 원장·태그·잠금

@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LEDGER", str(tmp_path / "wt"))
    monkeypatch.setattr(P, "SHADOW", str(tmp_path / "shadow"))
    monkeypatch.setattr(P, "PRIVATE", str(tmp_path / "private"))
    monkeypatch.setattr(P, "ROOT", str(tmp_path / "norepo"))
    return tmp_path


def test_ledger_double_write_and_longer_wins(ledgers):
    for s in ("a", "b", "a"):
        P._append("RUNS.jsonl", {"train_sha": s})
    assert P.n_trials() == 2
    (ledgers / "wt" / "RUNS.jsonl").write_text("")
    assert P.n_trials() == 2


def test_missing_tag_allowed_only_with_empty_ledger(ledgers):
    P.integrity()                                                            # 태그 없음 + 원장 빔 → 경고만
    P._append("RUNS.jsonl", {"train_sha": "a"})
    with pytest.raises(SystemExit, match="태그"):
        P.integrity()                                                        # 태그를 지워 평가기를 고친 뒤 돌리는 길


def test_final_refuses_outside_main_session(ledgers, monkeypatch):
    monkeypatch.delenv("AR_FINAL", raising=False)
    with pytest.raises(SystemExit):
        P.final(1)
    monkeypatch.setenv("AR_FINAL", "main-session")
    monkeypatch.setenv("AR_LOOP", "1")
    with pytest.raises(SystemExit):
        P.final(1)


def test_final_refuses_after_three_openings_and_on_mixed_manifest(ledgers, monkeypatch):
    monkeypatch.setenv("AR_FINAL", "main-session")
    monkeypatch.delenv("AR_LOOP", raising=False)
    monkeypatch.setattr(P, "integrity", lambda check_cache=True: None)
    P._append("RUNS.jsonl", {"train_sha": "a", "manifest": "old"})
    with pytest.raises(SystemExit, match="MANIFEST"):
        P.final(1)                                                           # 루프 중간에 평가기가 바뀐 원장
    for _ in range(3):
        P._append("LOCKBOX.jsonl", {"event": "open"})
    with pytest.raises(SystemExit, match="이미 3회"):
        P.final(1)


def test_search_records_do_not_count_as_openings(ledgers):
    P._append("LOCKBOX.jsonl", {"event": "search"})
    P._append("LOCKBOX.jsonl", {"event": "search"})
    assert P.lockbox_used() == 0


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


def test_nw_t_signs():
    rng = np.random.default_rng(1)
    assert P.nw_t(rng.normal(0.5, 1, 2000)) > 5 and abs(P.nw_t(rng.normal(0, 1, 2000))) < 4
