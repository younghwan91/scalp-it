#!/usr/bin/env python3
"""짝꿍 라이브 규칙 — 저장 틱·호가 리플레이 최적화 (사전등록 75번).

``docs/research/manju/75-tick-replay-optimization-preregistration.md`` 의 절차를 그대로
코드로 옮긴다. 그 문서가 **판정 기준의 정본**이다 — 여기 숫자를 바꾸려면 문서부터
바꿔야 하는데, 측정 후에는 바꿀 수 없다.

단계
====

    replay   날짜별로 저장 틱·호가를 실제 PairDetector·RiskGuard 에 흘려 후보 신호와
             가드 청산(through·touch·무작위 대조군)을 캐시한다. 멱등(있으면 건너뜀).
    parity   2026-09-14 라이브 config 리플레이를 로그의 후보 3건과 대조(R-stored·R-log).
    grid     IS 8일에서 48셀을 평가·원장 기록·사전등록 목적함수로 1셀 선택.
    oos      선택 셀 + 라이브 셀을 OOS 8일에서 **한 번** 평가하고 7조건 게이트 판정.
    report   위 결과를 마크다운 표로.

재사용 원칙(복사 금지): 틱 로드·감지기 투입은 ``cli_pair_detect._fetch_ticks``·
``_feed``, 설정은 ``build_parser``/``build_config``/``build_guard`` 에 **cron 스크립트의
실제 플래그**를 넣어 만든다. 가드는 실제 ``RiskGuard`` 이고 여기서는 알림 순간을
관찰하는 얇은 하위클래스만 둔다.

안전: DB 는 SELECT 만. 주문 실행기를 만들지 않는다. 운영 로그에 쓰지 않는다(sink 없는
버스). 출력은 ``data/tick_replay_75/`` 캐시·리포트와 ``research/logs/75-tick-replay/``
시행 원장뿐.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/tick_replay_optimize.py replay --all
    uv run --extra pg python scripts/tick_replay_optimize.py parity
    uv run --extra pg python scripts/tick_replay_optimize.py grid
    uv run --extra pg python scripts/tick_replay_optimize.py oos
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import itertools
import json
import math
import os
import random
import re
import shlex
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from krx_quant_core.backtest.orderbook import roundtrip_bp
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.session import past_cutoff
from krx_quant_core.stats import trials as trial_ledger
from krx_quant_core.stats.metrics import paired_date_bootstrap
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

from scalp_it.cli_pair_detect import (
    _as_dt,
    _feed,
    _fetch_ticks,
    build_config,
    build_guard,
    build_parser,
)
from scalp_it.intraday.store import connect
from scalp_it.market_regime import evaluate_gate, previous_session
from scalp_it.realtime.alerts import AlertBus
from scalp_it.realtime.collector import quote_levels
from scalp_it.realtime.pair_detector import PairDetector
from scalp_it.realtime.risk_guard import RiskGuard
from scalp_it.ticksize import tick_size, ticks_in

ROOT = Path(__file__).resolve().parent.parent
LIVE_SH = ROOT / "scripts" / "pair_detect_daily.sh"
OUT = ROOT / "data" / "tick_replay_75"
LOGS_DIR = ROOT / "research" / "logs"
LABEL = "75-tick-replay"

# ---------------------------------------------------------------- 사전등록 상수(§1·§3·§7)
IS_DAYS = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27",
           "2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02"]
OOS_DAYS = ["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08",
            "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]
ALL_DAYS = IS_DAYS + OOS_DAYS
WEEKS = {"W35(08-24주)": ALL_DAYS[0:5], "W36(08-31주)": ALL_DAYS[5:10],
         "W37(09-07주)": ALL_DAYS[10:15], "W38(09-14주)": ALL_DAYS[15:16]}
PARITY_DAY = "2026-09-14"

CAPS = [30.0, 60.0, None]            # None = off
DEPTH = [True, False]
MIN_SL = [3.0, 0.0]
CUTOFF = ["1000", None]
REGIME = [5.0, None]
LIVE_CELL = (30.0, True, 3.0, "1000", 5.0)
N_PRIOR_TRIALS = 30                  # 54번 백로그의 누적 시험 수
N_PRIOR_SENS = 30 + 240              # + manju_optimize 240셀(병기용)
IS_MIN_N = 20
OOS_MIN_N = 30
N_BOOT = 10_000
SEED = 75
NULL_WINDOW_S = 60
COST_LEGACY = 0.0064                 # SECONDARY 64bp

LOGGED_PARITY = [  # 09-14 pair_alerts.log 후보 신호(전부 필터 배제)
    {"ts": "09:09:00", "leader": "411080", "follower": "140430",
     "reason": "depth_short", "entry": 2975.0, "rt_bp": None},
    {"ts": "09:41:16", "leader": "012210", "follower": "053260",
     "reason": "roundtrip_bp", "entry": 8160.0, "rt_bp": 40.3},
    {"ts": "12:05:27", "leader": "069540", "follower": "441270",
     "reason": "roundtrip_bp", "entry": 9140.0, "rt_bp": 32.8},
]


# ================================================================ 라이브 설정
def live_argv() -> list[str]:
    """cron 스크립트의 실제 실행줄에서 감지기 플래그만 뽑는다(주문·소켓·기간 제거)."""
    text = LIVE_SH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if "scalp-pair-detect" in ln and not ln.lstrip().startswith("#"))
    buf = []
    for ln in lines[start:]:
        buf.append(ln.rstrip().rstrip("\\"))
        if ">>" in ln:
            break
    cmd = " ".join(buf).split(">>")[0]
    cmd = re.sub(r'--duration\s+"[^"]*"', "", cmd)
    toks = shlex.split(cmd)
    toks = toks[toks.index("scalp-pair-detect") + 1:]
    drop = {"--live", "--live-order", "--confirm-real"}
    return [t for t in toks if t not in drop]


def live_setup(day: str) -> tuple[Any, Any, Any]:
    """(라이브 DetectorConfig, 리플레이 DetectorConfig(필터 끔), 리플레이 가드 config)."""
    args = build_parser().parse_args([*live_argv(), "--replay", day])
    live_cfg = build_config(args)
    run_cfg = replace(live_cfg, max_roundtrip_bp=0.0, min_sl_ticks=0.0)
    # 가드: 킬스위치·상한 끔(§2) — 포지션 수명 독립 → 게이트 사후 적용이 동치.
    args.daily_loss_limit = None
    args.max_consecutive_losses = 0
    args.profit_giveback = 0.0
    args.drawdown_krw = 0.0
    args.daily_profit_target = 0.0
    args.max_positions = 10**9
    args.no_stop_file = True
    args.cost_rate = 0.0
    args.size_multiplier = 1.0
    args.fill_basis = "through"
    guard_cfg = build_guard(args).config
    return live_cfg, run_cfg, guard_cfg


# ================================================================ DB 읽기(SELECT 만)
def _q(conn: Any, sql: str, params: Any = ()) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


QCOLS = ([f"bid{i}" for i in range(1, 11)] + [f"bidqty{i}" for i in range(1, 11)]
         + [f"ask{i}" for i in range(1, 11)] + [f"askqty{i}" for i in range(1, 11)])


def load_day(conn: Any, day: str) -> dict[str, Any]:
    d = date.fromisoformat(day)
    nxt = (d + timedelta(days=1)).isoformat()
    codes = [r[0] for r in _q(conn, "select distinct code from ticks where ts >= %s and ts < %s",
                              (day, nxt))]
    rows = _fetch_ticks(conn, sorted(codes), since=None, day=day)
    qrows = _q(conn, f"select code, ts, {', '.join(QCOLS)} from quotes "
                     "where ts >= %s and ts < %s order by ts", (day, nxt))
    pc = dict(_q(conn, "select distinct on (code) code, close from daily_bars "
                       "where code = any(%s) and date < %s and close > 0 "
                       "order by code, date desc", (codes, d)))
    meta = _q(conn, "select code, sector, market from stocks where code = any(%s)", (codes,))
    sectors = {c: s for c, s, _m in meta if s}
    markets = {c: m for c, _s, m in meta}
    snap = _q(conn, "select max(snapshot_date) from theme_members where snapshot_date < %s", (d,))
    themes: dict[str, set[str]] = {}
    if snap and snap[0][0]:
        for c, t in _q(conn, "select code, theme_code from theme_members where snapshot_date = %s",
                       (snap[0][0],)):
            themes.setdefault(str(c), set()).add(str(t))
    prev = previous_session(conn, d)
    jo = evaluate_gate(conn, prev, target=d, with_leader=False).turnover_jo
    return {"codes": codes, "rows": rows, "qrows": qrows, "prev_close": pc,
            "sectors": sectors, "markets": markets, "themes": themes,
            "theme_snapshot": str(snap[0][0]) if snap and snap[0][0] else None,
            "prev_session": str(prev), "kosdaq_jo": float(jo)}


# ================================================================ 관찰 가드
class RecordingGuard(RiskGuard):
    """실제 RiskGuard 에 알림 순간의 호가·대조군 기록만 얹는다(판정 로직은 부모 그대로)."""

    def __init__(self, config: Any, *, live_cfg: Any, null_pick: Any) -> None:
        super().__init__(config)
        self.live_cfg = live_cfg
        self.detector: PairDetector | None = None
        self.null_guard = RiskGuard(config)
        self.null_pick = null_pick
        self.cands: list[dict[str, Any]] = []

    def on_alert(self, alert: Any) -> Any:
        dec = super().on_alert(alert)
        if not dec.allowed:  # 킬·상한을 껐으므로 오면 안 된다
            raise RuntimeError(f"리플레이 가드가 알림을 막았다: {dec.reason}")
        book = self.detector._book.get(alert.follower) if self.detector else None  # noqa: SLF001
        bp = roundtrip_bp(book[0], book[1], self.live_cfg.order_krw) if book else None
        sl = self.live_cfg.stop_loss
        sl_ticks = ticks_in(abs(sl) * alert.entry, alert.entry) if sl else None
        rec = {"alert": alert, "rt_bp": bp, "sl_ticks": sl_ticks,
               "through": self.open_positions[-1], "touch": self.shadow_positions[-1],
               "null": None, "null_code": None}
        pick = self.null_pick(alert)
        if pick is not None:
            code, px = pick
            entry = px + self.live_cfg.entry_ticks_up * tick_size(px)
            na = replace(alert, follower=code, follower_name="", price=px, bid1=px, entry=entry,
                         stop_loss=(round(entry * (1 + sl), 4) if sl is not None else None),
                         disaster_line=(round(entry * (1 + self.live_cfg.disaster_line), 4)
                                        if self.live_cfg.disaster_line is not None else None),
                         rt_bp=-1.0)
            self.null_guard.on_alert(na)
            rec["null"] = self.null_guard.open_positions[-1]
            rec["null_code"] = code
        self.cands.append(rec)
        return dec

    def on_tick(self, code: str, ts: datetime, price: float) -> None:
        super().on_tick(code, ts, price)
        self.null_guard.on_tick(code, ts, price)

    def on_leader_state(self, leader_code: str, ts: datetime, state: str) -> None:
        super().on_leader_state(leader_code, ts, state)
        self.null_guard.on_leader_state(leader_code, ts, state)

    def flush_expired(self, now: datetime) -> None:
        super().flush_expired(now)
        self.null_guard.flush_expired(now)


# ================================================================ 리플레이
def _series(rows: list[tuple], qrows: list[tuple]) -> tuple[dict, dict, dict]:
    """종목별 (체결가 시계열, 틱 매수호가 시계열, 호가 매수1 시계열) — 초 단위 epoch."""
    px: dict[str, tuple[list, list]] = {}
    tb: dict[str, tuple[list, list]] = {}
    qb: dict[str, tuple[list, list]] = {}
    for code, ts, price, _v, bid, _s in rows:
        t = _as_dt(ts).timestamp()
        if price is not None:
            a = px.setdefault(code, ([], []))
            a[0].append(t)
            a[1].append(float(price))
        if bid:
            b = tb.setdefault(code, ([], []))
            b[0].append(t)
            b[1].append(float(bid))
    for r in qrows:
        if r[2]:
            b = qb.setdefault(r[0], ([], []))
            b[0].append(_as_dt(r[1]).timestamp())
            b[1].append(float(r[2]))
    return px, tb, qb


def exit_bid(code: str, t: datetime, tb: dict, qb: dict) -> float | None:
    """청산 시각의 매수1호가 — 틱 best_bid(≤t) 와 호가 bid1(<t) 중 늦은 것(동시면 틱)."""
    ts = t.timestamp()
    best: tuple[float, float] | None = None
    if code in tb:
        tt, bb = tb[code]
        i = bisect.bisect_right(tt, ts) - 1
        if i >= 0:
            best = (tt[i], bb[i])
    if code in qb:
        qt, qv = qb[code]
        j = bisect.bisect_left(qt, ts) - 1
        if j >= 0 and (best is None or qt[j] > best[0]):
            best = (qt[j], qv[j])
    return None if best is None else best[1]


def _pos(p: Any, basis: str, day: str, tb: dict, qb: dict) -> dict[str, Any]:
    if p is None:
        return None
    fill = p.through_fill_ts if basis == "through" else p.touch_fill_ts
    out = {"status": p.status, "reason": p.exit_reason, "entry": p.entry,
           "fill_ts": fill.isoformat() if fill else None,
           "exit_ts": p.exit_ts_actual.isoformat() if p.exit_ts_actual else None,
           "exit_px": p.exit_price, "mae": p.mae, "mfe": p.mfe, "exit_bid": None}
    if p.status == "closed" and p.exit_ts_actual is not None:
        out["exit_bid"] = exit_bid(p.follower, p.exit_ts_actual, tb, qb)
    return out


def log_registered_codes() -> set[str]:
    """R-log: 09-14 pair_detect.log 마지막 세션에서 감지기 편입으로 귀속되는 코드."""
    lines = (ROOT / "data" / "pair_detect.log").read_text(encoding="utf-8").splitlines()
    start = max(i for i, ln in enumerate(lines) if ln.startswith("# 소켓 직결 감지 시작"))
    pending: list[str] = []
    got: set[str] = set()
    for ln in lines[start:]:
        m = re.search(r"장중 구독 추가 \d+종목: (.+)$", ln)
        if m:
            pending += [c.strip() for c in m.group(1).split(",")]
            continue
        m = re.search(r"\[대장스캔\] 수집 자리 \d+개 선점 — (.+)$", ln)
        if m:
            got |= {c.strip() for c in m.group(1).split(",")}
            continue
        if "[대장스캔] 장중 편입" in ln or "[돌발재료] 장중 편입" in ln:
            got |= set(pending)
            pending = []
            m = re.search(r"\[돌발재료\] 장중 편입 \d+종목: ([0-9A-Z, ]+)", ln)
            if m:
                got |= {c.strip() for c in m.group(1).split(",")}
        elif "[수집채움]" in ln:
            pending = []
    return got


def log_universe(conn: Any, day: str, argv_args: Any) -> list[str]:
    """R-log 장전 유니버스 재구성(simnode dart.db 기준 — 트레이더와 다를 수 있다)."""
    from scalp_it.universe import (
        UniverseConfig,
        build_universe,
        fetch_first_days,
        fetch_recent_materials,
        pick_recent_listings,
    )
    d = date.fromisoformat(day)
    prev = previous_session(conn, d)
    pinned = tuple(pick_recent_listings(fetch_first_days(conn), asof=prev,
                                        within_days=argv_args.new_listing_days))
    mats = fetch_recent_materials(str(ROOT / argv_args.dart_db),
                                  since=prev - timedelta(days=argv_args.material_days), until=d)
    uni = build_universe(conn, prev, UniverseConfig(limit=argv_args.universe_size, pinned=pinned,
                                                    weight_material=argv_args.material_weight),
                         mats, None)
    return list(uni.pairable_codes)


def replay_day(day: str, *, mode: str = "stored", force: bool = False) -> Path:
    path = OUT / mode / f"{day}.json.gz"
    if path.exists() and not force:
        print(f"[캐시] {path} 있음 — 건너뜀", flush=True)
        return path
    t0 = time.perf_counter()
    live_cfg, run_cfg, guard_cfg = live_setup(day)
    conn = connect(os.environ.get("KR_QUANT_DB"))
    data = load_day(conn, day)
    reg_codes = data["codes"]
    reg_note = "R-stored: 그날 틱 저장 전 종목"
    if mode == "log":
        args = build_parser().parse_args([*live_argv(), "--replay", day])
        uni = log_universe(conn, day, args)
        scan = log_registered_codes()
        reg_codes = sorted((set(uni) | scan) & set(data["codes"]))
        reg_note = (f"R-log: 장전유니버스 재구성 {len(uni)} ∪ 로그 편입귀속 {len(scan)} "
                    f"∩ 저장 {len(reg_codes)}")
    conn.close()
    rows, qrows = data["rows"], data["qrows"]
    print(f"[{day}/{mode}] 종목 {len(data['codes'])}(등록 {len(reg_codes)}) · 틱 {len(rows):,} · "
          f"호가 {len(qrows):,} · 로드 {time.perf_counter() - t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit(f"{day} 틱 없음")
    px, tb, qb = _series(rows, qrows)

    def null_pick(alert: Any) -> tuple[str, float] | None:
        ts = alert.ts.timestamp()
        pool = []
        for code, (tt, pp) in px.items():
            if code in (alert.leader, alert.follower):
                continue
            i = bisect.bisect_right(tt, ts) - 1
            if i >= 0 and ts - tt[i] <= NULL_WINDOW_S:
                pool.append((code, pp[i]))
        if not pool:
            return None
        pool.sort()
        rng = random.Random(f"{SEED}|{alert.ts.isoformat()}|{alert.leader}|{alert.follower}")
        return rng.choice(pool)

    themes = {c: frozenset(t) for c, t in data["themes"].items()}
    detector = PairDetector([], run_cfg, prev_close=data["prev_close"], sectors=data["sectors"],
                            universe=reg_codes, themes=themes)
    guard = RecordingGuard(guard_cfg, live_cfg=live_cfg, null_pick=null_pick)
    guard.detector = detector
    bus = AlertBus([], dry_run=False)          # sink 없음 — 운영 로그 미접촉

    # 초 단위로 틱(_feed) → 같은 초 호가(on_quote) 순서(§2: 선견 방지).
    qi, nq = 0, len(qrows)
    i, n = 0, len(rows)
    n_alerts = 0
    while i < n or qi < nq:
        t_tick = _as_dt(rows[i][1]) if i < n else None
        t_q = _as_dt(qrows[qi][1]) if qi < nq else None
        if t_tick is not None and (t_q is None or t_tick <= t_q):
            j = i
            while j < n and _as_dt(rows[j][1]) == t_tick:
                j += 1
            n_alerts += _feed(detector, bus, rows[i:j], guard=guard)
            i = j
            while qi < nq and _as_dt(qrows[qi][1]) == t_tick:
                q = qrows[qi]
                bids, asks = quote_levels(dict(zip(QCOLS, q[2:], strict=True)))
                detector.on_quote(str(q[0]), _as_dt(q[1]), q[2], bids=bids, asks=asks)
                qi += 1
        else:
            q = qrows[qi]
            bids, asks = quote_levels(dict(zip(QCOLS, q[2:], strict=True)))
            detector.on_quote(str(q[0]), _as_dt(q[1]), q[2], bids=bids, asks=asks)
            qi += 1
    guard.flush_expired(_as_dt(rows[-1][1]))

    cands = []
    for rec in guard.cands:
        a = rec["alert"]
        bp = rec["rt_bp"]
        cands.append({
            "day": day, "ts": a.ts.isoformat(), "leader": a.leader, "follower": a.follower,
            "trigger": a.trigger, "price": a.price, "entry": a.entry,
            "leader_ret": a.leader_ret, "market": data["markets"].get(a.follower, "코스닥"),
            "rt_bp": (None if bp is None else ("inf" if math.isinf(bp) else float(bp))),
            "sl_ticks": rec["sl_ticks"],
            "through": _pos(rec["through"], "through", day, tb, qb),
            "touch": _pos(rec["touch"], "touch", day, tb, qb),
            "null_code": rec["null_code"],
            "null": _pos(rec["null"], "through", day, tb, qb),
        })
    payload = {"day": day, "mode": mode, "registration": reg_note,
               "n_codes": len(data["codes"]), "n_registered": len(reg_codes),
               "ticks": len(rows), "quotes": len(qrows), "kosdaq_jo": data["kosdaq_jo"],
               "prev_session": data["prev_session"], "theme_snapshot": data["theme_snapshot"],
               "live_argv": live_argv(), "candidates": cands,
               "elapsed_s": round(time.perf_counter() - t0, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, default=str)
    tmp.replace(path)
    print(f"[{day}/{mode}] 후보 {len(cands)} · {payload['elapsed_s']}s → {path}", flush=True)
    return path


def load_cache(day: str, mode: str = "stored") -> dict[str, Any]:
    with gzip.open(OUT / mode / f"{day}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


# ================================================================ 셀 평가
def cell_name(cell: tuple) -> str:
    cap, depth, msl, cut, reg = cell
    return (f"cap={'off' if cap is None else int(cap)} depth={'on' if depth else 'off'} "
            f"minsl={int(msl)} cutoff={cut or 'none'} regime={'5조' if reg else 'off'}")


def cell_config(cell: tuple) -> dict[str, Any]:
    cap, depth, msl, cut, reg = cell
    return {"max_roundtrip_bp": cap, "depth_short_filter": depth, "min_sl_ticks": msl,
            "no_entry_after": cut, "halt_below_jo": reg, "program_gate": "off(replay)",
            "fill_basis_primary": "through", "cost_primary": "exit@bid+round_trip_cost",
            "registration": "R-stored", "prereg": "docs/research/manju/75"}


def n_relaxed(cell: tuple) -> int:
    return sum(a != b for a, b in zip(cell, LIVE_CELL, strict=True))


def live_filter_reason(c: dict, cap: float | None = 30.0, depth: bool = True,
                       msl: float = 3.0) -> str | None:
    bp = c["rt_bp"]
    if bp == "inf":
        if depth:
            return "depth_short"
    elif bp is not None and cap is not None and bp > cap:
        return "roundtrip_bp"
    if msl > 0 and c["sl_ticks"] is not None and c["sl_ticks"] < msl:
        return "min_sl_ticks"
    return None


def stage(c: dict, cell: tuple, jo: float) -> str:
    """후보가 어느 단계에서 떨어지는가: filter | cutoff | regime | unfilled | trade."""
    cap, depth, msl, cut, reg = cell
    if live_filter_reason(c, cap, depth, msl):
        return "filter"
    if cut and past_cutoff(datetime.fromisoformat(c["ts"]).strftime("%H%M"), cut):
        return "cutoff"
    if reg and jo < reg:
        return "regime"
    return "trade"


def trade_return(c: dict, basis: str = "through", cost: str = "primary",
                 key: str | None = None) -> float | None:
    p = c[key or basis]
    if not p or p["status"] != "closed" or p["exit_px"] is None:
        return None
    entry = p["entry"]
    day = date.fromisoformat(c["day"])
    mkt = c["market"] if c.get("market") in ("코스닥", "거래소") else "코스닥"
    mkt = "KOSPI" if mkt == "거래소" else "KOSDAQ"
    if cost == "gross":
        return p["exit_px"] / entry - 1.0
    if cost == "primary":
        bid = p["exit_bid"] if p["exit_bid"] else p["exit_px"]
        return bid / entry - 1.0 - round_trip_cost(day, mkt, commission_rate=0.00015)
    if cost == "secondary":
        bp = c["rt_bp"]
        if bp is None or bp == "inf":
            return None
        return p["exit_px"] / entry - 1.0 - COST_LEGACY - bp / 1e4
    raise ValueError(cost)


def trades(caches: dict[str, dict], days: list[str], cell: tuple, *, basis: str = "through",
           cost: str = "primary") -> pd.DataFrame:
    out = []
    for d in days:
        cache = caches[d]
        for c in cache["candidates"]:
            if stage(c, cell, cache["kosdaq_jo"]) != "trade":
                continue
            r = trade_return(c, basis, cost)
            if r is None:
                continue
            nr = trade_return(c, "through", cost, key="null")
            out.append({"day": d, "ts": c["ts"], "leader": c["leader"],
                        "follower": c["follower"], "ret": r, "null_ret": nr,
                        "reason": c[basis]["reason"]})
    return pd.DataFrame(out, columns=["day", "ts", "leader", "follower", "ret", "null_ret",
                                      "reason"])


def daymean_objective(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df.groupby("day")["ret"].mean().mean())


def day_cluster_ci(df: pd.DataFrame, days: list[str]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    by = {d: df.loc[df.day == d, "ret"].to_numpy() for d in days}
    means = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(days), len(days))
        arr = np.concatenate([by[days[k]] for k in pick])
        if arr.size:
            means.append(arr.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gate(df: pd.DataFrame, caches: dict[str, dict], cell: tuple) -> dict[str, Any]:
    R = df["ret"].to_numpy()
    n = len(R)
    res: dict[str, Any] = {"n": n}
    if n == 0:
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    res["mean"] = float(R.mean())
    res["median"] = float(np.median(R))
    res["win_rate"] = float((R > 0).mean())
    res["ci"] = day_cluster_ci(df, OOS_DAYS)
    k = max(1, math.ceil(0.01 * n))
    trimmed = np.sort(R)[:-k] if n > k else np.array([])
    res["trim_mean"] = float(trimmed.mean()) if trimmed.size else float("nan")
    full = trades(caches, ALL_DAYS, cell)
    weeks = {w: (float(full.loc[full.day.isin(ds), "ret"].mean())
                 if (full.day.isin(ds)).any() else None,
                 int(full.day.isin(ds).sum())) for w, ds in WEEKS.items()}
    res["weeks"] = weeks
    res["neg_weeks"] = sum(1 for m, _c in weeks.values() if m is not None and m < 0)
    dsr = deflated_sharpe_from_sample(R, len(cells()) + N_PRIOR_TRIALS)
    dsr_s = deflated_sharpe_from_sample(R, len(cells()) + N_PRIOR_SENS)
    res["dsr"] = dsr
    res["dsr_sens"] = dsr_s
    nd = df.dropna(subset=["null_ret"])
    if nd.empty:
        res["null"] = None
    else:
        a = nd.groupby("day")["ret"].mean().rename("mean_excess").reset_index()
        b = nd.groupby("day")["null_ret"].mean().rename("mean_excess").reset_index()
        a.columns = b.columns = ["date", "mean_excess"]
        pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
        res["null"] = {"n_pairs": len(nd), "strategy_mean": float(nd["ret"].mean()),
                       "null_mean": float(nd["null_ret"].mean()), **pb,
                       "p": 1.0 - pb["p_greater"]}
    conds = {
        "1 평균>0·중앙>0": res["mean"] > 0 and res["median"] > 0,
        "2 일클러스터 CI하한>0": res["ci"][0] > 0,
        "3 상위1%제거 평균>0": bool(trimmed.size) and res["trim_mean"] > 0,
        "4 음수 주 ≤1": res["neg_weeks"] <= 1,
        "5 DSR확률>0.95": (dsr.get("prob_deflated_sharpe") or 0) > 0.95,
        "6 무작위대조 p<0.05": bool(res["null"]) and res["null"]["p"] < 0.05,
        "7 OOS n≥30": n >= OOS_MIN_N,
    }
    res["conds"] = conds
    if n < OOS_MIN_N:
        res["verdict"] = "표본 부족 — 판정 불가"
    else:
        res["verdict"] = "채택(그림자 비교 단계로)" if all(conds.values()) else "기각"
    return res


def cells() -> list[tuple]:
    return list(itertools.product(CAPS, DEPTH, MIN_SL, CUTOFF, REGIME))


# ================================================================ 명령
def cmd_replay(ns: argparse.Namespace) -> None:
    days = ALL_DAYS if ns.all else ns.days
    for d in days:
        replay_day(d, mode=ns.mode, force=ns.force)


def _match(c: dict, logged: dict) -> bool:
    ts = datetime.fromisoformat(c["ts"])
    lt = datetime.fromisoformat(f"{PARITY_DAY} {logged['ts']}")
    return (c["leader"] == logged["leader"] and c["follower"] == logged["follower"]
            and abs((ts - lt).total_seconds()) <= 5)


def cmd_parity(ns: argparse.Namespace) -> str:
    lines = ["### 패리티 체크 — 2026-09-14 라이브 config", ""]
    for mode in ("stored", "log"):
        replay_day(PARITY_DAY, mode=mode, force=False)
        cache = load_cache(PARITY_DAY, mode)
        cs = cache["candidates"]
        lines.append(f"**{mode}** — {cache['registration']} · 후보 {len(cs)}건")
        lines.append("")
        lines.append("| 로그 신호 | 재현 | 리플레이 시각 | 진입 | rt_bp | 사유 | 판정 |")
        lines.append("|---|---|---|--:|--:|---|---|")
        matched = set()
        for lg in LOGGED_PARITY:
            hit = [c for c in cs if _match(c, lg)]
            if not hit:
                lines.append(f"| {lg['ts']} {lg['leader']}→{lg['follower']} {lg['reason']} "
                             f"| ✗ | — | — | — | — | 불일치(후보 없음) |")
                continue
            c = hit[0]
            matched.add(id(c))
            reason = live_filter_reason(c)
            bp = c["rt_bp"]
            bp_s = "inf" if bp == "inf" else ("—" if bp is None else f"{bp:.1f}")
            ok = (reason == lg["reason"] and abs(c["entry"] - lg["entry"]) < 1e-6
                  and (lg["rt_bp"] is None or (bp not in (None, "inf")
                                               and abs(bp - lg["rt_bp"]) <= 5)))
            lines.append(f"| {lg['ts']} {lg['leader']}→{lg['follower']} {lg['reason']} | ✓ "
                         f"| {c['ts'][11:19]} | {c['entry']:,.0f} | {bp_s} | {reason or '통과'} "
                         f"| {'일치' if ok else '부분일치'} |")
        extra = [c for c in cs if id(c) not in matched and not any(_match(c, lg)
                                                                   for lg in LOGGED_PARITY)]
        lines.append("")
        lines.append(f"로그에 없는 추가 후보 {len(extra)}건:")
        lines.append("")
        if extra:
            lines.append("| 시각 | 대장→후속 | 진입 | rt_bp | 라이브 필터 | 라이브 게이트로 가려졌나 |")
            lines.append("|---|---|--:|--:|---|---|")
            for c in extra:
                r = live_filter_reason(c)
                hhmm = c["ts"][11:16].replace(":", "")
                hidden = ("—" if r else
                          ("10:00 컷오프 후(로그 안 남음)" if past_cutoff(hhmm, "1000")
                           else "프로그램 차단 구간이면 로그 안 남음(09:00~09:18·09:20~)"))
                bp = c["rt_bp"]
                bp_s = "inf" if bp == "inf" else ("—" if bp is None else f"{bp:.1f}")
                lines.append(f"| {c['ts'][11:19]} | {c['leader']}→{c['follower']} | "
                             f"{c['entry']:,.0f} | {bp_s} | {r or '통과'} | {hidden} |")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "parity.md").write_text(text, encoding="utf-8")
    print(text)
    return text


def cmd_grid(ns: argparse.Namespace) -> None:
    caches = {d: load_cache(d) for d in IS_DAYS}
    rows = []
    for cell in cells():
        cfg = cell_config(cell)
        trial_ledger.record_trial(LABEL, cfg, logs_dir=LOGS_DIR, note="IS grid cell (prereg 75 §3)")
        df = trades(caches, IS_DAYS, cell)
        dft = trades(caches, IS_DAYS, cell, basis="touch")
        rows.append({"cell": cell, "name": cell_name(cell), "n": len(df),
                     "obj": daymean_objective(df),
                     "mean": float(df.ret.mean()) if len(df) else float("nan"),
                     "median": float(df.ret.median()) if len(df) else float("nan"),
                     "n_touch": len(dft),
                     "mean_touch": float(dft.ret.mean()) if len(dft) else float("nan"),
                     "relaxed": n_relaxed(cell), "live": cell == LIVE_CELL})
    elig = [r for r in rows if r["n"] >= IS_MIN_N and not math.isnan(r["obj"])]
    sel = None
    if elig:
        sel = max(elig, key=lambda r: (round(r["obj"], 6), -r["relaxed"]))
    out = {"rows": [{**r, "cell": list(r["cell"])} for r in rows],
           "selected": list(sel["cell"]) if sel else None,
           "n_trials_ledger": trial_ledger.count_trials(LABEL, logs_dir=LOGS_DIR)}
    (OUT / "grid_is.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                 default=str), encoding="utf-8")
    print(f"IS 자격 셀 {len(elig)}/{len(rows)} · 선택: {sel['name'] if sel else '없음'}")
    for r in sorted(rows, key=lambda r: -(r["obj"] if not math.isnan(r["obj"]) else -9))[:10]:
        print(f"  {r['name']:<60} n={r['n']:>3} obj={r['obj']:+.4%}")


def cmd_oos(ns: argparse.Namespace) -> None:
    grid = json.loads((OUT / "grid_is.json").read_text(encoding="utf-8"))
    caches = {d: load_cache(d) for d in ALL_DAYS}
    targets = [("live", LIVE_CELL)]
    if grid["selected"] is not None:
        sel = tuple(grid["selected"])
        if sel != LIVE_CELL:
            targets.insert(0, ("selected", sel))
    res = {}
    for label, cell in targets:
        df = trades(caches, OOS_DAYS, cell)
        g = gate(df, caches, cell)
        g["cell"] = list(cell)
        g["name"] = cell_name(cell)
        g["touch"] = _summ(trades(caches, OOS_DAYS, cell, basis="touch"))
        g["gross"] = _summ(trades(caches, OOS_DAYS, cell, cost="gross"))
        g["secondary"] = _summ(trades(caches, OOS_DAYS, cell, cost="secondary"))
        g["trades"] = df.to_dict("records")
        res[label] = g
    if grid["selected"] is not None and tuple(grid["selected"]) == LIVE_CELL:
        res["selected"] = res["live"]
    (OUT / "oos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
    for k, g in res.items():
        print(k, g["name"], g["n"], g.get("mean"), g.get("median"), g["verdict"])


def _summ(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    R = df.ret.to_numpy()
    return {"n": len(R), "mean": float(R.mean()), "median": float(np.median(R)),
            "win_rate": float((R > 0).mean())}


def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.2f}%"


def cmd_report(ns: argparse.Namespace) -> None:
    grid = json.loads((OUT / "grid_is.json").read_text(encoding="utf-8"))
    oos = json.loads((OUT / "oos.json").read_text(encoding="utf-8"))
    caches = {d: load_cache(d) for d in ALL_DAYS}
    L: list[str] = []
    L += ["### IS 그리드 — 상위 10 (목적함수 = IS 일평균 순수익, PRIMARY·through) + 라이브", "",
          "| # | 셀 | IS n | 일평균(목적) | 건평균 | 중앙 | touch n | touch 평균 | 자격(n≥20) |",
          "|--:|---|--:|--:|--:|--:|--:|--:|---|"]
    rows = grid["rows"]
    ranked = sorted(rows, key=lambda r: -(r["obj"] if r["obj"] == r["obj"] else -9))
    for i, r in enumerate(ranked[:10], 1):
        L.append(f"| {i} | {r['name']}{' **(라이브)**' if r['live'] else ''} | {r['n']} | "
                 f"{_pct(r['obj'])} | {_pct(r['mean'])} | {_pct(r['median'])} | {r['n_touch']} | "
                 f"{_pct(r['mean_touch'])} | {'예' if r['n'] >= IS_MIN_N else '아니오'} |")
    live = next(r for r in rows if r["live"])
    rank_live = ranked.index(live) + 1
    if rank_live > 10:
        L.append(f"| {rank_live} | {live['name']} **(라이브)** | {live['n']} | {_pct(live['obj'])} | "
                 f"{_pct(live['mean'])} | {_pct(live['median'])} | {live['n_touch']} | "
                 f"{_pct(live['mean_touch'])} | {'예' if live['n'] >= IS_MIN_N else '아니오'} |")
    pos = sum(1 for r in rows if r["obj"] == r["obj"] and r["obj"] > 0)
    elig = [r for r in rows if r["n"] >= IS_MIN_N]
    L += ["", f"48셀 중 IS 목적함수 양수 **{pos}**셀 · 자격(n≥20) {len(elig)}셀 · "
              f"선택: `{cell_name(tuple(grid['selected'])) if grid['selected'] else '없음'}` · "
              f"원장 시행 수 {grid['n_trials_ledger']}", ""]
    L += ["### OOS — 선택 셀 vs 라이브 (PRIMARY·through, 한 번)", "",
          "| 셀 | n | 평균 | 중앙 | 승률 | 일클러스터 95% CI | 상위1%제거 평균 | DSR 확률(N=78) | DSR(N=318) | 판정 |",
          "|---|--:|--:|--:|--:|---|--:|--:|--:|---|"]
    for k in ("selected", "live"):
        g = oos[k]
        if not g.get("n"):
            L.append(f"| {k} {g['name']} | 0 | | | | | | | | {g['verdict']} |")
            continue
        L.append(f"| {k}: {g['name']} | {g['n']} | {_pct(g['mean'])} | {_pct(g['median'])} | "
                 f"{g['win_rate'] * 100:.0f}% | [{_pct(g['ci'][0])}, {_pct(g['ci'][1])}] | "
                 f"{_pct(g['trim_mean'])} | {g['dsr']['prob_deflated_sharpe']:.4f} | "
                 f"{g['dsr_sens']['prob_deflated_sharpe']:.4f} | **{g['verdict']}** |")
    L += ["", "게이트 조건(선택 셀):", ""]
    for c, ok in oos["selected"]["conds"].items():
        L.append(f"- {'✓' if ok else '✗'} {c}")
    L += ["", "### 무작위 대조군 (같은 시각·같은 대장·같은 청산, 구독 중이던 다른 종목)", "",
          "| 셀 | 짝 n | 날짜 | 전략 평균 | 대조군 평균 | 차이 | 95% CI | p = P(차이≤0) |",
          "|---|--:|--:|--:|--:|--:|---|--:|"]
    for k in ("selected", "live"):
        nu = oos[k].get("null")
        if nu:
            L.append(f"| {k} | {nu['n_pairs']} | {nu['n_dates']} | {_pct(nu['strategy_mean'])} | "
                     f"{_pct(nu['null_mean'])} | {_pct(nu['mean_diff'])} | "
                     f"[{_pct(nu['ci_low'])}, {_pct(nu['ci_high'])}] | {nu['p']:.3f} |")
    L += ["", "### 주별 폴드 (16일 전체, PRIMARY·through)", "",
          "| 셀 | " + " | ".join(WEEKS) + " | 음수 주 |", "|---|" + "--:|" * (len(WEEKS) + 1)]
    for k in ("selected", "live"):
        w = oos[k]["weeks"]
        L.append(f"| {k} | " + " | ".join(f"{_pct(m)} (n={c})" for m, c in w.values())
                 + f" | {oos[k]['neg_weeks']} |")
    L += ["", "### 비용 민감도 (OOS)", "",
          "| 셀 | 세전(체결가 청산) | PRIMARY(매수호가 청산+23bp) | touch PRIMARY | SECONDARY(64bp+1천만원 충격) |",
          "|---|---|---|---|---|"]
    for k in ("selected", "live"):
        g = oos[k]
        f = lambda s: f"{_pct(s.get('mean'))} / 중앙 {_pct(s.get('median'))} (n={s['n']})"  # noqa: E731
        prim = {"n": g["n"], "mean": g.get("mean"), "median": g.get("median")}
        L.append(f"| {k} | {f(g['gross'])} | {f(prim)} | {f(g['touch'])} | {f(g['secondary'])} |")
    sel_rows = [r for r in oos["selected"]["trades"]]
    reasons: dict[str, list[float]] = {}
    for r in sel_rows:
        reasons.setdefault(r["reason"], []).append(r["ret"])
    L += ["", "선택 셀 OOS 청산 사유: " + " · ".join(
        f"{k} {len(v)}건(평균 {_pct(float(np.mean(v)))})" for k, v in
        sorted(reasons.items(), key=lambda kv: -len(kv[1]))), ""]
    # 오늘 후보 3건(맥락용, 필터·게이트 무시)
    L += ["### 맥락 — 09-14 로그 후보 3건이 들어갔다면 (필터·게이트 무시, PRIMARY·through)", "",
          "| 로그 신호 | 리플레이(모드) | 체결 | 청산 사유 | 세전 | PRIMARY |", "|---|---|---|---|--:|--:|"]
    for lg in LOGGED_PARITY:
        found = None
        for mode in ("stored", "log"):
            hit = [c for c in load_cache(PARITY_DAY, mode)["candidates"] if _match(c, lg)]
            if hit:
                found = (mode, hit[0])
                break
        label = f"{lg['ts']} {lg['leader']}→{lg['follower']}"
        if not found:
            L.append(f"| {label} | 재현 안 됨 | — | — | — | — |")
            continue
        mode, c = found
        p = c["through"]
        L.append(f"| {label} | {mode} | {'예' if p['status'] == 'closed' else p['status']} | "
                 f"{p['reason']} | {_pct(trade_return(c, cost='gross'))} | {_pct(trade_return(c))} |")
    L.append("")
    # 퍼널
    L += ["### 왜 거래가 적은가 — 일자별 퍼널 (라이브 셀)", "",
          "| 날짜 | 구분 | 저장종목 | 전일 코스닥 | 후보 | 필터배제 | 컷오프 | 국면정지 | 미체결 | 거래 | 필터 전부 끄면 거래 |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    open_cell = (None, False, 0.0, None, None)
    tot = collections.Counter()
    for d in ALL_DAYS:
        cache = caches[d]
        cnt = collections.Counter()
        for c in cache["candidates"]:
            stg = stage(c, LIVE_CELL, cache["kosdaq_jo"])
            if stg == "trade" and trade_return(c) is None:
                stg = "unfilled"
            cnt[stg] += 1
        n_open = sum(1 for c in cache["candidates"] if trade_return(c) is not None
                     and stage(c, open_cell, cache["kosdaq_jo"]) == "trade")
        tot.update(cnt)
        tot["cands"] += len(cache["candidates"])
        tot["open"] += n_open
        L.append(f"| {d} | {'IS' if d in IS_DAYS else 'OOS'} | {cache['n_codes']} | "
                 f"{cache['kosdaq_jo']:.2f}조 | {len(cache['candidates'])} | {cnt['filter']} | "
                 f"{cnt['cutoff']} | {cnt['regime']} | {cnt['unfilled']} | {cnt['trade']} | {n_open} |")
    L.append(f"| 합계 | | | | {tot['cands']} | {tot['filter']} | {tot['cutoff']} | {tot['regime']} | "
             f"{tot['unfilled']} | {tot['trade']} | {tot['open']} |")
    reasons_f = collections.Counter(live_filter_reason(c) for d in ALL_DAYS
                                    for c in caches[d]["candidates"] if live_filter_reason(c))
    L += ["", "라이브 필터 배제 사유(16일): " + " · ".join(f"{k} {v}" for k, v in reasons_f.most_common()), ""]
    L += ["### 셀별 일자별 거래 수 (48셀, PRIMARY·through 체결 기준)", "",
          "| 셀 | " + " | ".join(d[5:] for d in ALL_DAYS) + " | 합 |", "|---|" + "--:|" * (len(ALL_DAYS) + 1)]
    for cell in cells():
        df = trades(caches, ALL_DAYS, cell)
        per = df.groupby("day").size().to_dict()
        L.append(f"| {cell_name(cell)} | " + " | ".join(str(per.get(d, 0)) for d in ALL_DAYS)
                 + f" | {len(df)} |")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("days", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--mode", default="stored", choices=["stored", "log"])
    r.add_argument("--force", action="store_true")
    sub.add_parser("parity")
    sub.add_parser("grid")
    sub.add_parser("oos")
    sub.add_parser("report")
    ns = p.parse_args(argv)
    {"replay": cmd_replay, "parity": cmd_parity, "grid": cmd_grid, "oos": cmd_oos, "report": cmd_report}[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
