#!/usr/bin/env python3
"""짝꿍 감지기 파라미터 스윕 — 틱을 **한 번만** 로드하고 조합마다 감지기·가드만 새로.

왜 필요한가
===========

``scalp-pair-detect --replay YYYY-MM-DD`` 는 조합 하나당 DB 에서 그날 틱(08-24 기준
179만 건)을 다시 읽는다. 수십 조합에는 못 쓴다. 여기서는 틱을 한 번 읽어 메모리에
두고, 조합마다 :class:`PairDetector` 와 :class:`RiskGuard` 만 새로 만들어 **같은 틱열을
다시 흘린다**. 틱 로드·정렬·감지기에 먹이는 순서·필드 매핑은 리플레이 경로 그대로다 —
``cli_pair_detect`` 의 ``_fetch_ticks``/``_feed``/``_as_dt`` 를 그대로 import 해서 쓰기
때문이다(복사본을 만들면 스윕 결과가 실전과 무관해진다).

⚠ 이 스크립트가 무엇을 위한 게 **아닌지**
=========================================

**최적값을 고르는 도구가 아니다.** 틱 데이터가 08-24(80종목) 하루뿐이라 하루에 수십
조합을 맞추면 반드시 좋아 보이는 조합이 나온다 — 그건 엣지가 아니라 노이즈다. 여기서
볼 것은 **방향** 하나뿐이다: *필터를 풀면 신호가 늘면서 성과가 따라오는가, 아니면
신호만 늘고 무너지는가.* 표의 최고 행을 파라미터로 채택하지 마라.

회계·안전
=========

* 손익 회계는 :class:`RiskGuard` 가 한다. touch(낙관)/through(보수)를 **독립 그림자**로
  굴리므로 두 가정을 모두 보고한다.
* 킬스위치·연속손절·정지파일은 스윕에서 **끈다**. 조합마다 다른 시점에 표집이 잘리면
  조합 간 비교가 불가능해지기 때문이다(출력 헤더에도 명시한다).
* 가드 ``cost_rate`` 는 0 — 즉 회계는 **세전**이고, 비용은 집계 단계에서
  ``scripts/verify_pair.py`` 의 COST_LO/BASE/HI(34/64/94bp)를 차감해 병기한다.
* **읽기 전용**: DB 는 SELECT 만, 주문 실행기 없음, 운영 로그(``data/pair_positions.log``·
  ``data/pair_alerts.log``)에 아무것도 쓰지 않는다(sink 없는 버스). 출력은 ``--out``
  JSONL 과 stdout 뿐.

사용법
======

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/pair_sweep.py --replay 2026-08-24 \\
        --grid 'surge_window=10,15,30' --grid 'cosurge_min=0.005,0.002,0.001,0' \\
        --out data/pair_sweep.jsonl

    # 내가 지금 추가 중인 축까지 (없는 필드는 경고만 내고 건너뛴다)
    uv run --extra pg python scripts/pair_sweep.py --replay 2026-08-24 \\
        --grid 'accel_min_ratio=0,1.2,1.5' --grid 'volume_burst_ratio=0,2,3' --yes

``--grid NAME=v1,v2,...`` 를 여러 번 주면 축의 데카르트곱을 돈다. 값 ``none``/``off``
(그리고 Optional 필드의 ``0``)은 "끔"으로 읽는다 — ``--stop-loss 0`` 이 손절 없음인
CLI 관례와 같다. ``DetectorConfig`` 에 없는 이름은 건너뛰며 경고한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st
import sys
import time
from collections import Counter
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_daily_summary import _gross, _scenario  # 집계 산술 재사용
from verify_pair import COST_BASE, COST_HI, COST_LO  # 비용 상수(재정의 금지)

from scalp_it.cli_pair_detect import (
    _as_dt,
    _feed,
    _fetch_ticks,
    resolve_pairs,
)
from scalp_it.intraday.store import connect
from scalp_it.realtime.alerts import AlertBus
from scalp_it.realtime.pair_detector import (
    DetectorConfig,
    Pair,
    PairDetector,
)
from scalp_it.realtime.risk_guard import RiskGuard, RiskGuardConfig

COSTS = {"net34": COST_LO, "net64": COST_BASE, "net94": COST_HI}
DEFAULT_OUT = "data/pair_sweep.jsonl"
_NONE_WORDS = {"none", "off", "null", "-", "없음"}

WARNING = (
    "⚠ 최적값 선택 금지 — 틱이 하루뿐이라 수십 조합 중 '좋아 보이는' 행은 거의 확실히 "
    "노이즈다. 볼 것은 방향뿐: 필터를 풀면 신호가 늘면서 성과가 따라오는가, "
    "아니면 신호만 늘고 무너지는가."
)
GUARD_NOTE = (
    "킬스위치 OFF(일일손실한도 0 · 연속손절 0 · 정지파일 없음), 동시보유 상한 넉넉히 — "
    "조합마다 다른 시점에 표집이 잘리면 비교가 불가능하기 때문. 실전 가드와 다르다."
)

# 스윕 대상으로 문서화한 축(존재 여부는 실행 시 DetectorConfig 에서 확인한다).
SWEEP_HINT = [
    "surge_window", "surge_ret", "cosurge_min", "premove_max",
    "take_profit", "stop_loss", "time_exit", "min_sl_ticks",
    "accel_min_ratio", "volume_burst_ratio",
]


# --------------------------------------------------------------------- 그리드

def _config_fields() -> dict[str, Any]:
    return {f.name: f for f in fields(DetectorConfig)}


def _is_optional(field: Any) -> bool:
    """``float | None`` 처럼 None 을 받는 필드인가(문자열 annotation 기준)."""
    return "None" in str(field.type)


def coerce_value(field: Any, raw: str) -> Any:
    """그리드 값 하나를 필드 타입에 맞게 변환. ``none``/Optional 의 ``0`` → None."""
    s = raw.strip()
    if s.lower() in _NONE_WORDS:
        if not _is_optional(field):
            raise ValueError(f"{field.name} 은 None 을 받지 않는다: {raw!r}")
        return None
    v = float(s)
    if _is_optional(field) and v == 0.0:
        return None                    # CLI 관례(--stop-loss 0 = 손절 없음)와 동일
    if str(field.type).strip().startswith("int"):
        return int(v)
    return v


def parse_grid(specs: list[str],
               known: dict[str, Any] | None = None) -> tuple[list[tuple[str, list[Any]]],
                                                             list[str]]:
    """``['surge_window=10,15']`` → ``([('surge_window', [10.0, 15.0])], 경고들)``.

    ``DetectorConfig`` 에 없는 이름은 **건너뛰고 경고**한다(하드코딩 금지 — 필드는
    :func:`dataclasses.fields` 로만 확인한다). 같은 축을 두 번 주면 에러.
    """
    known = _config_fields() if known is None else known
    axes: list[tuple[str, list[Any]]] = []
    warns: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--grid 는 'NAME=v1,v2' 형식이어야 한다: {spec!r}")
        name, raw = spec.split("=", 1)
        name = name.strip()
        if name in seen:
            raise ValueError(f"--grid 축 '{name}' 이 중복이다. 한 축은 한 번만.")
        field = known.get(name)
        if field is None:
            warns.append(f"[경고] DetectorConfig 에 '{name}' 필드가 없다 — 이 축을 건너뛴다"
                         f" (아직 안 들어온 파라미터인지 확인하라).")
            continue
        seen.add(name)
        vals = [coerce_value(field, chunk) for chunk in raw.split(",") if chunk.strip()]
        if not vals:
            warns.append(f"[경고] '{name}' 축에 값이 없다 — 건너뛴다.")
            continue
        axes.append((name, vals))
    return axes, warns


def build_combos(axes: list[tuple[str, list[Any]]]) -> list[dict[str, Any]]:
    """축들의 데카르트곱. 축이 없으면 기본 설정 1건."""
    if not axes:
        return [{}]
    names = [n for n, _ in axes]
    return [dict(zip(names, vals, strict=True))
            for vals in itertools.product(*[v for _, v in axes])]


def history_warning(cfg: DetectorConfig) -> str | None:
    """관측 창이 히스토리 보관(``history``)을 넘으면 조용히 잘린다 — 경고한다."""
    need = max(cfg.surge_window * 1.5,
               cfg.cosurge_window + cfg.premove_window,
               cfg.volume_burst_window + cfg.volume_burst_base)
    if need > cfg.history:
        return (f"[경고] 관측 창 합계 {need:.0f}s > history {cfg.history:.0f}s — "
                f"이 조합은 히스토리가 잘려 신호가 과소평가된다. --grid 'history=...' 로 "
                f"같이 올려라.")
    return None


# --------------------------------------------------------------------- 집계

def _empty_scenario() -> dict[str, Any]:
    return {"n": 0, "wins": 0, "win_rate_pct": None, "mean_pct": None, "total_krw": 0.0}


def summarize_records(records: list[dict[str, Any]], notional: float) -> dict[str, Any]:
    """청산 로그 레코드들 → 한 체결가정의 성과 요약.

    * 미체결(``status == 'unfilled'``)은 건수로만 센다. mae/mfe 가 갱신된 적 없는 0.0
      이라 섞으면 중앙값이 리스크를 작아 보이게 왜곡한다(일일 집계와 같은 규칙).
    * 세전 수익률은 ``exit_price/entry - 1`` 로 직접 계산한다(가드 cost_rate 설정에
      흔들리지 않게). 거기서 34/64/94bp 를 차감해 병기한다.
    """
    closed = [r for r in records if r.get("status") == "closed"]
    unfilled = [r for r in records if r.get("status") == "unfilled"]
    rets = [g for g in (_gross(r) for r in closed) if g is not None]
    notionals = [notional] * len(rets)
    mae = [r["mae"] * 100 for r in closed if r.get("mae") is not None]
    mfe = [r["mfe"] * 100 for r in closed if r.get("mfe") is not None]
    out: dict[str, Any] = {
        "positions": len(records),
        "filled": len(closed),
        "unfilled": len(unfilled),
        "exit_reasons": dict(Counter(str(r.get("exit_reason") or "") for r in records)),
        "mae_med_pct": round(st.median(mae), 4) if mae else None,
        "mfe_med_pct": round(st.median(mfe), 4) if mfe else None,
    }
    out["gross"] = _scenario(rets, notionals, 0.0) if rets else _empty_scenario()
    for key, cost in COSTS.items():
        out[key] = _scenario(rets, notionals, cost) if rets else _empty_scenario()
    return out


def summarize_guard(guard: RiskGuard) -> dict[str, dict[str, Any]]:
    """가드의 회계 포지션과 그림자를 **체결가정별로** 요약한다."""
    notional = guard.config.position_notional
    out: dict[str, dict[str, Any]] = {}
    for group in (guard.closed_positions, guard.closed_shadows):
        if not group:
            continue
        basis = group[0].basis
        out[basis] = summarize_records([p.to_record(p.basis) for p in group], notional)
    # 한 건도 안 나온 가정도 빈 칸으로 남겨 표의 행 수가 조합마다 흔들리지 않게 한다.
    for basis in (guard.config.fill_basis,
                  "touch" if guard.config.fill_basis == "through" else "through"):
        out.setdefault(basis, summarize_records([], notional))
    return out


# --------------------------------------------------------------------- 실행

def run_combo(pairs: list[Pair], cfg: DetectorConfig, rows: list[tuple],
              guard_cfg: RiskGuardConfig, last_ts: datetime) -> dict[str, Any]:
    """조합 하나 — 같은 틱열을 새 감지기·새 가드에 다시 흘린다(리플레이와 동일 경로)."""
    detector = PairDetector(pairs, cfg)
    guard = RiskGuard(guard_cfg)
    bus = AlertBus([], dry_run=False)      # sink 없음 = 운영 로그·웹훅·stdout 미접촉
    t0 = time.perf_counter()
    alerts = _feed(detector, bus, rows, guard=guard)
    guard.flush_expired(last_ts)
    elapsed = time.perf_counter() - t0
    return {
        "alerts": alerts,
        "filtered": detector.filtered_count,
        "by_basis": summarize_guard(guard),
        "elapsed_s": round(elapsed, 2),
    }


def _fmt_param(v: Any) -> str:
    if v is None:
        return "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _bp(x: float | None) -> str:
    """수익률(%) → bp 문자열. 집계는 % 단위로 들어온다."""
    return "—" if x is None else f"{x * 100:+.0f}"


def _reasons(d: dict[str, int]) -> str:
    return "/".join(f"{k or '?'}{v}" for k, v in
                    sorted(d.items(), key=lambda kv: -kv[1])) or "—"


def print_table(results: list[dict[str, Any]], axis_names: list[str]) -> None:
    """조합 × 체결가정 한 행씩. 숫자는 bp(1bp = 0.01%)."""
    head = [*[n[:12] for n in axis_names], "가정", "알림", "배제", "체결", "미체결",
            "승64%", "세전bp", "n34", "n64", "n94", "net64합원", "MAE", "MFE", "청산사유"]
    rows: list[list[str]] = []
    for res in results:
        params = [_fmt_param(res["params"].get(n)) for n in axis_names]
        for basis, m in sorted(res["by_basis"].items()):
            g, n64 = m["gross"], m["net64"]
            rows.append([
                *params, basis, str(res["alerts"]), str(res["filtered"]),
                str(m["filled"]), str(m["unfilled"]),
                "—" if n64["win_rate_pct"] is None else f"{n64['win_rate_pct']:.0f}",
                _bp(g["mean_pct"]), _bp(m["net34"]["mean_pct"]), _bp(n64["mean_pct"]),
                _bp(m["net94"]["mean_pct"]), f"{n64['total_krw']:,.0f}",
                _bp(m["mae_med_pct"]), _bp(m["mfe_med_pct"]),
                _reasons(m["exit_reasons"]),
            ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(head)]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))


def write_jsonl(path: Path, day: str, results: list[dict[str, Any]],
                meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().isoformat(sep=" ", timespec="seconds")  # noqa: DTZ005
    with path.open("a", encoding="utf-8") as fh:
        for res in results:
            rec = {"run_ts": run_ts, "day": day, "warning": WARNING,
                   "guard_note": GUARD_NOTE, **meta, **res}
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="짝꿍 감지기 파라미터 스윕(틱 1회 로드·조합 재실행). 최적값 선택 도구 아님.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=WARNING)
    p.add_argument("--replay", required=True, metavar="YYYY-MM-DD",
                   help="재현할 날짜(이 하루의 ticks 를 한 번만 읽는다)")
    p.add_argument("--grid", action="append", default=[], metavar="NAME=v1,v2",
                   help=f"스윕 축(여러 번). 예: {SWEEP_HINT[0]}=10,15,30")
    p.add_argument("--pairs", help="'대장:후속,대장:후속' (없으면 워치리스트)")
    p.add_argument("--watchlist", default="data/pair_watchlist.json")
    p.add_argument("--aggressive", action="store_true",
                   help="기준 설정을 공격형으로(기본: 안정형). 그리드는 이 위에 얹힌다")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"JSONL 출력(기본 {DEFAULT_OUT})")
    p.add_argument("--dsn", help="DB DSN (기본 KR_QUANT_DB)")
    p.add_argument("--fill-basis", default="through", choices=["through", "touch"],
                   help="회계 기준 가정(반대 가정은 그림자로 항상 병기)")
    p.add_argument("--position-notional", type=float, default=100_000.0)
    p.add_argument("--max-positions", type=int, default=1000,
                   help="동시 보유 상한 — 스윕에선 넉넉히(표집이 잘리면 비교 불가)")
    p.add_argument("--limit-ticks", type=int, default=0,
                   help="로드할 틱 수 상한(스모크 테스트용). 0 = 전부")
    p.add_argument("--confirm-above", type=int, default=12,
                   help="이 조합 수를 넘으면 실행 전에 확인(기본 12)")
    p.add_argument("--yes", action="store_true", help="확인 생략")
    p.add_argument("--list-only", action="store_true", help="조합만 찍고 종료(DB 접속 안 함)")
    return p


def _load_env() -> None:
    """``KR_QUANT_DB`` 가 없으면 관례대로 ../quant-airflow/.env 를 시도한다(있으면 유지)."""
    if os.environ.get("KR_QUANT_DB"):
        return
    try:
        import _envload  # noqa: F401  import 만으로 환경변수 주입
    except Exception as exc:                                   # noqa: BLE001
        print(f"[경고] .env 자동 로드 실패({exc}) — KR_QUANT_DB 를 직접 export 하라.",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = DetectorConfig.aggressive() if args.aggressive else DetectorConfig.stable()

    axes, warns = parse_grid(args.grid)
    for w in warns:
        print(w, file=sys.stderr, flush=True)
    combos = build_combos(axes)
    axis_names = [n for n, _ in axes]

    print(f"# 짝꿍 파라미터 스윕 — {args.replay}  조합 {len(combos)}개"
          f" (기준 {'공격형' if args.aggressive else '안정형'})", flush=True)
    print(WARNING, flush=True)
    print(f"[가드] {GUARD_NOTE}", flush=True)
    for name, vals in axes:
        print(f"  축 {name}: {', '.join(_fmt_param(v) for v in vals)}", flush=True)
    if args.list_only:
        for c in combos:
            print("  " + " ".join(f"{k}={_fmt_param(v)}" for k, v in c.items()), flush=True)
        return 0
    if len(combos) > args.confirm_above and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(f"조합 {len(combos)}개 — 비대화 실행이라 확인을 못 받는다. "
                             f"--yes 를 주거나 --confirm-above 를 올려라.")
        input(f"조합 {len(combos)}개를 돌린다. 계속하려면 Enter (중단 Ctrl-C) … ")

    pairs = resolve_pairs(args)
    if not pairs:
        raise SystemExit("감시할 쌍이 없다. --pairs 또는 워치리스트를 확인하라.")

    _load_env()
    conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
    codes = sorted({c for p in pairs for c in (p.leader, p.follower)})
    t0 = time.perf_counter()
    rows = _fetch_ticks(conn, codes, since=None, day=args.replay)   # SELECT 만
    conn.close()
    if args.limit_ticks > 0:
        rows = rows[:args.limit_ticks]
    load_s = time.perf_counter() - t0
    print(f"쌍 {len(pairs)}개 · 종목 {len(codes)}개 · 틱 {len(rows):,}건 로드"
          f" ({load_s:.1f}s, 조합마다 재사용).", flush=True)
    if not rows:
        raise SystemExit("그날 틱이 없다.")
    last_ts = _as_dt(rows[-1][1])

    guard_cfg = RiskGuardConfig(
        daily_loss_limit=0.0, daily_loss_limit_pct=None,       # 킬스위치 OFF
        max_consecutive_losses=0, stop_file=None,              # 킬스위치 OFF
        max_positions=args.max_positions,
        position_notional=args.position_notional,
        fill_basis=args.fill_basis, track_shadow_fill=True,
        cost_rate=0.0,                                         # 세전 — 비용은 집계에서
    )

    results: list[dict[str, Any]] = []
    for i, params in enumerate(combos, 1):
        cfg = replace(base, **params) if params else base
        hw = history_warning(cfg)
        if hw:
            print(hw, file=sys.stderr, flush=True)
        res = run_combo(pairs, cfg, rows, guard_cfg, last_ts)
        res["params"] = {k: params.get(k) for k in axis_names}
        res["config"] = {f.name: getattr(cfg, f.name) for f in fields(DetectorConfig)}
        results.append(res)
        desc = " ".join(f"{k}={_fmt_param(v)}" for k, v in params.items()) or "(기본)"
        print(f"  [{i}/{len(combos)}] {desc} → 알림 {res['alerts']} · 배제 "
              f"{res['filtered']} ({res['elapsed_s']:.1f}s)", flush=True)

    print(flush=True)
    print_table(results, axis_names)
    meta = {"pairs": len(pairs), "ticks": len(rows), "fill_basis": args.fill_basis,
            "position_notional": args.position_notional,
            "base": "aggressive" if args.aggressive else "stable"}
    out = Path(args.out)
    write_jsonl(out, args.replay, results, meta)
    total = sum(r["elapsed_s"] for r in results)
    print(f"\n{len(results)}개 조합 · 합계 {total:.0f}s (조합당 평균 "
          f"{total / max(len(results), 1):.1f}s) → {out}", flush=True)
    print(WARNING, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
