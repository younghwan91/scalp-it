"""``scalp-pair-detect`` — 짝꿍 실시간 감지·알림기 (등록 쌍만 감시, 주문은 사람이).

세 모드:

* **--live 소켓 직결**(실전 주경로): 수집 데몬과 한 프로세스로 붙어 키움 WS 틱을 받는
  즉시 판정한다. DB 폴링을 안 거쳐 지연 ~0 — "안 뛴 후속주"를 잡는 게 핵심이라 속도가
  엣지다. 감지 뒤 같은 프레임을 ticks·quotes 로 저장한다(--no-store 로 끔). App key
  1세션 제약은 :func:`run_live` 참고(전용 키 권장, 또는 tick-collect 대체).

      scalp-pair-detect --live --duration 09:00-15:20
      scalp-pair-detect --live --aggressive       # PAIR_DETECT_APP_KEY 있으면 전용 키

* **폴링**(백업): 수집 데몬이 쌓는 ``ticks`` 를 짧은 주기(기본 1.5초)로 폴링. 최악
  3초 지연이지만 소켓을 못 쓸 때 백업. ``ticks`` 읽기만·주문 안 함.

      scalp-pair-detect --duration 09:00-15:20                 # data/pair_watchlist.json 감시

* **--replay 검증**: 과거 날짜의 틱을 시간순으로 흘려 그날 감지기가 뭘 알렸을지 재현
  (--dry-run 강제, 프로덕션 로그·웹훅 안 건드림). 폴링 정밀 재현이라 검증용.

      scalp-pair-detect --replay 2026-08-24

청산 모드: ``--stable``(기본, 익절+0.5%/일상손절−0.4%/90초) 또는 ``--aggressive``
(익절+0.8%/일상손절없음+재난선−0.7%/90초). 익절·손절·재난선·홀드초는 개별 플래그로 덮는다.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import signal
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

from .intraday.store import _placeholder, connect, is_postgres
from .realtime.alerts import AlertBus, build_default_bus
from .realtime.order_executor import OrderExecutor
from .realtime.pair_detector import (
    DEFAULT_WATCHLIST,
    Alert,
    DetectorConfig,
    Pair,
    PairDetector,
    load_watchlist,
    parse_pairs_arg,
)
from .realtime.risk_guard import PositionLogSink, RiskGuard, RiskGuardConfig

DEFAULT_WATCHLIST_PATH = "data/pair_watchlist.json"

_TICK_COLS = "code, ts, price, volume, best_bid, side"


# --------------------------------------------------------------------- 쌍 로딩

def resolve_pairs(args: argparse.Namespace) -> list[Pair]:
    if args.pairs:
        return parse_pairs_arg(args.pairs)
    path = Path(args.watchlist)
    if path.exists():
        return load_watchlist(path)
    # 워치리스트 파일이 없으면 내장 기본 쌍으로.
    from .realtime.pair_detector import _pair_from_dict

    print(f"[경고] {path} 없음 → 내장 기본 쌍 {len(DEFAULT_WATCHLIST)}개 사용", file=sys.stderr)
    return [_pair_from_dict(d) for d in DEFAULT_WATCHLIST]


def build_config(args: argparse.Namespace) -> DetectorConfig:
    cfg = DetectorConfig.aggressive() if args.aggressive else DetectorConfig.stable()
    over: dict[str, Any] = {}
    if args.take_profit is not None:
        # 0 = 상한 없음. 손절·재난선과 같은 관례(만쥬: 짧게 잃고 크게 먹는다).
        over["take_profit"] = None if args.take_profit == 0 else args.take_profit
    if args.stop_loss is not None:
        over["stop_loss"] = None if args.stop_loss == 0 else args.stop_loss
    if args.disaster_line is not None:
        over["disaster_line"] = None if args.disaster_line == 0 else args.disaster_line
    if args.hold is not None:
        over["time_exit"] = args.hold
    if args.surge is not None:
        over["surge_ret"] = args.surge
    if args.cooldown is not None:
        over["cooldown"] = args.cooldown
    if args.min_turnover is not None:
        over["min_turnover"] = args.min_turnover
    if args.min_buy_ratio is not None:
        over["min_buy_ratio"] = args.min_buy_ratio
    if args.min_sl_ticks is not None:
        over["min_sl_ticks"] = args.min_sl_ticks
    if args.cosurge is not None:
        over["cosurge_min"] = args.cosurge
    if args.premove is not None:
        over["premove_max"] = args.premove
    if args.accel is not None:
        over["accel_min_ratio"] = args.accel
    if args.volume_burst is not None:
        over["volume_burst_ratio"] = args.volume_burst
    if args.entry_ticks_up:
        over["entry_ticks_up"] = args.entry_ticks_up
    if args.limitup_touch:
        over["limitup_touch"] = True
    if args.limitup_near is not None:
        over["limitup_near_pct"] = args.limitup_near
    if args.limitup_near_ticks:
        over["limitup_near_ticks"] = args.limitup_near_ticks
    if args.follower_priority != "rank":
        over["follower_priority"] = args.follower_priority
    if args.limitup_lock is not None:
        over["limitup_lock_secs"] = args.limitup_lock
    if args.limitup_require_surge:
        over["limitup_require_surge"] = True
    if args.vi_near is not None:
        over["vi_near_pct"] = args.vi_near
    if args.max_roundtrip_bp is not None:
        over["max_roundtrip_bp"] = args.max_roundtrip_bp
    if args.order_krw is not None:
        over["order_krw"] = args.order_krw
    if args.lock_min_krw is not None:
        over["lock_min_krw"] = args.lock_min_krw
    if args.lock_wall_decay is not None:
        over["lock_wall_decay"] = args.lock_wall_decay
    if args.dump_krw is not None:
        over["dump_krw"] = args.dump_krw
    if args.lock_wall_window is not None:
        over["lock_wall_window"] = args.lock_wall_window
    if args.leader_stall is not None:
        over["leader_stall_secs"] = args.leader_stall
    if args.leader_drop_ticks is not None:
        over["leader_drop_ticks"] = args.leader_drop_ticks
    if args.big_trades is not None:
        over["min_big_trades"] = args.big_trades
    if args.stale_sector_limitups > 0:
        over["max_sector_limitups"] = args.stale_sector_limitups
    from dataclasses import replace

    return replace(cfg, **over) if over else cfg


def _parse_loss_limit(spec: str | None) -> tuple[float, float | None]:
    """``"20000"`` → (20000원, None) · ``"2%"`` → (0, 0.02). (원, pct)."""
    if spec is None:
        return 0.0, None
    s = spec.strip()
    if s.endswith("%"):
        return 0.0, float(s[:-1]) / 100.0
    return float(s), None


def build_guard(args: argparse.Namespace) -> RiskGuard | None:
    """가드 구성. ``--no-guard`` 면 None(기존 동작)."""
    if args.no_guard:
        return None
    loss_krw, loss_pct = _parse_loss_limit(args.daily_loss_limit)
    cfg = RiskGuardConfig(
        daily_loss_limit=loss_krw,
        daily_loss_limit_pct=loss_pct,
        daily_capital=args.capital,
        position_notional=args.position_notional,
        max_consecutive_losses=args.max_consecutive_losses,
        trail_arm=args.trail_arm,
        trail_pct=args.trail,
        drawdown_krw=args.drawdown_krw,
        thin_book_bp=args.thin_book_bp,
        thin_book_multiplier=args.thin_book_size,
        scratch_ticks=args.scratch_ticks,
        scratch_secs=args.scratch_secs,
        profit_giveback_ratio=args.profit_giveback,
        profit_giveback_min_krw=args.profit_giveback_min,
        shadow_when_blocked=args.shadow_when_blocked,
        stop_needs_leader_weak=args.stop_needs_leader_weak,
        drawdown_size_multiplier=args.drawdown_size,
        max_positions=args.max_positions,
        fill_basis=args.fill_basis,
        cost_rate=args.cost_rate,
        leader_exit=not args.no_leader_exit,
        size_multiplier=args.size_multiplier,
        daily_profit_target=args.daily_profit_target,
        profit_size_multiplier=args.profit_size_multiplier,
        partial_exit_ratio=args.partial_exit,
        partial_exit_gain=args.partial_exit_gain,
        partial_exit_after_s=args.partial_exit_after,
        stop_file=(None if args.no_stop_file else args.stop_file),
    )
    return RiskGuard(cfg)


# --------------------------------------------------------------------- DB 읽기

def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _fetch_ticks(conn: Any, codes: list[str], *, since: datetime | None,
                 day: str | None) -> list[tuple]:
    """감시 종목의 틱을 (ts, seq) 순으로. ``since`` 이후 또는 ``day`` 하루치."""
    ph = _placeholder(conn)
    in_clause = ",".join([ph] * len(codes))
    sql = f"select {_TICK_COLS} from ticks where code in ({in_clause})"
    params: list[Any] = list(codes)
    if day is not None:
        sql += f" and ts >= {ph} and ts < {ph}"
        params += [f"{day} 00:00:00", f"{day} 23:59:59"]
    if since is not None:
        sql += f" and ts > {ph}"
        params.append(since if is_postgres(conn) else since.strftime("%Y-%m-%d %H:%M:%S"))
    sql += " order by ts, seq"
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _disclosure_rows(db_path: str, asof: date, days: int) -> list[dict[str, Any]]:
    """최근 공시 원본 행(제목 포함) — LLM 채점 입력. 실패하면 빈 리스트."""
    import sqlite3
    from datetime import timedelta

    p = Path(db_path)
    if not p.exists():
        return []
    lo = (asof - timedelta(days=days)).isoformat().replace("-", "")
    hi = asof.isoformat().replace("-", "")
    try:
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            cur = conn.execute(
                "select rcept_no, code, corp_name, report_nm, material_type "
                "from disclosures where rcept_dt >= ? and rcept_dt <= ? "
                "and code is not null", (lo, hi))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    except sqlite3.Error:
        return []


def build_universe_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """만쥬 동적 모드 입력(유니버스·섹터·전일종가)을 만든다. **실패해도 죽지 않는다.**

    08:55 에 이게 예외로 죽으면 그날을 통째로 날린다. 그래서 어떤 실패든 잡아
    빈 dict 를 돌려주고, 호출부는 기존 워치리스트 고정 쌍으로 폴백한다 — 만쥬
    손익 구조(익절 없음·약손절·타임아웃)는 폴백에서도 그대로 적용된다.
    """
    if args.no_universe:
        return {}
    conn = None
    try:
        from .market_regime import evaluate_gate, previous_session
        from .universe import UniverseConfig, build_universe

        # 감지기 생성 시점엔 아직 커넥션이 없다(live 모드는 run_live 가 따로 연다).
        # 유니버스는 장전 1회만 읽으면 되므로 짧게 열고 닫는다.
        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))

        today = date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
        prev = previous_session(conn, today)
        if prev is None:
            print("[경고] 직전 영업일을 못 찾음 — 고정 쌍으로 폴백", file=sys.stderr)
            return {}
        # 시황 게이트는 **판정만 하고 막지 않는다.** dry-run 은 만쥬가 쉬라고 한
        # 국면의 데이터도 필요하다(활황/침체 성과 차이를 실측할 재료). 실발주로
        # 갈 때 이 판정을 강제로 바꿔야 한다.
        try:
            gate = evaluate_gate(conn, prev, target=today)
            # 만쥬 §4-3 광기 구간 — 우리 보유 4분이라 "1시간 제한·오버나이트 금지"는
            # 자동 충족이다. 그래서 이 판정의 쓸모는 **기록**이다: 그날이 광기였는지
            # 알아야 나중에 성적을 제대로 읽는다.
            try:
                from .market_regime import count_limitups, is_mania

                n_lim = count_limitups(conn, prev)
                if is_mania(turnover_jo=gate.turnover_jo, limitups=n_lim):
                    print(f"#   ⚠ 광기 구간 — 거래대금 {gate.turnover_jo:.1f}조 · "
                          f"상한가 {n_lim}건(둘 다 1년 90분위 초과). 만쥬는 이때 "
                          f"1시간 이내·오버나이트 금지라 했다(보유 4분이라 이미 충족).",
                          file=sys.stderr)
                else:
                    print(f"#   상한가 {n_lim}건(전일)", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — 광기 판정 실패로 기동을 막지 않는다
                print(f"[경고] 광기 구간 판정 실패(계속): {exc}", file=sys.stderr)
            print(f"# 시황 — {gate.asof} 코스닥 {gate.turnover_jo:.2f}조 · {gate.regime}"
                  f" · 매매권고 {'예' if gate.tradable else '아니오'}"
                  f" · 비중배수 {gate.size_multiplier:g}", file=sys.stderr)
            print(f"#   {gate.note}", file=sys.stderr)
            if not gate.tradable:
                print("#   ↑ 만쥬 기준으론 쉬는 날이다. dry-run 이라 계속 진행하되, "
                      "이 결과는 '쉬라고 한 국면'의 것으로 해석해야 한다.", file=sys.stderr)
            # 만쥬: "코스닥 거래대금이 7~8조 원 이하(침체장)이면 짝꿍 신호가 떠도
            # 진입하지 않는다." 기존 `tradable` 은 5조(극침체)에서만 꺼지므로 5~8조
            # 구간이 그대로 통과했다. 배수 0 이면 **신규 진입만** 막히고 이미 열린
            # 포지션의 청산은 계속 돈다 — 장중에 국면이 나빠져도 손절은 나가야 한다.
            if args.halt_below_jo > 0 and gate.turnover_jo < args.halt_below_jo:
                args.size_multiplier = 0.0
                print(f"#   ↑ 거래대금 {gate.turnover_jo:.2f}조 < "
                      f"{args.halt_below_jo:g}조 — 만쥬 기준 진입 금지. 신규 진입을 "
                      f"막는다(청산은 계속).", file=sys.stderr)
            elif args.scout_gate_jo > 0 and gate.turnover_jo < args.scout_gate_jo:
                # 만쥬 A1(2026-08-30) — Gate 0(7~8조) 아래는 "쉬거나 극소액". 우리는
                # 실측 데이터를 얻으려 게이트를 5조로 낮춰 뒀는데(2026-08-27 사용자
                # 결정), 그 구간의 **비중은 1/10 이어야** 만쥬 기준에 맞는다. 실주문은
                # 이미 1주지만 가상 회계 명목이 1천만원 그대로면 회계가 '평소 비중'으로
                # 기록돼 성과 해석이 틀어진다.
                from .market_regime import scout_multiplier

                args.size_multiplier = scout_multiplier(
                    turnover_jo=gate.turnover_jo, gate_jo=args.scout_gate_jo)
                print(f"#   ↑ 거래대금 {gate.turnover_jo:.2f}조 < Gate 0 "
                      f"{args.scout_gate_jo:g}조 — 정찰병 비중 "
                      f"{args.size_multiplier:g} 배(만쥬 A1).", file=sys.stderr)
            elif args.size_by_regime:
                # 만쥬 §8-1 — 국면이 나빠지면 시드를 줄인다("1억을 그대로 베팅하지
                # 않고 3천만 이하로"). 배수 0 이면 신규 진입만 막고 기존 청산은 돈다.
                args.size_multiplier = gate.size_multiplier
                print(f"#   비중 배수 {gate.size_multiplier:g} 적용(--size-by-regime)",
                      file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — 게이트 실패로 매매를 막지 않는다
            print(f"[경고] 시황 판정 실패(계속 진행): {exc}", file=sys.stderr)
        # 만쥬 §4-5 — 신규상장주는 과거 일봉이 없어 거래대금·변동성 필터를 못 넘는다.
        # 일반 랭킹으로는 영영 안 뽑히므로 pinned 로 박는다("상장 당일은 무조건 등록").
        pinned: tuple[str, ...] = ()
        if args.new_listing_days > 0:
            from .universe import fetch_first_days, pick_recent_listings

            fresh = pick_recent_listings(fetch_first_days(conn), asof=prev,
                                         within_days=args.new_listing_days)
            pinned = tuple(fresh)
            if pinned:
                print(f"# 신규상장 편입 {len(pinned)}종목(최근 {args.new_listing_days}일): "
                      f"{', '.join(pinned)}", file=sys.stderr)
        # 만쥬 §2-1 3대 요건의 세 번째 — "명확한 재료 및 미래 가치". 거래대금·변동성만
        # 보면 "수급은 있는데 이야기가 없는" 종목이 올라온다. 재료는 DART 공시에서
        # 온다(뉴스 크롤러 금지 정책, docs/research/manju/17 — ToS·저작권).
        materials: dict[str, list[str]] = {}
        llm_materials: dict[str, float] = {}
        if args.material_days > 0:
            from datetime import timedelta

            from .universe import fetch_recent_materials

            # `until` 을 **오늘**로 둔다. 전일까지만 보면 07:30~09:00 에 접수된 공시 —
            # 그날 가장 신선한 재료 — 가 유니버스에서 빠지고, 장중 감시자도 기동
            # 시점을 커서로 잡아 건너뛰므로 **양쪽 어디에서도 안 쓰인다**.
            materials = fetch_recent_materials(
                args.dart_db, since=prev - timedelta(days=args.material_days),
                until=today)
            print(f"# 재료 공시 {len(materials)}종목(최근 {args.material_days}일, "
                  f"{args.dart_db})", file=sys.stderr)
            # 만쥬 §2-1 은 "명확한 재료 및 **미래 가치**", "관심을 **지속**시킬 강력한
            # 스토리"를 요구한다. 유형별 가중치표로는 그 판단이 안 된다 — "공급계약"이
            # 1억인지 1조인지 표는 모른다. LLM 이 있으면 그걸 쓰고, 없으면(키 미설정 등)
            # 조용히 규칙 기반으로 간다. 어느 단계도 장 시작을 막지 않는다.
            if args.material_llm:
                from .material_llm import score_disclosures

                # 전 종목 5일치를 그대로 넘기면 수천 건이라, 캐시가 빈 첫 실행에서
                # 08:55 기동이 동기 LLM 루프에 묶여 개장을 놓친다. 재료가 있는 종목만
                # 넘기고 상한도 건다 — 유니버스는 45종목이라 그 이상은 어차피 안 쓴다.
                rows = [r for r in _disclosure_rows(args.dart_db, prev,
                                                    args.material_days)
                        if str(r.get("code") or "") in materials]
                llm_scores = score_disclosures(rows, cache_path=args.llm_cache,
                                               limit=args.material_llm_max)
                if llm_scores:
                    llm_materials = llm_scores
                    print(f"#   LLM 재료 채점 {len(llm_scores)}종목", file=sys.stderr)
                else:
                    print("#   LLM 재료 채점 불가(키·캐시 확인) — 규칙 기반으로 진행",
                          file=sys.stderr)
        uni = build_universe(conn, prev,
                             UniverseConfig(limit=args.universe_size, pinned=pinned,
                                            weight_material=args.material_weight),
                             materials, llm_materials or None)
        if not uni.codes:
            print("[경고] 유니버스가 비었다 — 고정 쌍으로 폴백", file=sys.stderr)
            return {}
        print(f"# 유니버스 {len(uni.codes)}종목(기준일 {uni.asof}) · "
              f"짝꿍성립 {len(uni.pairable_codes)} · 섹터 {len(uni.sector_counts)}개",
              file=sys.stderr)
        out: dict[str, Any] = {"universe": uni.pairable_codes,
                               "sectors": uni.sector_by_code,
                               "prev_close": uni.prev_close_by_code}
        # 실전 테마 — 만쥬: "27종 표준 산업분류로는 실전 테마를 절대 잡아낼 수
        # 없습니다." 키움 테마 스냅샷이 있으면 짝꿍을 그 무리에서 고른다. 전 종목의
        # 25% 만 덮으므로(643/2700) 주 선택자는 못 되고, 없는 종목은 섹터로 떨어진다.
        if not args.no_themes:
            themes = _fetch_themes(conn)
            if themes:
                out["themes"] = themes
                covered = sum(1 for c in uni.pairable_codes if c in themes)
                print(f"# 실전 테마 — {len(themes)}종목 매핑 · 유니버스 커버 "
                      f"{covered}/{len(uni.pairable_codes)}", file=sys.stderr)
        # 만쥬 §5-5 테마 신선도 — 최근 상한가가 잦은 섹터는 '소비된 테마'로 본다.
        # 정확한 재현이 아니라 대용치다(신선도는 등급 C, 자동화 불가 항목).
        if args.stale_sector_limitups > 0:
            from .universe import fetch_sector_limitups

            counts = fetch_sector_limitups(conn, prev, days=args.stale_sector_days)
            out["sector_limitup_counts"] = counts
            top = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])
            print(f"# 섹터 신선도 — 최근 {args.stale_sector_days}영업일 상한가 {top}",
                  file=sys.stderr)
        return out
    except Exception as exc:  # noqa: BLE001 — 하루를 날리느니 폴백한다
        print(f"[경고] 유니버스 선정 실패({exc}) — 고정 쌍으로 폴백", file=sys.stderr)
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001, S110 — 닫기 실패로 기동을 막지 않는다
                pass


def _pin_codes(inline: str, path: str) -> list[str]:
    """고정 수집 종목 = 인자 목록 + 파일 목록. 파일은 다른 전략(daytrade-it)이 전일 마감 뒤 떨군다.

    파일이 없거나 못 읽으면 **경고만 하고 인자 목록으로 간다** — 남의 파일 때문에 매매가 안 뜨면 안 된다.
    """
    out = [c.strip() for c in str(inline or "").split(",") if c.strip()]
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[경고] 고정 수집 파일을 못 읽는다({path}): {exc} — 인자 목록만 쓴다",
                  file=sys.stderr, flush=True)
        else:
            bad = []
            for line in text.splitlines():
                line = line.split("#", 1)[0]          # 줄 주석 — 날짜·메모가 종목으로 들어가면 안 된다
                for tok in line.replace(",", " ").split():
                    if not (len(tok) == 6 and tok.isalnum()):
                        bad.append(tok)               # 남의 파일이다 — 코드가 아닌 건 구독에 넘기지 않는다
                    elif tok not in out:
                        out.append(tok)
            if bad:
                print(f"[경고] 고정 수집 파일에서 종목코드가 아닌 항목 {len(bad)}개를 건너뛴다: {bad[:3]}",
                      file=sys.stderr, flush=True)
    return out


def _cap_codes_for_single_key(codes: list[str], types: list[str], *, dedicated: bool) -> list[str]:
    """전용 키가 없으면 커넥션 1개에 담기는 만큼으로 자른다.

    키움 WebSocket 은 App key 당 동시 세션 1개다. 등록 한도(≈100건 = 종목×type)를
    넘으면 수집기가 커넥션을 쪼개는데, 같은 키로 두 개를 열면 서로 kick 하며 죽는다.
    2026-08-24 에 실제로 80종목 중 **30종목이 10:58 에 끊겨** 하루의 3분의 2를 잃었다
    (둘째 커넥션이 통째로 사망). 조용히 반쪽 데이터를 받느니 처음부터 받을 수 있는
    만큼만 구독하는 게 낫다.
    """
    from .realtime.collector import codes_per_connection

    if dedicated:
        return codes                      # 전용 키가 있으면 다커넥션이 안전하다
    cap = codes_per_connection(types)
    if len(codes) <= cap:
        return codes
    print(f"[경고] 종목 {len(codes)}개 > 단일 키 한도 {cap}개(type {len(types)}종) — "
          f"앞의 {cap}개만 구독한다. 2026-08-24 에 이 한도를 넘겨 30종목이 10:58 에 "
          f"끊긴 전례가 있다. 전 종목이 필요하면 PAIR_DETECT_APP_KEY 를 두라.",
          file=sys.stderr, flush=True)
    return codes[:cap]


def _advance_cursor(rows: list[tuple[str, str, str]], take: list[str],
                    defer: list[str], cursor: str | None) -> str | None:
    """이번에 처리한 데까지만 커서를 민다.

    상한에 걸려 미룬 종목(``defer``)이 있으면 그 종목의 **첫 공시 직전**까지만
    전진시킨다 — 끝까지 밀면 다음 사이클에 후보로 다시 안 올라와 영영 놓친다.
    """
    if not defer:
        return rows[-1][0] if rows else cursor
    deferred = set(defer)
    for rcept, code, _mtype in rows:
        if code in deferred:
            return cursor if rcept == rows[0][0] else _prev_rcept(rows, rcept)
    return rows[-1][0]


def _prev_rcept(rows: list[tuple[str, str, str]], rcept: str) -> str:
    """``rcept`` 바로 앞 접수번호(없으면 그대로)."""
    prev = rcept
    for r, _c, _m in rows:
        if r == rcept:
            return prev
        prev = r
    return prev


def _breaking_meta(args: argparse.Namespace, codes: list[str]) -> dict[str, Any]:
    """장중 편입 종목의 섹터·전일종가를 읽어 온다. 실패하면 빈 dict(편입은 계속)."""
    try:
        from .intraday.store import connect

        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
    except Exception:  # noqa: BLE001 — 메타 조회 실패로 감시를 죽이지 않는다
        return {}
    try:
        ph = _placeholder(conn)
        marks = ",".join([ph] * len(codes))
        cur = conn.cursor()
        cur.execute(f"select code, sector from stocks where code in ({marks})", codes)
        sectors = {str(c): str(s) for c, s in cur.fetchall() if s}
        cur.execute(
            f"select code, max(date) from daily_bars where code in ({marks}) group by code",
            codes)
        last = dict(cur.fetchall())
        prev: dict[str, float] = {}
        for code, day in last.items():
            cur.execute(
                f"select close from daily_bars where code = {ph} and date = {ph}",
                [code, day])
            row = cur.fetchone()
            if row and row[0]:
                prev[str(code)] = float(row[0])
        # 실전 테마도 같이 넣는다 — 장중에 붙은 종목이 테마 무리에 안 들어가면
        # 감지기가 27종 섹터로만 짝꿍을 고르게 되어 규칙이 갈린다.
        themes = {c: t for c, t in _fetch_themes(conn).items() if c in set(codes)}
        return {"sectors": sectors, "prev_close": prev, "themes": themes}
    except Exception:  # noqa: BLE001
        return {}
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def codes_to_enter(wanted: Sequence[str], added: Sequence[str],
                   rejected: Sequence[str]) -> list[str]:
    """감지기에 등록할 종목 — **새로 붙은 것 + 이미 구독 중이던 것.**

    ``collector.add_codes`` 는 이미 구독 중인 종목을 ``added`` 에도 ``rejected`` 에도
    안 담고 조용히 건너뛴다(`plan_additions` 의 ``if code in seen: continue``).
    ``added`` 만 감지기에 넣으면 그 종목은 **구독은 있는데 감지기가 모르는** 상태가
    되고, 틱이 `on_tick` 첫 줄에서 버려져 신호도 주문도 안 난다.

    수집 채움이 오늘 등락률 상위를 먼저 붙들기 때문에, 하필 **대장이 될 확률이 가장
    높은 종목들**이 이 구멍에 빠진다. 이 레포는 같은 종류의 사고를 두 번 겪었다
    (주문 화이트리스트 0종목, 장중 편입 종목이 화이트리스트 밖).

    거절된 것만 뺀다 — 그건 자리가 없어 진짜로 구독이 없다.
    """
    drop = set(rejected)
    got = set(added)
    return [c for c in dict.fromkeys(wanted) if c not in drop or c in got]


async def _watch_program_flow(
    guard: RiskGuard | None, bus: AlertBus, args: argparse.Namespace,
) -> None:
    """장중 프로그램 수급 감시 — 만쥬 "쏟아내는 장세면 진입을 거릅니다".

    우리는 지금까지 **대장 종목**의 매도 폭탄(`--dump-krw`)만 봤다. 만쥬가 말한 건
    시장 전체 프로그램 수급이라 종목 단위로는 표현이 안 된다.

    **킬이 아니다** — 신규 진입만 막고 열린 포지션은 규칙대로 청산된다. 수급이
    돌아오면 그 자리에서 풀린다.
    """
    import asyncio
    from datetime import date as _date

    from .adapters.kiwoom import KiwoomClient
    from .realtime.program_flow import latest_flow

    if guard is None:
        return
    appkey = os.environ.get("KIWOOM_APP_KEY", "")
    secret = os.environ.get("KIWOOM_APP_SECRET", "")
    if not appkey or not secret:
        bus.notice("[프로그램수급] 인증 정보 없음 → 감시 비활성")
        return
    client = KiwoomClient(appkey, secret)
    day = _date.today().strftime("%Y%m%d")  # noqa: DTZ011 — KST naive, 프로젝트 표준
    was = False
    while True:
        await asyncio.sleep(max(30.0, args.program_interval))
        try:
            rows = await asyncio.to_thread(
                client.fetch_program_rows, day=day, market=args.program_market)
            flow = latest_flow(rows)
            if flow is None:
                continue
            now = flow.dumping(min_net_qty=args.program_dump_qty)
            if now != was:
                guard.set_program_dumping(now)
                bus.notice(
                    f"[프로그램수급] {flow.time} 순매수 {flow.net_qty:+,.0f}천주"
                    f"(비차익 {flow.nondiff_net_qty:+,.0f}) — "
                    + ("**쏟아내는 장세, 신규 진입 차단**(만쥬)" if now
                       else "회복 — 신규 진입 재개"))
                was = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 수급 조회 실패로 감지를 죽이지 않는다
            bus.notice(f"[프로그램수급] 조회 실패(계속): {exc}")


async def _preempt_collect_slots(
    collector: Any, detector: PairDetector, guard: RiskGuard | None,
    collect_taken: dict[str, float] | None, rejected: Sequence[str],
) -> tuple[list[str], list[str]]:
    """자리가 없어 거절당한 종목을 위해 **수집 자리를 뺏어** 다시 붙인다.

    → ``(뺏어서 붙인 것, 그래도 못 붙인 것)``

    2026-08-31 이 이 함수를 만들었다. 수집채움이 09:05 에 39자리를 잡아 정원 50 을
    채웠고, 예비분 12자리는 스캐너가 붙인 종목이 안 식어 열리지 않았다. 그날 대장스캔은
    **233회 거절**당했고(317330 이 60회, 178780 이 55회 두드렸다) 돌발재료 010140 도
    자리가 없어 놓쳤다. 예비 슬롯 수를 키우는 건 미봉책이다 — 자리를 나누는 게 아니라
    **우선순위**를 정해야 한다. 매매가 저장을 이긴다.

    수집이 내줄 게 없으면 조용히 원래 거절 목록을 돌려준다(스캔을 죽이지 않는다).
    """
    from .realtime.collect_fill import plan_preemption

    if not rejected or not collect_taken:
        return [], list(rejected)
    keep = set(detector.codes)
    if guard is not None:
        for pos in guard.open_positions:
            keep.add(pos.follower)
            keep.add(pos.leader)
    freed = plan_preemption(collect_taken, need=len(rejected), keep=keep)
    if not freed:
        return [], list(rejected)
    await collector.remove_codes(freed)
    for code in freed:
        collect_taken.pop(code, None)
    return await collector.add_codes(rejected)


async def _watch_leader_scan(
    collector: Any, detector: PairDetector, bus: AlertBus, args: argparse.Namespace,
    guard: RiskGuard | None = None, executor: OrderExecutor | None = None,
    movers: list[str] | None = None, collect_taken: dict[str, float] | None = None,
) -> None:
    """장중 시장 전체 대장 스캔 — 만쥬가 등락률 상위 창으로 하는 일(§2).

    우리는 전날 밤에 유니버스를 고르고 그것만 구독했다. 그런데 선정 기준이 **전날
    거래대금**이라 오늘 상한가로 가는 종목과 거의 안 겹친다 — 실측 커버리지 2%
    (최근 7거래일 상한가 52건 중 1건). 방아쇠를 아무리 정확히 맞춰도 대장을 안
    보고 있으면 신호가 안 뜬다.

    그래서 ``ka10027`` 등락률 상위를 주기적으로 훑어 **시장 전체에서** 대장 후보를
    찾고, 그 종목과 같은 섹터 2·3등주를 장중에 구독에 추가한다.

    등록 한도가 있어 자리가 차면 **이 스캐너가 붙였던 것 중 식은 것부터** 내린다.
    장전 유니버스와 열린 포지션 종목은 절대 안 내린다.
    """
    import asyncio

    from .adapters.kiwoom import KiwoomClient
    from .realtime.leader_scan import (
        in_adoption_window,
        leader_candidates,
        parse_scan_rows,
        sector_followers,
    )

    appkey = os.environ.get("KIWOOM_APP_KEY", "")
    secret = os.environ.get("KIWOOM_APP_SECRET", "")
    if not appkey or not secret:
        bus.notice("[대장스캔] 인증 정보 없음 → 스캔 비활성")
        return
    client = KiwoomClient(appkey, secret)
    base = set(detector.codes)          # 장전 유니버스 — 절대 안 내린다
    hot: dict[str, float] = {}          # 스캐너가 붙인 코드 → 마지막으로 뜨거웠던 시각
    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(max(5.0, args.leader_scan_interval))
        try:
            # 만쥬는 오전에 그날 대장을 낙점한다. 창을 벗어나면 **새 대장만** 안
            # 받는다 — 이미 붙은 종목은 종일 그대로 돈다.
            hhmm = datetime.now().strftime("%H%M")  # noqa: DTZ005 — KST naive
            if not in_adoption_window(hhmm, args.leader_scan_window):
                continue
            rows = await asyncio.to_thread(
                client.fetch_rank_rows, market=args.leader_scan_market)
            scanned = parse_scan_rows(rows)
            if movers is not None:
                # 수집 채움에 넘긴다 — **낙점 문턱 아래까지** 전부. 12% 를 못 넘은
                # 종목이 이따가 상한가로 가고, 그때 이미 붙어 있어야 만쥬의 순간을
                # 본다. 여기가 아니면 그 순위를 다시 받을 곳이 없다.
                movers[:] = [r.code for r in
                             sorted(scanned, key=lambda r: -r.flu_rt)]
            leaders = leader_candidates(
                scanned, min_flu_rt=args.leader_scan_min_rt,
                min_buy_qty=args.leader_scan_min_wall)
            if not leaders:
                continue
            now = loop.time()
            sectors = _scan_sectors(args, [r.code for r in scanned])
            # 무리는 **감지기와 같은 규칙**으로 정한다 — 실전 테마가 있으면 테마,
            # 없으면 27종 섹터. 다른 규칙이면 스캐너가 붙인 짝꿍을 감지기가 안 산다.
            groups = {c: _group_key(c, detector, sectors) for c in sectors}
            want: list[str] = []
            for lead in leaders[:args.leader_scan_max]:
                hot[lead.code] = now
                want.append(lead.code)
                # 순위 범위는 **감지기 설정 하나**를 쓴다 — 스캐너가 붙이는 짝꿍과
                # 감지기가 고르는 짝꿍이 다른 규칙이면 붙여놓고 안 사는 일이 생긴다.
                for fol in sector_followers(scanned, lead.code, groups,
                                            rank_min=detector.config.follower_rank_min,
                                            rank_max=detector.config.follower_rank_max):
                    hot[fol] = now
                    want.append(fol)
            fresh = [c for c in dict.fromkeys(want) if c not in detector.codes]
            if not fresh:
                continue
            # 자리를 먼저 비운다 — 붙일 게 있는데 한도에 걸리면 그날 나머지 대장을
            # 영영 못 본다. 내리는 건 **이 스캐너가 붙였고 식은 것**뿐이다.
            stale = _stale_scan_codes(hot, base, guard, keep_after=now - args.leader_scan_ttl)
            if stale:
                await collector.remove_codes(stale)
                for c in stale:
                    hot.pop(c, None)
            added, rejected = await collector.add_codes(fresh)
            if rejected:
                # 수집이 잡은 자리를 뺏어서라도 붙인다 — 대장을 못 보면 그날 그
                # 종목은 없는 것이고, 저장은 내일 다시 할 수 있다.
                stolen, rejected = await _preempt_collect_slots(
                    collector, detector, guard, collect_taken, rejected)
                if stolen:
                    added = [*added, *stolen]
                    bus.notice(f"[대장스캔] 수집 자리 {len(stolen)}개 선점 — "
                               f"{', '.join(stolen[:5])}")
            enter = codes_to_enter(fresh, added, rejected)
            already = [c for c in enter if c not in added]
            if enter:
                det_added = detector.add_codes(enter, **_breaking_meta(args, enter))
                # **주문 화이트리스트도 같이 넓힌다.** 안 그러면 감지기는 신호를
                # 내는데 주문이 전부 차단된다 — 스캔을 붙여놓고 못 사는 상태다.
                _allow_orders(executor, det_added)
                names = ", ".join(f"{r.name}({r.code})" for r in leaders[:3])
                bus.notice(f"[대장스캔] 장중 편입 {len(det_added)}종목"
                           f"{f'(수집 선점 {len(already)})' if already else ''}"
                           f" — 상위 {names}")
            if rejected:
                # 수집 자리까지 뺏고도 모자란 상태다. 남는 건 매매가 쓰는 자리뿐이라
                # 유니버스(--universe-size)나 스캔 폭(--leader-scan-max)을 줄여야 한다.
                bus.notice(f"[대장스캔] 자리 없어 거절 {len(rejected)}종목 "
                           f"— 수집 자리까지 선점해도 부족하다. "
                           f"--universe-size 나 --leader-scan-max 를 줄여라")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 스캔 실패로 감지를 죽이지 않는다
            bus.notice(f"[대장스캔] 실패(계속): {exc}")


def _stale_scan_codes(hot: dict[str, float], base: set[str], guard: RiskGuard | None,
                      *, keep_after: float) -> list[str]:
    """내려도 되는 코드 — 스캐너가 붙였고, 장전 유니버스가 아니고, 식은 것.

    **열린 포지션이 있는 종목은 절대 안 내린다** — 구독을 끊으면 틱이 안 와서
    청산 판정이 멈춘다(대장이탈도 손절도 못 하고 그대로 물린다).
    """
    live: set[str] = set()
    if guard is not None:
        live = {p.follower for p in guard.open_positions} | {
            p.leader for p in guard.open_positions}
    return sorted(c for c, seen in hot.items()
                  if seen < keep_after and c not in base and c not in live)


def _allow_orders(executor: OrderExecutor | None, codes: Sequence[str]) -> None:
    """장중 편입 종목을 주문 화이트리스트에 더한다.

    감시 구독과 주문 허용은 **다른 목록**이다. 구독만 늘리면 감지기는 신호를 내는데
    실행기가 전부 차단한다 — 2026-08-27 실측에서 스캔이 11종목을 붙였는데 화이트리스트는
    장전 11종목 그대로였다.
    """
    if executor is None or not codes:
        return
    client = getattr(executor, "_client", None)
    allow = getattr(client, "allow_codes", None)
    if callable(allow):
        allow(list(codes))


def _group_key(code: str, detector: PairDetector, sectors: dict[str, str]) -> str:
    """그 종목이 속한 무리의 키 — 실전 테마가 있으면 테마 코드, 없으면 섹터."""
    themes = detector.themes_of(code)
    if themes:
        return f"T:{min(themes)}"
    return f"S:{sectors.get(code, '')}"


def _fetch_themes(conn: Any) -> dict[str, frozenset[str]]:
    """``theme_members`` 최신 스냅샷 → ``{종목: {테마코드}}``. 없으면 빈 dict.

    스냅샷은 매일 장 마감 뒤 `scalp-theme-snapshot` cron 이 쌓는다. 최신 하루만
    쓴다 — 테마 편입은 며칠 단위로 바뀌고, 옛 스냅샷을 섞으면 이미 빠진 종목이
    무리에 남는다.
    """
    try:
        cur = conn.cursor()
        cur.execute("select code, theme_code from theme_members where snapshot_date = "
                    "(select max(snapshot_date) from theme_members)")
        rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — 테마 테이블이 없어도 기동을 막지 않는다
        return {}
    grouped: dict[str, set[str]] = {}
    for code, theme in rows:
        grouped.setdefault(str(code), set()).add(str(theme))
    return {c: frozenset(t) for c, t in grouped.items()}


def _scan_sectors(args: argparse.Namespace, codes: list[str]) -> dict[str, str]:
    """스캔된 종목의 섹터. 못 읽으면 빈 dict(짝꿍은 못 붙고 대장만 붙는다)."""
    try:
        from .intraday.store import connect

        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        ph = _placeholder(conn)
        cur = conn.cursor()
        cur.execute(f"select code, sector from stocks where code in "
                    f"({','.join([ph] * len(codes))})", codes)
        return {str(c): str(sec) for c, sec in cur.fetchall() if sec}
    except Exception:  # noqa: BLE001 — 섹터 조회 실패로 스캔을 죽이지 않는다
        return {}


async def _watch_entry_cutoff(
    guard: RiskGuard | None, bus: AlertBus, args: argparse.Namespace,
) -> None:
    """오후가 되면 **신규 진입만** 막는다 — 청산은 계속 돈다.

    만쥬 통합 매뉴얼(49 §1-5): "오전 핵심 구간 매매 후 오후 추격 매매는 철저히 배제."
    45번 대조표에 "미결정"으로 남겨뒀던 자리를 이 답변이 확정지었다.

    시황 게이트와 **같은 경로**를 쓴다(`size_multiplier = 0`) — 이미 검증된 길이고,
    신규 진입만 막고 열린 포지션의 손절·대장이탈 청산은 그대로 돈다.
    """
    import asyncio

    from .realtime.leader_scan import past_cutoff

    if guard is None or not args.no_entry_after.strip():
        return
    while True:
        await asyncio.sleep(20.0)
        hhmm = datetime.now().strftime("%H%M")  # noqa: DTZ005 — KST naive
        if past_cutoff(hhmm, args.no_entry_after):
            guard.set_size_multiplier(0.0)
            bus.notice(f"[오후차단] {args.no_entry_after} 경과 — 신규 진입 금지"
                       f"(만쥬: 오후 추격 매매 배제). 청산은 계속 돈다")
            return


async def _watch_collect_fill(
    collector: Any, detector: PairDetector, bus: AlertBus, args: argparse.Namespace,
    conn: Any = None, movers: list[str] | None = None,
    guard: RiskGuard | None = None, taken: dict[str, float] | None = None,
) -> None:
    """감지기가 안 쓰는 등록 자리를 **수집 전용**으로 채운다.

    이건 만쥬 규칙이 아니라 우리 데이터 문제다. 감지기는 그날 대장·짝꿍 후보만
    구독하므로 후보가 적은 날은 정원 50 중 10~20자리만 쓴다 — 2026-08-28 실측
    구독 12종목·저장 틱 6만행(8/24 는 80종목·180만행). 사전등록 47번이 "굳힘을
    가르는 정보는 일봉에 없고 장중 미시구조에 있다"로 끝났는데, 그 데이터를
    모으는 자리를 비워두는 건 손해다.

    **여기서 붙인 종목은 감지기에 등록하지 않는다** — 저장만 되고 신호도 주문도
    나오지 않는다. 감지기가 대장을 붙일 자리(`--collect-reserve`)는 남긴다.
    """
    import asyncio

    from .realtime.collect_fill import fill_pool, plan_evictions, plan_fill

    if conn is None:
        bus.notice("[수집채움] DB 연결 없음 → 비활성")
        return
    if movers is None:
        movers = []
    try:
        cur = conn.cursor()
        ph = _placeholder(conn)          # postgres·sqlite 둘 다 돈다
        cur.execute(
            "select code from daily_bars where date = (select max(date) from daily_bars)"
            f" and trade_value > 0 order by trade_value desc limit {ph}",
            (args.collect_fill_pool,))
        pool = [r[0] for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — 수집 채움 실패로 매매를 죽이지 않는다
        bus.notice(f"[수집채움] 후보 조회 실패 → 비활성: {exc}")
        return
    if not pool:
        bus.notice("[수집채움] 후보 0종목 → 비활성")
        return
    # 수집이 붙인 종목 → 마지막으로 순위에 보인 시각. **호출부와 공유한다** —
    # 대장스캔·돌발재료가 자리가 없을 때 이 목록에서 뺏어 간다(_preempt_collect_slots).
    if taken is None:
        taken = {}
    loop = asyncio.get_running_loop()
    # 고정 수집(--collect-pin) — 80번 기각 뒤 매도세 면제 ETF 의 틱을 모은다. 전일
    # 거래대금 풀(daily_bars 는 주식만)에 없으니 후보 맨 앞에 박고, 매 바퀴 "방금 봤다"로
    # 갱신해 ttl 로 안 내려가게 한다. 매매가 자리를 달라면(_preempt_collect_slots)
    # 가장 뜨거운 것으로 보여 맨 나중에 내주고, 다음 바퀴에 다시 붙는다.
    pin_inline = getattr(args, "collect_pin", "")
    pin_path = getattr(args, "collect_pin_file", "")
    pins = _pin_codes(pin_inline, pin_path)

    # 채우고 나서 쉰다 — 장 시작 직후가 가장 아까운 구간이라 첫 사이클을 미루지 않는다.
    # 소켓이 아직 안 붙었으면 add_codes 가 조용히 빈 손으로 돌아오고 다음 바퀴에 다시 본다.
    while True:
        try:
            now = loop.time()
            if pin_path:
                # 매 바퀴 다시 읽는다 — 다른 전략이 장중에 신호 종목을 더할 수 있게 한다.
                # 저장 전용이라 매매 판단에는 영향이 없고, 파일이 사라지면 인자 목록으로 돌아간다.
                pins = _pin_codes(pin_inline, pin_path)
            for code in [*movers, *pins]:
                if code in taken:
                    taken[code] = now       # 아직 순위에 있다 — 유예를 갱신한다
            # 순위에서 사라진 지 오래된 건 자리를 내준다. 안 그러면 오전에 붙은
            # 종목이 하루를 다 먹고 오후에 올라오는 대장을 못 본다.
            # 매매가 쓰는 종목은 절대 안 내린다 — 수집이 붙인 종목이 대장·짝꿍으로
            # 승격되는 일이 흔하고, 그때 구독을 끊으면 신호가 죽고 포지션이 있으면
            # 청산 판정이 멈춘다(손절도 대장이탈도 못 하고 물린다).
            keep = set(detector.codes)
            if guard is not None:
                for pos in guard.open_positions:
                    keep.add(pos.follower)
                    keep.add(pos.leader)
            stale = plan_evictions(taken, now=now, ttl=args.collect_fill_ttl,
                                   keep=keep)
            if stale:
                await collector.remove_codes(stale)
                for code in stale:
                    taken.pop(code, None)
            subscribed = collector.subscribed_codes
            # 감지기가 아는 종목은 후보에서 뺀다 — 그쪽은 매매 자리다.
            want = [c for c in fill_pool([*pins, *movers], pool)
                    if c not in detector.codes and c not in taken]
            take = plan_fill(want, subscribed,
                             cap=collector.code_capacity,
                             reserve=args.collect_reserve)
            added: list[str] = []
            if take:
                added, _ = await collector.add_codes(take)
            if added:
                for code in added:
                    taken[code] = now
                bus.notice(f"[수집채움] {len(added)}종목 추가 — 구독 "
                           f"{len(subscribed)}/{collector.code_capacity}"
                           f"(예비 {args.collect_reserve})")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 수집 채움 실패로 매매를 죽이지 않는다
            bus.notice(f"[수집채움] 실패(계속): {exc}")
        # 하한 0.5초는 설정 실수(0)로 바쁜 루프가 되는 걸 막는 용도다.
        # 실제 운영값은 cron 의 60초다.
        await asyncio.sleep(max(0.5, args.collect_fill_interval))


async def _watch_breaking_materials(
    collector: Any, detector: PairDetector, bus: AlertBus, args: argparse.Namespace,
    executor: OrderExecutor | None = None, guard: RiskGuard | None = None,
    collect_taken: dict[str, float] | None = None,
) -> None:
    """장중 돌발 재료 감시 — 만쥬 §6.

    > "예상치 못한 돌발 뉴스가 나올 경우 **즉각적인 판단으로 관련 종목 매수 대응**"

    장전 유니버스에 없던 종목에 강한 재료 공시가 뜨면, 그 종목을 **장중에 구독에
    추가**한다. 구독이 없으면 감시조차 못 하기 때문이다.

    뉴스가 아니라 공시를 쓰는 이유는 재료 판정과 같다(크롤러 금지 정책,
    docs/research/manju/17). DART 는 장중에도 계속 접수되므로 주기적으로 새로
    수집하고, 접수번호를 커서로 써서 **새로 들어온 것만** 본다.
    """
    import asyncio

    from .universe import breaking_materials, fetch_new_disclosures

    day = date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
    cursor: str | None = None
    # 시작 시점의 접수번호를 커서로 잡아 **장 시작 전 공시는 건너뛴다**(그건 장전
    # 유니버스 선정에 이미 반영됐다). 여기서 보는 건 '장중에 새로 뜬 것'뿐이다.
    seed = fetch_new_disclosures(args.dart_db, day=day)
    if seed:
        cursor = seed[-1][0]
    while True:
        await asyncio.sleep(max(30.0, args.breaking_interval))
        try:
            rows = fetch_new_disclosures(args.dart_db, day=day, after_rcept=cursor)
            if not rows:
                continue
            codes = breaking_materials([(c, m) for _, c, m in rows],
                                       min_weight=args.breaking_min_weight)
            fresh = [c for c in codes if c not in detector.codes]
            take, defer = fresh[:args.breaking_max], fresh[args.breaking_max:]
            # 커서는 **이번에 실제로 처리한 데까지만** 전진시킨다. 먼저 끝까지 밀면
            # 상한(breaking_max)에 걸려 못 붙인 종목이 영영 후보에 다시 안 오른다.
            cursor = _advance_cursor(rows, take, defer, cursor)
            if not take:
                continue
            added, rejected = await collector.add_codes(take)
            if rejected:
                # §6 의 돌발 재료는 그 순간을 놓치면 끝이다 — 수집 자리를 뺏는다.
                stolen, rejected = await _preempt_collect_slots(
                    collector, detector, guard, collect_taken, rejected)
                if stolen:
                    added = [*added, *stolen]
            # **구독만 늘리면 아무 일도 안 일어난다** — 감지기가 모르는 코드의
            # 틱은 on_tick 첫 줄에서 버려진다. 섹터·전일종가를 같이 넣어야
            # 그 종목이 대장이 될 수 있다(§6 의 목적). 수집 채움이 먼저 붙여둔
            # 종목도 챙긴다(codes_to_enter).
            enter = codes_to_enter(take, added, rejected)
            if enter:
                meta = _breaking_meta(args, enter)
                det_added = detector.add_codes(enter, **meta)
                _allow_orders(executor, det_added)
                bus.notice(f"[돌발재료] 장중 편입 {len(det_added)}종목: "
                           f"{', '.join(det_added)} — 강한 재료 공시 접수")
            if rejected:
                bus.notice(f"[돌발재료] 수집 자리까지 선점하고도 놓침 "
                           f"{len(rejected)}종목: "
                           f"{', '.join(rejected[:5])}")
        except Exception as exc:  # noqa: BLE001 — 감시 실패로 수집을 죽이지 않는다
            print(f"[돌발재료][경보] 감시 중 예외(계속): {exc}", file=sys.stderr)


def _drain_leader_events(detector: PairDetector, guard: RiskGuard | None,
                         executor: OrderExecutor | None) -> None:
    """감지기가 판정한 대장 상태 전이를 가드·실행기로 흘린다(만쥬 §5-3).

    이게 없으면 만쥬 방식의 절반만 돈다 — 남는 절반은 이미 기각된 고정 % 청산이다.
    가드와 실행기에 **같은 값**을 먹여야 회계와 실주문이 갈리지 않는다.
    """
    for code, ts, state in detector.drain_leader_events():
        if guard is not None:
            guard.on_leader_state(code, ts, state)
        if executor is not None:
            executor.on_leader_state(code, ts, state)


def _drain_filtered(detector: PairDetector, bus: AlertBus) -> None:
    """호가단위 필터로 버린 신호를 알림 로그에 남긴다.

    조용히 버리면 며칠 뒤 "필터가 과했나"를 셀 수 없다. 알림이 아니므로 ``notice``
    로 흘려 주문 경로에는 절대 닿지 않게 한다.
    """
    for rec in detector.drain_filtered():
        bus.notice(rec.to_line())


def _drain_price_anomalies(detector: PairDetector, bus: AlertBus) -> None:
    """직전 틱 대비 급변한 가격을 알림 로그에 남긴다(2026-09-07 원인미상 사례 대응).

    판정은 안 막는다 — 조용히 넘어가면 재발해도 못 잡으니 경고만 남긴다.
    """
    for rec in detector.drain_price_anomalies():
        bus.notice(
            f"[가격이상] {rec['code']} {rec['ts']:%H:%M:%S} {rec['price']:,.0f}원"
            f" (직전 {rec['prev_ts']:%H:%M:%S} {rec['prev_price']:,.0f}원,"
            f" {rec['gap_sec']:.1f}초 간격, {rec['pct']:+.1%}) — 확인 필요"
        )


def _emit_guarded(alert: Alert, bus: AlertBus, guard: RiskGuard | None,
                  poslog: PositionLogSink | None,
                  executor: OrderExecutor | None = None) -> bool:
    """알림을 가드에 태워 통과분만 emit. 억제되면 배너·킬 로그를 남긴다. emit 여부.

    **주문 발주는 오직 가드 통과(allowed) 경로에서만** — 감사 요구(가드와 동일 게이트).
    킬 발동 시 실행기의 대기 매수를 취소한다.
    """
    if guard is None:
        bus.emit(alert)
        if executor is not None:
            executor.on_signal(alert)
        return True
    decision = guard.on_alert(alert)
    if decision.banner:
        bus.notice(decision.banner)
        if poslog is not None:
            poslog.log_kill(alert.ts, guard.kill_reason, guard.realized_krw)
        if executor is not None:
            executor.on_kill()
    if not decision.allowed:
        return False
    bus.emit(alert)
    if poslog is not None:
        poslog.log_open(alert)
    if executor is not None:                 # 동일 게이트 통과분만 주문
        executor.on_signal(alert)
    return True


def _feed(detector: PairDetector, bus: AlertBus, rows: list[tuple],
          guard: RiskGuard | None = None,
          poslog: PositionLogSink | None = None,
          executor: OrderExecutor | None = None) -> int:
    n = 0
    for code, ts, price, volume, best_bid, side in rows:
        if price is None:
            continue
        t = _as_dt(ts)
        p = float(price)
        # 가드는 매 틱으로 열린 가상 포지션의 체결·MAE·청산을 리허설한다.
        if guard is not None:
            guard.on_tick(str(code), t, p)
        if executor is not None:             # 실행기: 체결·미체결취소·청산
            executor.on_tick(str(code), t, p)
        for alert in detector.on_tick(str(code), t, p, bid1=best_bid,
                                       volume=volume, side=side):
            if _emit_guarded(alert, bus, guard, poslog, executor):
                n += 1
        _drain_leader_events(detector, guard, executor)
        _drain_filtered(detector, bus)
        _drain_price_anomalies(detector, bus)
    return n


def build_order_executor(
    args: argparse.Namespace, pairs: list[Pair], *, client: Any | None = None,
    universe: Iterable[str] = (),
) -> OrderExecutor | None:
    """``--live-order`` 일 때만 실계좌 주문 실행기를 만든다.

    이중 안전: ``--live-order`` 는 실행기를 켤 뿐 어댑터는 **dry-run** 이라 실주문이
    안 나간다. 진짜 발주는 ``--confirm-real`` 까지 있어야(어댑터 ``live=True``).
    시작 시 계좌 자동조회 + 보유 전종목 블록리스트 병합(fail-closed).

    ``client`` 를 주면 인증·계좌조회를 건너뛰고 그걸 쓴다 — 배선 스모크 테스트용이다.
    이 함수를 부르는 테스트가 하나도 없어서 2026-08-26 에 인자 오배선을 못 잡았고,
    ``--live-order`` 가 TypeError 로 죽는 상태로 커밋됐다.
    """
    if not args.live_order:
        return None
    from .adapters.kiwoom_order import (
        OrderGuardConfig,
    )

    # 주문 화이트리스트 — **후속주가 될 수 있는 종목 전부**여야 한다.
    # 만쥬는 고정 쌍을 안 쓴다. 후속주는 장중에 "대장과 같은 테마 등락률 2·3등"으로
    # 정해지므로, 등록 쌍만 허용하면 동적 모드에서 화이트리스트가 비어 **모든 주문이
    # 차단된다**(고정 쌍을 지우자 실제로 0종목이 됐다). 그날 유니버스를 합친다 —
    # 감지기가 후속주로 고를 수 있는 범위와 정확히 같은 집합이다.
    whitelist = frozenset(p.follower for p in pairs) | frozenset(
        str(c) for c in universe if c)
    guard = OrderGuardConfig(
        max_qty=args.qty,
        price_band_pct=args.order_band_pct,
        max_orders_per_code=args.max_orders_per_code,
        max_orders_total=args.max_orders_total,
        whitelist=whitelist,
    )
    if client is not None:
        # 주입된 클라이언트 — 인증·계좌조회를 건너뛴다(배선 스모크 테스트용).
        held = ()
    else:
        client, held = _connect_order_client(args, guard)
        if client is None:
            return None
    real = args.confirm_real
    mode = "🔴 실발주(LIVE)" if real else "🧪 DRY-RUN(발주 안 함)"
    print(f"# 주문 실행기 ON — {mode} · 계좌 {client.masked_account()} · 수량 {args.qty}주 "
          f"· 화이트리스트 {len(whitelist)}종목 · 보유차단 {len(held)}종목 "
          f"(하드 3 + 실시간) · 가격밴드 ±{args.order_band_pct:.1%}"
          + ("  ← --confirm-real 켜짐! 진짜 주문이 나갑니다." if real else ""),
          file=sys.stderr, flush=True)
    return OrderExecutor(client, qty=args.qty, live=real,
                         unfilled_timeout=args.unfilled_timeout,
                         fill_basis=args.fill_basis,   # 가드 회계와 같은 체결 가정
                         leader_exit=not args.no_leader_exit,
                         size_multiplier=args.size_multiplier,
                         # 일일 이익 절제(§9-6)는 **가드 전용**이다 — 실행기는 그 개념을
                         # 갖지 않는다. 여기에 넘기면 TypeError 로 기동이 죽는다.
                         partial_exit_ratio=args.partial_exit,
                         partial_exit_gain=args.partial_exit_gain,
                         partial_exit_after_s=args.partial_exit_after,
                         pyramid_arms=tuple(sorted(args.pyramid_arm or ())),
                         pyramid_add_qty=args.pyramid_add_qty,
                         notice=lambda s: print(s, flush=True),
                         # 장부는 **실발주일 때만** — dry-run 모의 체결이 실계좌 journal 처럼
                         # data/positions 에 쌓이면 사후 대사가 거짓 포지션을 본다.
                         book=_open_position_book() if real else None)


def _open_position_book() -> Any | None:
    """실행기 체결 장부(krx-quant-core ``PositionBook``) — 날짜별 journal, **기록 전용**.

    같은 날 재시작(런처의 크래시 재시작)이면 journal 을 재생해 이어 쓴다. 어떤 이유로든
    못 열면 장부만 끄고(None) 경고한다 — 장부 때문에 08:55 기동이 죽으면 안 된다.
    세율은 2026 기준 KOSPI·KOSDAQ 이 같아(0.20%) 시장 구분 없이 KOSDAQ 으로 계산한다
    (``market_of`` 하드코딩 — 세율이 갈리는 해가 오면 종목 시장 조회로 바꿔야 한다).

    한계(기록 전용이라 감수): 매수 기록가는 체결 확인 시점의 **지정가**(``pos.entry``)라
    실제 체결가와 다를 수 있다. 재시작 재생은 journal 만 보므로, 재시작 전에 들고 있던
    포지션을 새 프로세스가 인수하지 않으면(실행기는 메모리 상태라 인수하지 않는다) 장부에
    그 보유가 청산되지 않은 채 남는다 — 계좌 잔고가 정본이다.
    """
    path = Path("data/positions") / f"{date.today().isoformat()}.jsonl"  # noqa: DTZ011
    try:
        from krx_quant_core.execution.book import PositionBook

        return PositionBook.load(path, market_of=lambda _code: "KOSDAQ")
    except Exception as exc:  # noqa: BLE001 — 장부는 부가 기능, 기동을 막지 않는다
        print(f"[경고] 체결 장부를 못 열었다({path}: {exc}) — 장부 없이 진행한다",
              file=sys.stderr, flush=True)
        return None


def _connect_order_client(args: argparse.Namespace, guard: Any) -> tuple[Any, Any]:
    """인증 → 클라이언트 → 계좌·잔고 조회. 실패하면 ``(None, ())`` (fail-closed)."""
    from .adapters.kiwoom import KiwoomError, api_credentials
    from .adapters.kiwoom_order import (
        KIWOOM_BASE,
        KIWOOM_MOCK_BASE,
        KiwoomOrderClient,
        OrderLogSink,
    )

    # 전용 키 우선(감지 소켓과 동일 정책).
    env = dict(os.environ)
    if os.environ.get("PAIR_DETECT_APP_KEY"):
        env["KIWOOM_APP_KEY"] = os.environ["PAIR_DETECT_APP_KEY"]
        env["KIWOOM_APP_SECRET"] = os.environ.get("PAIR_DETECT_APP_SECRET", "")
    try:
        appkey, secret = api_credentials(env)
    except KiwoomError as exc:
        print(f"[주문] 인증 정보 없음 → 주문 실행기 비활성: {exc}", file=sys.stderr)
        return None, ()
    base = KIWOOM_MOCK_BASE if args.mock else KIWOOM_BASE
    log = OrderLogSink(args.order_log)
    client = KiwoomOrderClient(
        appkey, secret, guard=guard, base_url=base,
        on_order=lambda r: log.emit(r, timing=client.last_timing),
    )
    # 계좌 자동조회 + 실시간 보유 병합(fail-closed): 실패하면 주문 실행기를 끈다.
    try:
        client.account_number()
        held = client.refresh_blocklist_from_account()
    except KiwoomError as exc:
        print(f"[주문] 계좌/잔고 조회 실패 → 안전상 주문 실행기 비활성(fail-closed): {exc}",
              file=sys.stderr)
        return None, ()
    return client, held


def build_poslog(args: argparse.Namespace, guard: RiskGuard | None) -> PositionLogSink | None:
    """포지션 로그 sink 를 만들고 가드 청산 콜백에 연결한다."""
    if guard is None or args.dry_run:
        return None
    poslog = PositionLogSink(args.position_log, basis=args.fill_basis)
    guard.attach_sink(poslog)
    return poslog


# --------------------------------------------------------------------- 모드

def run_replay(args: argparse.Namespace, conn: Any, detector: PairDetector,
               guard: RiskGuard | None) -> int:
    day = args.replay
    print(f"# 리플레이 {day} — 등록 쌍 {len(detector.pairs)}개 재현 (dry-run)", flush=True)
    bus = AlertBus(dry_run=True)
    rows = _fetch_ticks(conn, sorted(detector.codes), since=None, day=day)
    print(f"틱 {len(rows):,}건 로드. 시간순 흘림 …", flush=True)
    n = _feed(detector, bus, rows, guard=guard)
    if guard is not None:
        guard.flush_expired(_as_dt(rows[-1][1]) if rows else datetime.now())  # noqa: DTZ005
        print("\n" + guard.status_line(), flush=True)
        c = guard.closed_positions
        filled = [p for p in c if p.status == "closed"]
        print(f"가상 포지션 {len(c)}건(회계 {guard.config.fill_basis}): "
              f"체결 {len(filled)} · 미체결 {len(c) - len(filled)} · "
              f"킬스위치 {'발동(' + guard.kill_reason + ')' if guard.killed else '미발동'}.",
              flush=True)
    print(f"\n통과 짝꿍 알림 {n}건 (쌍별 쿨다운 {detector.config.cooldown:.0f}s 적용).", flush=True)
    return 0


def run_realtime(args: argparse.Namespace, conn: Any, detector: PairDetector,
                 guard: RiskGuard | None, executor: OrderExecutor | None = None) -> int:
    codes = sorted(detector.codes)
    bus = build_default_bus(log_path=args.log, bell=args.bell,
                            discord=not args.no_discord, dry_run=args.dry_run)
    poslog = build_poslog(args, guard)
    # 워밍업: 최근 lookback 초의 틱을 먼저 먹여 롤링 창을 채운다.
    now = datetime.now()  # noqa: DTZ005 — KST 로컬
    warm_since = now.timestamp() - args.lookback
    warm_dt = datetime.fromtimestamp(warm_since)  # noqa: DTZ006
    warm = _fetch_ticks(conn, codes, since=warm_dt, day=None)
    # 워밍업은 알림을 내지 않게 조용히 상태만 채운다.
    for code, ts, price, volume, best_bid, side in warm:
        if price is not None:
            detector.on_tick(str(code), _as_dt(ts), float(price), bid1=best_bid,
                             volume=volume, side=side)
    watermark = max((_as_dt(r[1]) for r in warm), default=warm_dt)
    print(f"# 실시간 감지 시작 — 쌍 {len(detector.pairs)}개, 종목 {len(codes)}개, "
          f"폴링 {args.poll_interval}s, 로그 {args.log}"
          + (f", 가드 {'켬' if guard else '끔'}" )
          + (" [dry-run]" if args.dry_run else ""), flush=True)
    deadline = _parse_deadline(args.duration)
    try:
        while True:
            try:
                rows = _fetch_ticks(conn, codes, since=watermark, day=None)
                if rows:
                    _feed(detector, bus, rows, guard=guard, poslog=poslog, executor=executor)
                    watermark = max(_as_dt(r[1]) for r in rows)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — N-2: 한 틱배치 오류로 감지 루프 죽지 않게
                print(f"[감지][경보] 틱 처리 중 예외 — 루프 계속: {exc}",
                      file=sys.stderr, flush=True)
            if deadline is not None and datetime.now() >= deadline:  # noqa: DTZ005
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n중단.", file=sys.stderr)
    if guard is not None:
        guard.flush_expired(datetime.now())  # noqa: DTZ005
        print(guard.status_line(), flush=True)
    if executor is not None:
        executor.flush(datetime.now())  # noqa: DTZ005
        print(executor.summary(), flush=True)
    print(f"누적 알림 {bus.count}건.", flush=True)
    return 0


def run_live(args: argparse.Namespace, detector: PairDetector,
             guard: RiskGuard | None, executor: OrderExecutor | None = None) -> int:
    """소켓 직결 모드(실전 주경로) — 수집 데몬과 한 프로세스로, 틱 도착 즉시 판정.

    ``KiwoomRealtimeCollector`` 를 띄우되 ``on_frame`` 콜백에서 프레임을 파싱해 곧장
    :meth:`PairDetector.on_tick`/``on_quote`` 로 흘린다. DB 폴링을 안 거쳐 지연 ~0.
    감지 뒤(같은 프레임) collector 가 ``ticks``·``quotes`` 를 저장한다(--no-store 로 끔).

    **App key 1세션 제약**(어제 실측: 같은 키 두 소켓 → thrash·105110): 프로덕션
    tick-collect 데몬과 같은 키로 이 모드를 동시에 돌릴 수 없다. 두 해법:
      1. **전용 키**(권장): ``PAIR_DETECT_APP_KEY``/``PAIR_DETECT_APP_SECRET`` 를 두면
         그 키로 붙는다. 등록 쌍의 대장+후속(~8종목)만 구독하니 커넥션 1개로 충분하고
         프로덕션 수집과 충돌하지 않는다.
      2. **수집 대체**: 전용 키가 없으면 기본 ``KIWOOM_APP_KEY`` 로 붙되, 이 프로세스가
         곧 그 키의 유일한 세션이어야 한다 — tick-collect 데몬을 내리고 --live 로
         대체(감지+저장 겸함)하라. 안 내리면 서로 kick·재연결을 반복한다.

    **정원 2배 확장(2026-09-07)**: 위 두 해법 다 앱키가 하나라 정원(≈50종목)은
    그대로다. ``KIWOOM_APP_KEY_2``/``KIWOOM_APP_SECRET_2`` 를 두면
    :class:`~scalp_it.realtime.collector.DualKeyCollector` 로 커넥션을 하나 더
    열어 정원을 두 배(≈100종목)로 늘린다 — 09-07 커버리지 확인에서 급등 9종목 중
    3개·12%+ 31종목 중 15개를 구독 자체를 못 해서 놓친 것이 계기다. 두 커넥션
    모두 같은 ``on_frame`` 으로 흘러 같은 감지기·가드·실행기를 본다.
    """
    from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
    from .realtime.collector import (
        TYPE_QUOTE,
        TYPE_TRADE,
        WS_URL,
        WS_URL_MOCK,
        DualKeyCollector,
        KiwoomRealtimeCollector,
        SeqAssigner,
        codes_per_connection,
        parse_real_message,
        quote_levels,
    )
    from .realtime.store import init_realtime_tables

    url = WS_URL_MOCK if args.mock else WS_URL
    # ⚠️ 1h(VI) 는 REG 하지 않는다 — 이 데몬이 tick-collect 를 대체한 뒤로(2026-08-24~)
    # vi_events 가 한 건도 안 쌓인 원인이다(2026-10-12 감사). 1h 를 넣으면 종목당 등록
    # 3건이라 커넥션 정원이 50→33종목으로 준다 — 켤지는 정원과 맞바꾸는 결정이다.
    types = [TYPE_TRADE, TYPE_QUOTE]
    codes_all = sorted(detector.codes)
    store = not args.no_store and not args.dry_run
    bus = build_default_bus(log_path=args.log, bell=args.bell,
                            discord=not args.no_discord, dry_run=args.dry_run)
    poslog = build_poslog(args, guard)

    # 전용 키 우선 — 없으면 기본 키(그 경우 tick-collect 를 대체해야 함).
    env = dict(os.environ)
    dedicated = os.environ.get("PAIR_DETECT_APP_KEY")
    if dedicated:
        env["KIWOOM_APP_KEY"] = dedicated
        env["KIWOOM_APP_SECRET"] = os.environ.get("PAIR_DETECT_APP_SECRET", "")
        print("# 전용 키(PAIR_DETECT_APP_KEY)로 접속 — 프로덕션 수집과 충돌 없음", file=sys.stderr)
    else:
        print("# 기본 키(KIWOOM_APP_KEY)로 접속 — App key 1세션 제약: tick-collect 데몬을 "
              "내리고 이 프로세스로 대체하라(감지+저장 겸함). 전용 키를 두면 동시 가능.",
              file=sys.stderr)
    try:
        appkey, secret = api_credentials(env)
    except KiwoomError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    # 두 번째 앱키(KIWOOM_APP_KEY_2/KIWOOM_APP_SECRET_2) 가 있으면 정원을 2배로.
    # 2026-09-07 사용자 결정 — 09-07 커버리지 확인(급등 9종목 중 3개, 12%+ 31종목
    # 중 15개를 구독 자체를 못 해서 놓침)에서 나온 조치. 앱키가 다르면 독립
    # 세션이라 같은 키로 커넥션을 늘릴 때의 kick 문제가 없다(모듈 docstring 참고).
    key2 = os.environ.get("KIWOOM_APP_KEY_2", "").strip()
    secret2 = os.environ.get("KIWOOM_APP_SECRET_2", "").strip()
    dual_key = bool(key2 and secret2)

    token = KiwoomClient(appkey, secret).token
    token2 = KiwoomClient(key2, secret2).token if dual_key else None

    conn = None
    conn2 = None
    if store:
        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        init_realtime_tables(conn)
        if dual_key:
            conn2 = connect(args.dsn or os.environ.get("KR_QUANT_DB"))

    if dual_key:
        cap = codes_per_connection(types)
        codes_a, codes_b = codes_all[:cap], codes_all[cap:cap * 2]
        if len(codes_all) > cap * 2:
            print(f"[경고] 종목 {len(codes_all)}개 > 앱키 2개 정원 {cap * 2}개 — "
                  f"앞의 {cap * 2}개만 구독한다.", file=sys.stderr, flush=True)
        print(f"# 앱키 2개 사용 — 정원 {cap} x 2 = {cap * 2}", file=sys.stderr)
    else:
        codes_a = _cap_codes_for_single_key(codes_all, types, dedicated=bool(dedicated))
        codes_b = []

    trade_date = date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
    seq = SeqAssigner()
    etf_shadow = _build_etf_shadow(args, trade_date)
    tiger_shadow, tiger_codes = _build_tiger_shadow(args, trade_date)
    rl_shadow, rl_codes = _build_rl_shadow(args, trade_date)

    def on_frame(msg: dict) -> None:
        # N-2: 한 프레임 처리 중 어떤 예외도 소켓 루프(다른 실포지션 관리)를 죽이지
        # 못하게 감싼다. 주문 거부는 실행기 내부에서 이미 포획하지만, 파싱·판정 등
        # 어느 단계의 예외든 방어한다.
        try:
            now = datetime.now()  # noqa: DTZ005 — KST naive
            ticks, quotes, _vis = parse_real_message(msg, trade_date, seq, received_at=now)
            if etf_shadow is not None:
                _feed_etf_shadow(etf_shadow, ticks, quotes)
            if tiger_shadow is not None:
                _feed_tiger_shadow(tiger_shadow, tiger_codes, ticks, quotes)
            if rl_shadow is not None:
                _feed_rl_shadow(rl_shadow, rl_codes, ticks, quotes)
            for q in quotes:                      # 호가로 bid1 우선 갱신
                bids, asks = quote_levels(q)
                detector.on_quote(str(q["code"]), q["ts"], q.get("bid1"),
                                  bids=bids, asks=asks)
            for t in ticks:                       # 체결로 서지·매칭 판정
                if t.get("price") is None:
                    continue
                code_s, ts_t, px_t = str(t["code"]), t["ts"], float(t["price"])
                if guard is not None:             # 가드: 체결·MAE·청산 리허설
                    guard.on_tick(code_s, ts_t, px_t)
                if executor is not None:          # 실행기: 체결·미체결취소·청산
                    executor.on_tick(code_s, ts_t, px_t)
                for alert in detector.on_tick(code_s, ts_t, px_t,
                                              bid1=t.get("best_bid"), volume=t.get("volume"),
                                              side=t.get("side")):
                    _emit_guarded(alert, bus, guard, poslog, executor)
                _drain_leader_events(detector, guard, executor)
                _drain_filtered(detector, bus)
                _drain_price_anomalies(detector, bus)
        except Exception as exc:  # noqa: BLE001 — 프레임 오류로 소켓 감지 죽지 않게
            print(f"[감지][경보] 프레임 처리 중 예외 — 소켓 루프 계속: {exc}",
                  file=sys.stderr, flush=True)

    spool_base = args.spool_dir or None
    collector_a = KiwoomRealtimeCollector(
        token, codes_a, conn=conn, dsn=(args.dsn if store else None),
        types=types, url=url, store=store, trade_date=trade_date,
        on_frame=on_frame, flush_interval=args.flush_interval,
        spool_dir=f"{spool_base}/a" if spool_base else None,
    )
    if dual_key:
        collector_b = KiwoomRealtimeCollector(
            token2, codes_b, conn=conn2, dsn=(args.dsn if store else None),
            types=types, url=url, store=store, trade_date=trade_date,
            on_frame=on_frame, flush_interval=args.flush_interval,
            log=lambda m: print(f"[key2]{m}", flush=True),
            spool_dir=f"{spool_base}/b" if spool_base else None,
        )
        collector = DualKeyCollector(collector_a, collector_b)
    else:
        collector = collector_a
    n_codes = len(codes_a) + len(codes_b)
    duration = _duration_seconds(args.duration)
    print(f"# 소켓 직결 감지 시작 — 쌍 {len(detector.pairs)}개, 종목 {n_codes}개, "
          f"저장 {'켬' if store else '끔'}, 로그 {args.log}"
          + (" [dry-run]" if args.dry_run else ""), file=sys.stderr, flush=True)

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, collector.stop)
            except (NotImplementedError, ValueError):
                pass
        watchers: list[Any] = []
        # 스캐너가 받아온 등락률 순위를 수집 채움이 함께 본다 — 같은 프로세스라
        # 목록 하나를 공유하면 된다(스캐너가 쓰고 채움이 읽기만 한다).
        movers: list[str] = []
        # 수집채움이 잡은 자리 — 매매(대장스캔·돌발재료)가 자리가 없으면 여기서
        # 뺏어 간다. 2026-08-31 에 수집이 정원을 먹고 대장스캔이 233회 거절당했다.
        collect_taken: dict[str, float] = {}
        if args.program_interval > 0:
            # 만쥬 — 프로그램이 쏟아내는 장세면 신호가 떠도 안 산다.
            watchers.append(asyncio.create_task(
                _watch_program_flow(guard, bus, args)))
        if args.leader_scan_interval > 0:
            # 만쥬 §2 — 시장 전체에서 대장을 찾는다. 이게 없으면 전날 고른 45종목
            # 안에서만 대장을 기다리게 되고, 실측 커버리지는 2% 였다.
            watchers.append(asyncio.create_task(
                _watch_leader_scan(collector, detector, bus, args, guard, executor,
                                   movers, collect_taken)))
        if args.no_entry_after.strip():
            # 만쥬 49 §1-5 — 오후 추격 매매 배제. 신규 진입만 막고 청산은 돈다.
            watchers.append(asyncio.create_task(
                _watch_entry_cutoff(guard, bus, args)))
        if args.collect_fill_interval > 0:
            # 매매가 안 쓰는 자리로 틱을 쌓는다. 감지기에 등록하지 않으므로
            # 신호·주문 경로에는 한 줄도 안 닿는다(tests/test_collect_fill.py).
            watchers.append(asyncio.create_task(
                _watch_collect_fill(collector, detector, bus, args, conn, movers,
                                    guard, collect_taken)))
        watcher = None
        if args.breaking_interval > 0:
            # 만쥬 §6 — 장중 돌발 재료. 수집과 나란히 돌고, 죽어도 수집은 계속된다.
            watcher = asyncio.create_task(
                _watch_breaking_materials(collector, detector, bus, args, executor,
                                          guard, collect_taken))
        try:
            await collector.run(duration=duration)
        finally:
            if watcher is not None:
                watcher.cancel()
            for task in watchers:
                task.cancel()

    try:
        asyncio.run(_main())
    finally:
        if conn is not None:
            conn.close()
        if conn2 is not None:
            conn2.close()
    if executor is not None:
        executor.flush(datetime.now())  # noqa: DTZ005
        print(executor.summary(), file=sys.stderr, flush=True)
    if rl_shadow is not None:
        try:
            rl_shadow.stop()
            reason = getattr(rl_shadow.shadow, "disabled_reason", None)
            print(f"[RL그림자] 종료 — 버린 프레임 {rl_shadow.dropped:,}건"
                  + (f" · 꺼져 있었다: {reason}" if reason else ""), file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[RL그림자] 마감 실패: {exc}", file=sys.stderr, flush=True)
    if tiger_shadow is not None:
        try:
            tiger_shadow.flush()
        except Exception as exc:  # noqa: BLE001 — 그림자 마감 실패로 종료 요약을 막지 않는다
            print(f"[호랑이그림자] 마감 실패: {exc}", file=sys.stderr, flush=True)
    if etf_shadow is not None:
        try:
            etf_shadow.flush(datetime.now())  # noqa: DTZ005
        except Exception as exc:  # noqa: BLE001 — 그림자 마감 실패로 종료 요약을 막지 않는다
            print(f"[ETF그림자] 마감 실패: {exc}", file=sys.stderr, flush=True)
    print(f"누적 알림 {bus.count}건 · 틱필터 배제 {detector.filtered_count}건"
          f" · 프레임 {collector.frames_seen:,}"
          f" · 저장 틱 {collector.ticks_stored:,}", file=sys.stderr, flush=True)
    return 0


def _build_etf_shadow(args: argparse.Namespace, trade_date: date) -> Any:
    """81번 ETF 체결강도 그림자 — 문턱을 못 만들면 None(감지·매매는 그대로 돈다).

    주문 경로와 무관하다. 가상 체결을 ``--etf-shadow-log`` 에 JSON 한 줄씩 남길 뿐이다.
    """
    codes = [c.strip() for c in str(getattr(args, "etf_shadow", "") or "").split(",") if c.strip()]
    if not codes:
        return None
    try:
        import json

        from .realtime.etf_strength import CELLS, StrengthShadow, load_thresholds

        tconn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        try:
            thr = load_thresholds(tconn, codes, trade_date.isoformat())
        finally:
            tconn.close()
        log_path = Path(args.etf_shadow_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def sink(rec: dict) -> None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"day": trade_date.isoformat(), **rec}, ensure_ascii=False) + "\n")

        print(f"# ETF그림자(81) — 문턱 {len({c for c, _ in thr})}/{len(codes)}종목, 셀 {len(CELLS)}",
              file=sys.stderr, flush=True)
        return StrengthShadow(thr, cells=CELLS, sink=sink) if thr else None
    except Exception as exc:  # noqa: BLE001 — 그림자 준비 실패로 매매를 죽이지 않는다
        print(f"[ETF그림자] 비활성 — 문턱 준비 실패: {exc}", file=sys.stderr, flush=True)
        return None


def _feed_etf_shadow(shadow: Any, ticks: list[dict], quotes: list[dict]) -> None:
    """프레임을 그림자에 흘린다. 어떤 예외도 삼킨다 — 실매매 프레임 처리가 먼저다."""
    try:
        for q in quotes:
            shadow.on_quote(str(q["code"]), q["ts"], q.get("bid1"), q.get("ask1"))
        for t in ticks:
            if t.get("price") is None:
                continue
            shadow.on_tick(str(t["code"]), t["ts"], float(t["price"]), strength=t.get("strength"),
                           best_bid=t.get("best_bid"), best_ask=t.get("best_ask"))
    except Exception as exc:  # noqa: BLE001
        print(f"[ETF그림자] 프레임 예외(무시): {exc}", file=sys.stderr, flush=True)


def _build_rl_shadow(args: argparse.Namespace, trade_date: date) -> tuple[Any, set[str]]:
    """RL 정책 실시간 추론 그림자(주문 없음) — 승격 체크리스트 9번 경로.

    정책 산출물(`<dir>/policy.json` + `ckpt.pt`)이 있고 피처 커널이 그 피처를 전부 줄 때만 돈다.
    못 주면 모듈이 스스로 추론을 끈다(0 으로 채워 돌리지 않는다).
    """
    d = str(getattr(args, "rl_shadow", "") or "")
    if not d:
        return None, set()
    try:
        import json

        import numpy as np

        from krx_quant_core.backtest import lob

        from .realtime.rl_shadow import PolicyWeights, RLShadow

        base = Path(d)
        meta = json.loads((base / "policy.json").read_text(encoding="utf-8"))
        load = (PolicyWeights.from_npz if (base / "weights.npz").exists()
                else PolicyWeights.from_checkpoint)          # 운용은 numpy, 연구 환경은 torch
        pol = load(base / ("weights.npz" if (base / "weights.npz").exists() else "ckpt.pt"),
                   features=meta["features"], mu=np.array(meta["mu"]), sd_=np.array(meta["sd"]),
                   lo=np.array(meta["lo"]), hi=np.array(meta["hi"]),
                   actions=tuple(meta["actions"]) if meta.get("actions") else None)
        kernel = str(getattr(args, "rl_kernel", "auto"))
        if kernel == "window" or (kernel == "auto" and not hasattr(lob, "SecondFeatureStreamV2")):
            # core 증분 커널(v2)이 아직 없으면 백테스트 함수를 창으로 다시 돌린다 — 값은 정의상 같고, 대신 무겁다.
            from .realtime.window_kernel import WindowKernelStream

            def stream_cls(code: str = "", _day=trade_date.isoformat()):
                return _WindowAdapter(WindowKernelStream(code or "?", day=_day))
        else:
            v2 = getattr(lob, "SecondFeatureStreamV2", None)
            if v2 is not None:
                def stream_cls(code: str = "", _cls=v2):
                    return _V2Adapter(_cls())
            else:
                stream_cls = lob.SecondFeatureStream
        log_path = Path(args.rl_shadow_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def sink(rec: dict) -> None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"day": trade_date.isoformat(), **rec}, ensure_ascii=False) + "\n")

        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        try:
            cur = conn.cursor()
            n = int(getattr(args, "rl_codes", 8) or 8)
            cur.execute("select code from daily_bars where date = (select max(date) from daily_bars) "
                        f"and trade_value > 0 order by trade_value desc limit {n}")
            codes = {r[0] for r in cur.fetchall()}
            if not codes:
                cur.execute("select code from stocks limit %s", (n,))
                codes = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
        from .realtime.rl_order_bridge import is_promoted, make_sink

        study = meta.get("study", base.name)
        runs = Path(__file__).resolve().parents[2] / "research" / "runs"
        live = bool(getattr(args, "rl_live", False))
        sink = make_sink(sink, executor=None, study=study, runs_dir=runs, live=live)
        mode = ("실주문" if live and is_promoted(study, runs) else
                "그림자(미승격)" if live else "그림자")
        print(f"# RL그림자({study}) — 피처 {len(meta['features'])}개 · 커널 {stream_cls.__name__} · {mode}",
              file=sys.stderr, flush=True)
        # 종목 간 순위(cum_rank)는 커널이 못 준다 — 그림자가 당일 누적 체결대금으로 만든다(백테스트와 같은 정의).
        shadow = RLShadow(pol, stream_factory=stream_cls, sink=sink, codes=codes,
                          extra_features="turnover_rank")
        # 창 커널은 종목당 1초 81ms 다 — 이벤트 루프에서 돌리면 주문이 밀린다. 별도 스레드로 뺀다.
        from .realtime.rl_shadow_worker import ShadowWorker

        # 큐는 버스트 흡수용이라 크게 잡는다 — 튜플 10만개는 수십 MB 수준이다.
        worker = ShadowWorker(
            shadow, maxsize=100_000,
            warn=lambda m: print(m, file=sys.stderr, flush=True), warn_every=2000)
        worker.start()
        return worker, codes
    except Exception as exc:  # noqa: BLE001 — 그림자 준비 실패로 매매를 죽이지 않는다
        print(f"[RL그림자] 비활성 — 준비 실패: {exc}", file=sys.stderr, flush=True)
        return None, set()


class _WindowAdapter:
    """창 재계산 커널을 RLShadow 의 스트림 규약(update(**집계) → dict)에 맞춘다."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.sec = -1

    def feed_tick(self, ts, price, volume, side, strength, best_bid, best_ask) -> None:
        self.inner.add_tick(ts, price, volume or 0, side or 0, strength, best_bid, best_ask)

    def feed_quote(self, ts, bid1, ask1, bq1, bq2, bq3, aq1, aq2, aq3) -> None:
        self.inner.add_quote(ts, bid1, ask1, bq1, bq2, bq3, aq1, aq2, aq3)

    def update(self, **_agg: float) -> dict[str, float]:
        return self.inner.update(self.sec) if self.sec >= 0 else {}


class _V2Adapter:
    """core 증분 커널(SecondFeatureStreamV2)을 RLShadow 의 스트림 규약에 맞춘다.

    그림자는 집계값(buy/sell/cnt)과 체결 배열을 함께 넘긴다. v2 는 체결 배열만 받으므로
    **받는 인자만 골라 넘긴다** — 이름을 손으로 나열하면 core 가 시그니처를 바꿀 때 조용히 깨진다.
    창 재계산 커널(초당 55~129ms)과 달리 증분이라 매매 프로세스 안에서 돌 수 있다.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.sec = -1
        self._ok = set(inspect.signature(inner.update).parameters)

    def update(self, **kw: Any) -> dict[str, float]:
        if "bidqty1" in kw and "bidqty1_raw" not in kw:
            kw["bidqty1_raw"] = kw["bidqty1"]          # core 규약: *_raw 는 그 초의 마지막 스냅샷
        if "askqty1" in kw and "askqty1_raw" not in kw:
            kw["askqty1_raw"] = kw["askqty1"]
        return self.inner.update(**{k: v for k, v in kw.items() if k in self._ok})


def _feed_rl_shadow(worker: Any, codes: set[str], ticks: list[dict], quotes: list[dict]) -> None:
    """큐에 넣기만 한다 — 계산은 그림자 스레드가 하고, 밀리면 버린다(매매가 먼저다)."""
    try:
        for q in quotes:
            if str(q["code"]) in codes:
                worker.feed_quote(str(q["code"]), q["ts"], bid1=q.get("bid1"), ask1=q.get("ask1"),
                                  bidqty1=q.get("bidqty1"), askqty1=q.get("askqty1"),
                                  bid_depth3=q.get("total_bid_qty"), ask_depth3=q.get("total_ask_qty"))
        for t in ticks:
            if t.get("price") is None or str(t["code"]) not in codes:
                continue
            worker.feed_tick(str(t["code"]), t["ts"], float(t["price"]), volume=t.get("volume"),
                             side=t.get("side"), strength=t.get("strength"),
                             best_bid=t.get("best_bid"), best_ask=t.get("best_ask"))
    except Exception as exc:  # noqa: BLE001
        print(f"[RL그림자] 프레임 예외(무시): {exc}", file=sys.stderr, flush=True)


def _build_tiger_shadow(args: argparse.Namespace, trade_date: date) -> tuple[Any, set[str]]:
    """86번 호랑이 규칙 그림자 — 주문 없음. 준비 실패면 (None, ∅) 로 감지·매매는 그대로 돈다."""
    if not getattr(args, "tiger_shadow", False):
        return None, set()
    try:
        import json

        from .realtime.tiger_shadow import TigerShadow

        prev = None
        pq = Path(args.tiger_prevq)
        if pq.exists():
            prev = float(json.loads(pq.read_text())["q99_dstr30"])
        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        try:
            cur = conn.cursor()
            cur.execute("select code from stocks")      # 주식만(ETF 제외) — 리플레이와 같은 유니버스
            codes = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
        log_path = Path(args.tiger_shadow_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def sink(rec: dict) -> None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"day": trade_date.isoformat(), **rec}, ensure_ascii=False) + "\n")

        print(f"# 호랑이그림자(86) — 전일 문턱 {prev}, 주식 {len(codes)}종목 대상", file=sys.stderr, flush=True)
        return TigerShadow(prev_quantile=prev, sink=sink), codes
    except Exception as exc:  # noqa: BLE001 — 그림자 준비 실패로 매매를 죽이지 않는다
        print(f"[호랑이그림자] 비활성 — 준비 실패: {exc}", file=sys.stderr, flush=True)
        return None, set()


def _feed_tiger_shadow(shadow: Any, codes: set[str], ticks: list[dict], quotes: list[dict]) -> None:
    """프레임을 호랑이 그림자에 흘린다. 어떤 예외도 삼킨다 — 실매매 프레임 처리가 먼저다."""
    try:
        for q in quotes:
            if str(q["code"]) in codes:
                shadow.on_quote(str(q["code"]), q["ts"], bid1=q.get("bid1"), ask1=q.get("ask1"),
                                bidqty1=q.get("bidqty1"))
        for t in ticks:
            if t.get("price") is None or str(t["code"]) not in codes:
                continue
            shadow.on_tick(str(t["code"]), t["ts"], float(t["price"]), volume=t.get("volume"), side=t.get("side"),
                           strength=t.get("strength"), best_bid=t.get("best_bid"), best_ask=t.get("best_ask"))
    except Exception as exc:  # noqa: BLE001
        print(f"[호랑이그림자] 프레임 예외(무시): {exc}", file=sys.stderr, flush=True)


def _duration_seconds(spec: str | None) -> float | None:
    """``HH:MM-HH:MM`` → 지금부터 종료시각까지 초. None 이면 무기한."""
    dl = _parse_deadline(spec)
    if dl is None:
        return None
    return max(0.0, (dl - datetime.now()).total_seconds())  # noqa: DTZ005


def _parse_deadline(spec: str | None) -> datetime | None:
    if not spec:
        return None
    try:
        _start, end = spec.split("-", 1)
        hh, mm = (int(x) for x in end.strip().split(":", 1))
    except ValueError as exc:
        raise SystemExit(f"--duration 은 HH:MM-HH:MM 형식이어야 한다: {spec!r}") from exc
    now = datetime.now()  # noqa: DTZ005
    return datetime.combine(now.date(), dtime(hh, mm))


# --------------------------------------------------------------------- 파서

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="짝꿍 실시간 감지·알림기 (등록 쌍만, 주문은 사람이)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--pairs", default=None,
                     help="'대장:후속,대장:후속' 인라인 쌍 (워치리스트보다 우선)")
    src.add_argument("--watchlist", default=DEFAULT_WATCHLIST_PATH,
                     help=f"쌍 JSON (기본 {DEFAULT_WATCHLIST_PATH})")
    # 모드: --live(소켓 직결, 실전 주경로) / --replay(검증) / 기본=폴링(백업)
    p.add_argument("--live", action="store_true",
                   help="소켓 직결 모드(실전 주경로): 수집+감지 한 프로세스, 지연 ~0")
    p.add_argument("--replay", default=None, metavar="YYYY-MM-DD",
                   help="리플레이 검증 모드: 그날 틱을 시간순으로 재현(dry-run)")
    p.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    p.add_argument("--duration", default=None, help="실시간/소켓: HH:MM-HH:MM 종료시각까지")
    p.add_argument("--lookback", type=float, default=120.0,
                   help="폴링 워밍업으로 먼저 먹일 최근 초 (기본 120)")
    p.add_argument("--poll-interval", type=float, default=1.5, help="폴링 주기 초 (기본 1.5)")
    p.add_argument("--mock", action="store_true", help="소켓: 모의투자 WS 엔드포인트")
    p.add_argument("--no-store", action="store_true",
                   help="소켓: 이 프로세스에서 ticks·quotes 저장 안 함(감지만)")
    p.add_argument("--flush-interval", type=float, default=1.5,
                   help="소켓: 저장 배치 간격 초 (기본 1.5)")
    p.add_argument("--spool-dir", default="data/spool",
                   help="DB 순단 중 프로세스가 죽어도 버퍼를 잃지 않게 미러링할 "
                        "디렉터리(기본 data/spool). 빈 문자열이면 끔")
    p.add_argument("--log", default="data/pair_alerts.log", help="알림 로그 파일")
    p.add_argument("--dry-run", action="store_true", help="로그·웹훅 안 쓰고 stdout 미리보기만")
    p.add_argument("--bell", action="store_true", help="알림 시 터미널 벨")
    p.add_argument("--no-discord", action="store_true", help="Discord 웹훅 훅 끄기")
    # 청산 모드
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stable", action="store_true", help="안정형(기본): +0.5%%/−0.4%%/90s")
    mode.add_argument("--aggressive", action="store_true",
                      help="공격형: +0.8%%/일상손절없음+재난선−0.7%%/90s")
    p.add_argument("--take-profit", type=float, default=None,
                   help="익절 비율 덮기 (예 0.008, 0 = 상한 없음 — 수익을 열어둔다)")
    p.add_argument("--stop-loss", type=float, default=None,
                   help="일상 손절 비율 덮기 (예 -0.004, 0 = 없음)")
    p.add_argument("--disaster-line", type=float, default=None,
                   help="재난선 비율 덮기 (예 -0.007, 0 = 없음)")
    p.add_argument("--hold", type=float, default=None, help="시간청산 초 덮기 (기본 90)")
    p.add_argument("--surge", type=float, default=None, help="대장 서지 임계 덮기 (기본 0.015)")
    p.add_argument("--cosurge", type=float, default=None,
                   help="후속 동반급등 하한 (기본 0.005). **0 = 요구 안 함** — 만쥬는 "
                        "동반 급등을 진입 조건으로 말한 적이 없다. 반응 속도는 짝꿍 "
                        "'선정' 기준이지(--follower-priority manju) 게이트가 아니다")
    p.add_argument("--premove", type=float, default=None,
                   help="후속 프리무브 상한 (기본 0.002 = +0.2%%). 만쥬 \"이미 2~4%% "
                        "튀어 오른 뒤라 추격 매수 시 100%% 물립니다\"")
    p.add_argument("--cooldown", type=float, default=None, help="쌍 재알림 억제 초 (기본 90)")
    p.add_argument("--halt-below-jo", type=float, default=0.0,
                   help="코스닥 전일 거래대금이 이 조원 미만이면 신규 진입 금지 "
                        "(만쥬: 7~8조 이하 침체장에선 신호가 떠도 안 산다). 0 = 끔")
    p.add_argument("--min-turnover", type=float, default=None,
                   help="후속 당일 누적 체결대금 하한(원). 0 = 끔")
    p.add_argument("--no-universe", action="store_true",
                   help="만쥬 동적 모드 끄기 — 그날 유니버스 대신 고정 워치리스트 사용")
    p.add_argument("--universe-size", type=int, default=45,
                   help="그날 유니버스 종목 수. 단일 앱키 한도가 50이라 기본 45")
    p.add_argument("--entry-ticks-up", type=int, default=0,
                   help="진입 지정가를 현재가 위 몇 호가에 낼지 (만쥬: 2~3). "
                        "0 = 매수호가에 걸고 기다린다")
    p.add_argument("--limitup-touch", action="store_true",
                   help="상한가 터치를 방아쇠로 (만쥬 §5-1)")
    p.add_argument("--follower-priority", choices=["rank", "manju"], default="rank",
                   help="등락률 2~3위가 여럿일 때 고르는 순서. manju = 호가두께 → "
                        "틱 체결강도 → 가벼운 소형주(만쥬가 준 서열)")
    p.add_argument("--limitup-near-ticks", type=int, default=0,
                   help="상한가까지 남은 호가가 이 개수 이하면 발동 (만쥬: 3). "
                        "켜면 --limitup-near(비율)보다 우선한다. 0 = 끔")
    p.add_argument("--limitup-near", type=float, default=None,
                   help="상한가까지 이 비율 이내면 발동 (예 0.04 = 4%%p). 0 = 끔")
    p.add_argument("--lock-min-krw", type=float, default=None,
                   help="상한가 매수벽 금액 하한(원) — 만쥬 §5-2 \"수십만~수백만 주를 "
                        "안정적으로 굳히면\". 0 = 끔")
    p.add_argument("--lock-wall-decay", type=float, default=None,
                   help="매수벽이 최근 최대 대비 이 비율 아래로 깎이면 이탈로 본다 "
                        "(0.5 = 반토막). 0 = 끔")
    p.add_argument("--dump-krw", type=float, default=None,
                   help="대장 매도 체결대금이 관측창에서 이 금액을 넘으면 이탈 — "
                        "만쥬 §5-4 \"프로그램 매도 폭탄에 막히면 즉시 손절\". 0 = 끔")
    p.add_argument("--collect-fill-interval", type=float, default=0.0,
                   help="감지기가 안 쓰는 등록 자리를 수집 전용으로 채우는 주기(초). "
                        "0 = 끔. 붙인 종목은 저장만 되고 신호·주문은 안 난다")
    p.add_argument("--collect-reserve", type=int, default=12,
                   help="감지기가 장중에 대장을 붙일 예비 자리 — 수집이 안 건드린다")
    p.add_argument("--collect-fill-ttl", type=float, default=600.0,
                   help="수집 종목이 등락률 순위에서 사라진 뒤 자리를 내주기까지(초). "
                        "짧으면 등록만 낭비하고 길면 오후 대장이 붙을 자리가 없다")
    p.add_argument("--collect-pin", default="",
                   help="쉼표 구분 — 수집 채움이 항상 먼저 붙이고 안 내리는 종목(저장만). "
                        "80번: 매도세 면제 ETF 틱 수집")
    p.add_argument("--collect-pin-file", default="",
                   help="한 줄 또는 쉼표 구분 종목 파일 — --collect-pin 에 더해 고정 수집한다(저장만). "
                        "다른 전략이 전일 마감 뒤 떨구는 목록용. 없으면 조용히 건너뛴다")
    p.add_argument("--etf-shadow", default="",
                   help="쉼표 구분 ETF — 81번 체결강도 규칙을 소켓에서 그림자로 돌린다(주문 없음)")
    p.add_argument("--etf-shadow-log", default="data/etf81_shadow.jsonl")
    p.add_argument("--tiger-shadow", action="store_true",
                   help="86번 호랑이 규칙(알고리즘 체결·체결강도 가속·지정가 대기열) 실시간 그림자 — 주문 없음")
    p.add_argument("--tiger-shadow-log", default="data/tiger86_shadow.jsonl")
    p.add_argument("--rl-shadow", default="",
                   help="RL 정책 산출물 디렉터리(policy.json+ckpt.pt) — 실시간 추론 그림자, 주문 없음")
    p.add_argument("--rl-shadow-log", default="data/rl_shadow.jsonl")
    p.add_argument("--rl-kernel", choices=["auto", "core", "window"], default="auto",
                   help="피처 커널 — auto: core v2 가 있으면 core, 없으면 창 재계산(백테스트 함수 그대로)")
    p.add_argument("--rl-codes", type=int, default=8,
                   help="RL 그림자 대상 종목 수 — 창 커널은 종목당 1초 81ms 라 적게 잡는다")
    p.add_argument("--rl-live", action="store_true",
                   help="RL 정책 실주문 — research/runs/<study>/PROMOTED(승격 점검 통과)가 있어야 열린다")
    p.add_argument("--tiger-prevq", default="data/tiger86_prevq.json",
                   help="09:35 전 문턱(전일 dstr30 99%% 분위) — simnode 17:00 이 만들어 보낸다")
    p.add_argument("--collect-fill-pool", type=int, default=300,
                   help="수집 후보 풀 크기(전일 거래대금 상위 N종목)")
    p.add_argument("--thin-book-bp", type=float, default=0.0,
                   help="왕복 충격이 이 bp 를 넘으면 호가가 얇다고 보고 비중을 줄인다. "
                        "만쥬 A-2 Thin Orderbook Restriction(51번). 0 = 끔")
    p.add_argument("--thin-book-size", type=float, default=0.2,
                   help="얇을 때 곱할 비중 배수(지시서 10~20%%)")
    p.add_argument("--scratch-ticks", type=int, default=0,
                   help="진입 후 후속주가 진입가 위로 이만큼(호가) 못 뜨면 본절 탈출. "
                        "타 트레이더 B B-2 Immediate Acceleration Rule(51번). 0 = 끔")
    p.add_argument("--scratch-secs", type=float, default=0.0,
                   help="본절 탈출 판정까지 기다리는 시간(초). 원문에 없는 우리 값")
    p.add_argument("--profit-giveback", type=float, default=0.0,
                   help="정점 이익 대비 이 비율 이하로 반납하면 그날 하드스톱(0.5 = 절반). "
                        "타 트레이더 B B-3 '50%% Rule'(51번). 0 = 끔")
    p.add_argument("--profit-giveback-min", type=float, default=0.0,
                   help="위 규칙을 적용하기 시작하는 최소 정점 이익(원). "
                        "'significant P_peak' 단서 — 작은 정점으로 하루를 끝내지 않는다")
    p.add_argument("--scout-gate-jo", type=float, default=0.0,
                   help="코스닥 거래대금이 이 조원 미만이면 정찰병 비중(1/10)으로 매매한다. "
                        "만쥬 A1 의 Gate 0(7~8조). 0 = 끔")
    p.add_argument("--no-entry-after", default="",
                   help="이 시각(HHMM) 뒤로는 신규 진입을 막는다. 청산은 계속 돈다. "
                        "만쥬 49 §1-5 '오후 추격 매매는 철저히 배제'. 빈 값 = 끔")
    p.add_argument("--lock-wall-window", type=float, default=None,
                   help="상한가 매수벽 감소를 재는 창(초). 만쥬 원문은 '직전 3초 피크 대비'")
    p.add_argument("--leader-stall", type=float, default=None,
                   help="대장이 이 초 동안 신고가를 못 내면 이탈로 본다(만쥬 2026-08-29: "
                        "'더 이상 올라가지 못하는 순간'). 0 = 끔. 원문에 수치 없음 — 임의값")
    p.add_argument("--leader-drop-ticks", type=int, default=None,
                   help="대장이 당일 고점 대비 이 호가 아래로 밀리면 이탈로 본다"
                        "(만쥬: '하락하는 순간'). 0 = 끔. 원문에 수치 없음 — 임의값")
    p.add_argument("--big-trades", type=int, default=None,
                   help="대장에 필요한 대량체결(기본 5천만원↑) 최소 건수 — 만쥬 §5-1 "
                        "순간 체결량 확인. 0 = 끔")
    p.add_argument("--program-interval", type=float, default=0.0,
                   help="장중 프로그램 수급 감시 주기(초) — 만쥬 \"쏟아내는 장세면 "
                        "진입을 거릅니다\". 0 = 끔")
    p.add_argument("--program-dump-qty", type=float, default=700.0,
                   help="이만큼(천주) 이상 순매도면 신규 진입 차단. 만쥬 (비공개 회신 — 공개본에서 뺐다) = 700~800")
    p.add_argument("--program-market", default="P101",
                   help="P001 코스피 · P101 코스닥(기본)")
    p.add_argument("--no-themes", action="store_true",
                   help="실전 테마(키움 theme_members) 주입 끄기 — 27종 섹터만 쓴다")
    p.add_argument("--leader-scan-interval", type=float, default=0.0,
                   help="장중 시장 전체 대장 스캔 주기(초) — 만쥬 §2. ka10027 등락률 "
                        "상위를 훑어 대장·짝꿍을 구독에 추가한다. 0 = 끔")
    p.add_argument("--leader-scan-min-rt", type=float, default=20.0,
                   help="대장 후보 최소 등락률(%%). 만쥬 \"20%% 이상으로 수직 상승\"")
    p.add_argument("--leader-scan-min-wall", type=float, default=0.0,
                   help="상한가에 잠긴 종목에 요구할 최소 매수 잔량(주). 0 = 끔")
    p.add_argument("--leader-scan-max", type=int, default=6,
                   help="한 번에 붙일 대장 후보 수. 짝꿍까지 붙으니 자리를 보고 정한다")
    p.add_argument("--leader-scan-window", default="",
                   help="새 대장을 받아들이는 시간 창 (예 0900-0930) — 만쥬는 오전에 "
                        "그날 대장을 낙점한다. 이미 붙은 종목은 종일 그대로 돈다. "
                        "빈 값 = 종일 열림")
    p.add_argument("--leader-scan-market", default="101",
                   help="000 전체 · 001 코스피 · 101 코스닥(기본)")
    p.add_argument("--leader-scan-ttl", type=float, default=900.0,
                   help="스캔으로 붙인 종목을 이 초 동안 상위에서 못 보면 자리를 비운다")
    p.add_argument("--breaking-interval", type=float, default=0.0,
                   help="장중 돌발 재료 감시 주기(초) — 만쥬 §6. 새 재료 공시 종목을 "
                        "구독에 추가한다. 0 = 끔. 켤 거면 60 이상 권장")
    p.add_argument("--breaking-min-weight", type=float, default=0.7,
                   help="돌발 재료로 볼 최소 재료 가중치. 0.7 = 공급계약·기술인허가·시설투자")
    p.add_argument("--breaking-max", type=int, default=5,
                   help="한 번에 추가할 최대 종목 수(구독 자리 보호)")
    p.add_argument("--dart-db", default="data/dart.db",
                   help="재료 판정에 쓸 DART 공시 sqlite. 없으면 재료항 없이 돈다")
    p.add_argument("--material-days", type=int, default=5,
                   help="최근 N일 공시를 재료로 본다(만쥬 §2-1). 0 = 끔")
    p.add_argument("--material-llm", action="store_true",
                   help="재료를 LLM 으로 채점한다(만쥬 §2-1 \"미래 가치·지속시킬 스토리\"). "
                        "유형 가중치표보다 판단이 세다. 키가 없으면 조용히 규칙 기반으로 "
                        "폴백한다. SWING_LLM_PROVIDER·API 키 필요")
    p.add_argument("--material-llm-max", type=int, default=300,
                   help="한 번에 LLM 으로 채점할 공시 최대 건수(기동 지연 방지). 0 = 무제한")
    p.add_argument("--llm-cache", default="data/llm_material_cache.db",
                   help="LLM 채점 캐시(같은 공시를 두 번 부르지 않는다)")
    p.add_argument("--material-weight", type=float, default=0.5,
                   help="유니버스 점수의 재료 가중치. 0 = 재료 무시")
    p.add_argument("--new-listing-days", type=int, default=5,
                   help="최근 N일 내 상장 종목을 유니버스에 무조건 편입(만쥬 §4-5). 0 = 끔")
    p.add_argument("--stale-sector-days", type=int, default=10,
                   help="테마 신선도를 재는 기간(영업일). 기본 10")
    p.add_argument("--stale-sector-limitups", type=int, default=0,
                   help="이 횟수를 넘게 상한가가 나온 섹터는 '소비된 테마'로 보고 배제"
                        "(만쥬 §5-5). 0 = 끔 — 문턱이 낮으면 유니버스가 거덜난다")
    p.add_argument("--vi-near", type=float, default=None,
                   help="정적 VI 상단(당일 기준가 +10%%)까지 이 비율 이내면 발동 — "
                        "만쥬 §5-1 \"VI 에 들어가는 순간\". VI 는 +10%%라 상한가(+30%%)"
                        "보다 훨씬 이르다. 실시간 VI 피드 없이 가격으로 계산한다. 0 = 끔")
    p.add_argument("--limitup-require-surge", action="store_true",
                   help="상한가 방아쇠에 서지 조건까지 AND — 상한가 근처에서 정체된 "
                        "종목에 발동하는 걸 막는다(원본 presurge 는 +25%% 교차 AND "
                        "그 분 +1%% 모멘텀이었다)")
    p.add_argument("--limitup-lock", type=float, default=None,
                   help="상한가를 이 초만큼 연속 유지하면 발동(잠김). 0 = 끔")
    p.add_argument("--max-roundtrip-bp", type=float, default=None,
                   help="10호가 왕복 충격 상한(bp) — 만쥬 최우선 필터. 0 = 끔")
    p.add_argument("--order-krw", type=float, default=None,
                   help="왕복 충격을 재는 주문금액(원). 기본 1000만")
    p.add_argument("--accel", type=float, default=None,
                   help="대장 가격 가속 하한(배) — 서지창 뒤 절반 수익률 / 앞 절반. "
                        "기본 0(끔). 예 1.5 = 뒤가 앞보다 1.5배 가파른 상승만")
    p.add_argument("--volume-burst", type=float, default=None,
                   help="대장 거래량 폭발 하한(배) — 최근 10초 체결량률 / 직전 30초. "
                        "기본 0(끔). 프로그램매매는 가격보다 체결량에 먼저 찍힌다")
    p.add_argument("--min-sl-ticks", type=float, default=None,
                   help="손절폭 최소 호가단위 수(저가주 노이즈 청산 방어). "
                        "기본 3.0, 0 = 끔")
    p.add_argument("--min-buy-ratio", type=float, default=None,
                   help="후속 직전30초 매수체결비율 하한(역선택 방어, doc39). "
                        "기본 0.30, 0 = 끔")
    # 리스크 가드 (감사 선결조건: 일일손실한도·킬스위치·포지션추적)
    g = p.add_argument_group("리스크 가드")
    g.add_argument("--no-guard", action="store_true",
                   help="가드 끄기(기존 동작 — 손실통제 없이 알림만)")
    g.add_argument("--daily-loss-limit", default="50000",
                   help="일일 손실한도. '50000'(원) 또는 '2%%'(자본 대비). 기본 50000원")
    g.add_argument("--capital", type=float, default=1_000_000.0,
                   help="%% 손실한도·기록 기준 자본(원). 기본 100만")
    g.add_argument("--position-notional", type=float, default=100_000.0,
                   help="가상 포지션 1건 명목(원) — 수익률→원 환산. 기본 10만")
    g.add_argument("--stop-needs-leader-weak", action="store_true",
                   help="대장이 상한가를 굳히고 있는 동안은 가격 손절을 잠근다 — 만쥬 "
                        "\"2등주 가격 지표로 손절하지 말고 대장 이탈을 트리거로\". "
                        "재난선은 그대로 산다")
    g.add_argument("--shadow-when-blocked", action="store_true",
                   help="국면 차단(거래대금 미달)으로 안 산 신호도 그림자로 굴려 "
                        "\"샀으면 어땠나\"를 기록한다. 실현손익·킬스위치·주문엔 "
                        "안 닿는다 — 만쥬 규칙대로 안 사는 건 그대로다")
    g.add_argument("--drawdown-krw", type=float, default=0.0,
                   help="실현손익이 그날 정점 대비 이만큼(원) 밀리면 비중 축소 — 만쥬 "
                        "\"과거 정점 대비 계좌가 꺾일 때 예수금을 1/3~1/5로\". 0 = 끔")
    g.add_argument("--drawdown-size", type=float, default=1.0 / 3,
                   help="낙폭 방어 발동 시 명목 배수(만쥬: 1/3~1/5). 기본 1/3")
    g.add_argument("--trail-arm", type=float, default=0.0,
                   help="이만큼 오르면 트레일링 무장(만쥬: 0.03 = +3%%). 0 = 끔")
    g.add_argument("--trail", type=float, default=0.0,
                   help="무장 후 고점 대비 이만큼 밀리면 청산. 0 = 끔")
    g.add_argument("--max-consecutive-losses", type=int, default=3,
                   help="연속 손절 상한(킬 트리거). 기본 3")
    g.add_argument("--max-positions", type=int, default=2,
                   help="동시 보유 가상 포지션 상한. 기본 2")
    g.add_argument("--fill-basis", choices=("through", "touch"), default="through",
                   help="가상 체결 회계 가정: through(보수·기본)|touch(낙관)")
    g.add_argument("--cost-rate", type=float, default=0.0064,
                   help="왕복 비용률(수익률에서 차감). 기본 0.0064 = 왕복 34bp + "
                        "슬리피지 왕복 30bp(레포 기본 가정, scripts/verify_pair.py). "
                        "0 = 끔(세전)")
    g.add_argument("--partial-exit", type=float, default=0.0,
                   help="1차 분할 매도 비율(0<r<1) — 만쥬 §5-2 \"분할 또는 전량 매도\". "
                        "0 = 끔. **수량이 2주 이상일 때만 실효**(1주는 쪼갤 수 없다)")
    g.add_argument("--partial-exit-gain", type=float, default=None,
                   help="이 수익률에 닿으면 1차 분할(예 0.002 = +0.2%%)")
    g.add_argument("--partial-exit-after", type=float, default=None,
                   help="체결 후 이 초가 지나면 1차 분할(예 60)")
    g.add_argument("--pyramid-arm", type=float, action="append", default=None,
                   help="불타기 발동 수익률(entry 대비 누적, 만쥬 A-4: 0.03~0.04). "
                        "여러 번 줘서 단계를 늘릴 수 있다(예: --pyramid-arm 0.03 "
                        "--pyramid-arm 0.06 = 2단계). 안 주면 꺼짐(기존 동작). "
                        "물타기는 arm 이 항상 양수라 구조적으로 안 걸린다")
    g.add_argument("--pyramid-add-qty", type=int, default=1,
                   help="불타기 1단계당 추가매수 수량. 기본 1주")
    g.add_argument("--daily-profit-target", type=float, default=0.0,
                   help="일일 이익 목표(원). 도달하면 이후 진입 비중을 줄인다 — 만쥬 "
                        "§9-6 \"오전 수익 뒤 오후에 무리하게 베팅하는 패턴을 가장 경계\". "
                        "멈추는 게 아니라 줄인다. 0 = 끔")
    g.add_argument("--profit-size-multiplier", type=float, default=0.3,
                   help="이익 목표 도달 후 곱할 비중 배수. 기본 0.3")
    g.add_argument("--size-multiplier", type=float, default=1.0,
                   help="포지션 명목에 곱하는 국면 배수(만쥬 §8-1). 활황 1.0 / 중간 0.6 "
                        "/ 침체 0.3 / 극침체 0. --size-by-regime 로 자동 결정 가능")
    g.add_argument("--size-by-regime", action="store_true",
                   help="국면 배수를 코스닥 거래대금에서 자동 결정(만쥬 §8-1). "
                        "극침체면 배수 0 = 신규 진입 없음")
    g.add_argument("--no-leader-exit", action="store_true",
                   help="대장 조건부 청산 끄기 — 만쥬 §5-3 의 핵심 축이라 기본은 켬. "
                        "대장이 상한가를 놓치면 손익 무관 즉시 청산한다")
    g.add_argument("--stop-file", default="data/pair_STOP",
                   help="존재 시 수동 킬스위치. 기본 data/pair_STOP")
    g.add_argument("--no-stop-file", action="store_true",
                   help="수동 정지 파일 감시 끄기")
    g.add_argument("--position-log", default="data/pair_positions.log",
                   help="가상 포지션 로그(MAE·지연·체결·손익). 기본 data/pair_positions.log")
    # 실계좌 주문 실행 (이중 게이트: --live-order 만으로는 dry-run)
    o = p.add_argument_group("실계좌 주문 (기본 꺼짐 · 이중 안전)")
    o.add_argument("--live-order", action="store_true",
                   help="주문 실행기 켜기. 단 이것만으론 dry-run(발주 안 함). 기본 꺼짐")
    o.add_argument("--confirm-real", action="store_true",
                   help="진짜 발주. --live-order 와 **둘 다** 있어야 실주문(이중 확인)")
    o.add_argument("--qty", type=int, default=1,
                   help="주문 수량(주). 기본 1, 어댑터 하드캡 10")
    o.add_argument("--order-band-pct", type=float, default=0.03,
                   help="지정가 허용 범위(참조가 대비 ±). 기본 0.03(±3%%)")
    o.add_argument("--max-orders-per-code", type=int, default=5,
                   help="종목당 일일 주문 횟수 상한. 기본 5")
    o.add_argument("--max-orders-total", type=int, default=20,
                   help="일일 총 주문 횟수 상한. 기본 20")
    o.add_argument("--unfilled-timeout", type=float, default=15.0,
                   help="미체결 매수 취소까지 초. 기본 15")
    o.add_argument("--order-log", default="data/pair_orders.log",
                   help="주문 로그(종목·수량·가격·매수매도, 비밀 없음). 기본 data/pair_orders.log")
    return p


#: ``--live`` 단일 인스턴스 잠금(flock). 런처가 레포 루트로 cd 한 뒤 띄우므로 상대경로다.
PAIR_DETECT_LOCK = Path("data/pair_detect.lock")
#: 잠금을 못 잡았을 때 끝나기 전에 기다리는 초. 런처(pair_detect_daily.sh)는 종료코드와
#: 무관하게 5초 뒤 다시 띄우므로, 바로 끝내면 몇 초마다 uv 기동을 반복하는 헛바퀴가 된다.
#: 기다렸다 0 으로 끝나면 앞 프로세스가 살아 있는 동안은 1분에 한 번만 확인하고, 앞
#: 프로세스가 죽으면 다음 회차에 이어받는다(대기 중 잠금이 풀려도 매매 공백은 최대 ~1분).
ALREADY_RUNNING_WAIT_S = 60.0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.live:
        return _main(args)
    # 같은 계좌로 감지기 둘이 돌면 같은 신호에 주문이 두 번 나간다. 감지기·실행기(인증·
    # 계좌조회)를 만들기 **전에** 잠근다. 프로세스가 죽으면 커널이 flock 을 풀어 준다.
    try:
        from krx_quant_core.execution.oms import AlreadyRunning, InstanceLock
    except ImportError as exc:  # 코어 0.5.0 미만 — 잠금 때문에 장중 기동을 죽이지 않는다
        print(f"[경고] krx-quant-core 에 InstanceLock 이 없다({exc}) — 잠금 없이 진행한다. "
              f"'uv sync --extra pg' 로 코어를 0.5.x 로 올릴 것", file=sys.stderr, flush=True)
        return _main(args)
    lock = InstanceLock(PAIR_DETECT_LOCK)
    try:
        lock.acquire()
    except AlreadyRunning:
        print(f"[잠금] 짝꿍 감지기가 이미 실행 중이다({PAIR_DETECT_LOCK}) — 중복 주문을 막으려고 "
              f"뜨지 않는다. {ALREADY_RUNNING_WAIT_S:.0f}초 기다린 뒤 종료(0)한다.",
              file=sys.stderr, flush=True)
        time.sleep(ALREADY_RUNNING_WAIT_S)
        return 0
    except Exception as exc:  # noqa: BLE001 — 잠금 파일 문제(권한·경로)로 매 재기동마다 죽지 않는다
        print(f"[경고][잠금] 잠금 파일을 잡지 못했다({PAIR_DETECT_LOCK}: {exc!r}) — **잠금 없이** "
              f"진행한다. 중복 실행 보호가 꺼졌다: data/ 권한·경로를 확인할 것",
              file=sys.stderr, flush=True)
        return _main(args)
    try:
        return _main(args)
    finally:
        lock.release()


def _main(args: argparse.Namespace) -> int:
    pairs = resolve_pairs(args)
    uni = build_universe_inputs(args)
    # 만쥬는 고정 쌍을 안 쓴다 — 그날 대장이 상한가로 가면 **그 자리에서** 같은 테마
    # 2·3등주를 짝으로 삼는다. 그래서 워치리스트가 비어도 동적 모드면 정상이다.
    # 동적 모드 성립 조건은 감지기와 같다: 유니버스와 섹터가 **둘 다** 있어야 한다.
    if not pairs and not (uni.get("universe") and uni.get("sectors")):
        raise SystemExit("감시할 쌍이 없다. --pairs·워치리스트 또는 유니버스를 확인하라.")
    detector = PairDetector(pairs, build_config(args), **uni)
    guard = build_guard(args)
    # 주문 실행기 — 리플레이(과거 재현)에선 절대 만들지 않는다.
    executor = (None if args.replay
                else build_order_executor(args, pairs,
                                          universe=uni.get("universe") or ()))

    # 소켓 직결 모드 — DB 폴링을 안 거친다(저장은 collector 가 담당).
    if args.live:
        return run_live(args, detector, guard, executor)

    conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
    # 후속주 구독 커버리지 경고 (오늘 ticks 에 후속주가 있는지).
    if args.replay is None:
        _warn_missing(conn, detector)

    if args.replay:
        return run_replay(args, conn, detector, guard)
    return run_realtime(args, conn, detector, guard, executor)


def _warn_missing(conn: Any, detector: PairDetector) -> None:
    """오늘 ticks 에 후속주가 안 보이면 경고(구독 안 됨 → 감시 불가)."""
    ph = _placeholder(conn)
    codes = sorted(detector.codes)
    in_clause = ",".join([ph] * len(codes))
    today = date.today().isoformat()  # noqa: DTZ011
    cur = conn.cursor()
    cur.execute(
        f"select distinct code from ticks where code in ({in_clause}) and ts >= {ph}",
        [*codes, f"{today} 00:00:00"],
    )
    have = {str(r[0]) for r in cur.fetchall()}
    cur.close()
    for pair in detector.missing_followers(have):
        fol = pair.follower_name or pair.follower
        lead = pair.leader_name or pair.leader
        print(f"[경고] 후속주 {fol}({pair.follower}) 가 오늘 수집 종목에 없음 → "
              f"{lead}→{fol} 쌍 감시 불가. --auto 유니버스에 두 종목 다 들어와야 한다.",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
