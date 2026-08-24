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
import os
import signal
import sys
import time
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

from .intraday.store import _placeholder, connect, is_postgres
from .realtime.alerts import AlertBus, build_default_bus
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
        over["take_profit"] = args.take_profit
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
        max_positions=args.max_positions,
        fill_basis=args.fill_basis,
        cost_rate=args.cost_rate,
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


def _emit_guarded(alert: Alert, bus: AlertBus, guard: RiskGuard | None,
                  poslog: PositionLogSink | None) -> bool:
    """알림을 가드에 태워 통과분만 emit. 억제되면 배너·킬 로그를 남긴다. emit 여부."""
    if guard is None:
        bus.emit(alert)
        return True
    decision = guard.on_alert(alert)
    if decision.banner:
        bus.notice(decision.banner)
        if poslog is not None:
            poslog.log_kill(alert.ts, guard.kill_reason, guard.realized_krw)
    if not decision.allowed:
        return False
    bus.emit(alert)
    if poslog is not None:
        poslog.log_open(alert)
    return True


def _feed(detector: PairDetector, bus: AlertBus, rows: list[tuple],
          guard: RiskGuard | None = None,
          poslog: PositionLogSink | None = None) -> int:
    n = 0
    for code, ts, price, volume, best_bid, side in rows:
        if price is None:
            continue
        t = _as_dt(ts)
        p = float(price)
        # 가드는 매 틱으로 열린 가상 포지션의 체결·MAE·청산을 리허설한다.
        if guard is not None:
            guard.on_tick(str(code), t, p)
        for alert in detector.on_tick(str(code), t, p, bid1=best_bid,
                                       volume=volume, side=side):
            if _emit_guarded(alert, bus, guard, poslog):
                n += 1
    return n


def build_poslog(args: argparse.Namespace, guard: RiskGuard | None) -> PositionLogSink | None:
    """포지션 로그 sink 를 만들고 가드 청산 콜백에 연결한다."""
    if guard is None or args.dry_run:
        return None
    poslog = PositionLogSink(args.position_log, basis=args.fill_basis)
    guard.set_close_callback(poslog.log_close)
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
                 guard: RiskGuard | None) -> int:
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
            rows = _fetch_ticks(conn, codes, since=watermark, day=None)
            if rows:
                _feed(detector, bus, rows, guard=guard, poslog=poslog)
                watermark = max(_as_dt(r[1]) for r in rows)
            if deadline is not None and datetime.now() >= deadline:  # noqa: DTZ005
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n중단.", file=sys.stderr)
    if guard is not None:
        guard.flush_expired(datetime.now())  # noqa: DTZ005
        print(guard.status_line(), flush=True)
    print(f"누적 알림 {bus.count}건.", flush=True)
    return 0


def run_live(args: argparse.Namespace, detector: PairDetector,
             guard: RiskGuard | None) -> int:
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
    """
    from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
    from .realtime.collector import (
        TYPE_QUOTE,
        TYPE_TRADE,
        WS_URL,
        WS_URL_MOCK,
        KiwoomRealtimeCollector,
        SeqAssigner,
        parse_real_message,
    )
    from .realtime.store import init_realtime_tables

    codes = sorted(detector.codes)
    url = WS_URL_MOCK if args.mock else WS_URL
    types = [TYPE_TRADE, TYPE_QUOTE]
    store = not args.no_store and not args.dry_run
    bus = build_default_bus(log_path=args.log, bell=args.bell,
                            discord=not args.no_discord, dry_run=args.dry_run)
    poslog = build_poslog(args, guard)

    conn = None
    if store:
        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        init_realtime_tables(conn)

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
    token = KiwoomClient(appkey, secret).token

    trade_date = date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
    seq = SeqAssigner()

    def on_frame(msg: dict) -> None:
        now = datetime.now()  # noqa: DTZ005 — KST naive
        ticks, quotes, _vis = parse_real_message(msg, trade_date, seq, received_at=now)
        for q in quotes:                      # 호가로 bid1 우선 갱신
            detector.on_quote(str(q["code"]), q["ts"], q.get("bid1"))
        for t in ticks:                       # 체결로 서지·매칭 판정
            if t.get("price") is None:
                continue
            code_s, ts_t, px_t = str(t["code"]), t["ts"], float(t["price"])
            if guard is not None:             # 가드: 체결·MAE·청산 리허설
                guard.on_tick(code_s, ts_t, px_t)
            for alert in detector.on_tick(code_s, ts_t, px_t,
                                          bid1=t.get("best_bid"), volume=t.get("volume"),
                                          side=t.get("side")):
                _emit_guarded(alert, bus, guard, poslog)

    collector = KiwoomRealtimeCollector(
        token, codes, conn=conn, dsn=(args.dsn if store else None),
        types=types, url=url, store=store, trade_date=trade_date,
        on_frame=on_frame, flush_interval=args.flush_interval,
    )
    duration = _duration_seconds(args.duration)
    print(f"# 소켓 직결 감지 시작 — 쌍 {len(detector.pairs)}개, 종목 {len(codes)}개, "
          f"저장 {'켬' if store else '끔'}, 로그 {args.log}"
          + (" [dry-run]" if args.dry_run else ""), file=sys.stderr, flush=True)

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, collector.stop)
            except (NotImplementedError, ValueError):
                pass
        await collector.run(duration=duration)

    try:
        asyncio.run(_main())
    finally:
        if conn is not None:
            conn.close()
    print(f"누적 알림 {bus.count}건 · 프레임 {collector.frames_seen:,}"
          f" · 저장 틱 {collector.ticks_stored:,}", file=sys.stderr, flush=True)
    return 0


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
    p.add_argument("--log", default="data/pair_alerts.log", help="알림 로그 파일")
    p.add_argument("--dry-run", action="store_true", help="로그·웹훅 안 쓰고 stdout 미리보기만")
    p.add_argument("--bell", action="store_true", help="알림 시 터미널 벨")
    p.add_argument("--no-discord", action="store_true", help="Discord 웹훅 훅 끄기")
    # 청산 모드
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stable", action="store_true", help="안정형(기본): +0.5%%/−0.4%%/90s")
    mode.add_argument("--aggressive", action="store_true",
                      help="공격형: +0.8%%/일상손절없음+재난선−0.7%%/90s")
    p.add_argument("--take-profit", type=float, default=None, help="익절 비율 덮기 (예 0.008)")
    p.add_argument("--stop-loss", type=float, default=None,
                   help="일상 손절 비율 덮기 (예 -0.004, 0 = 없음)")
    p.add_argument("--disaster-line", type=float, default=None,
                   help="재난선 비율 덮기 (예 -0.007, 0 = 없음)")
    p.add_argument("--hold", type=float, default=None, help="시간청산 초 덮기 (기본 90)")
    p.add_argument("--surge", type=float, default=None, help="대장 서지 임계 덮기 (기본 0.015)")
    p.add_argument("--cooldown", type=float, default=None, help="쌍 재알림 억제 초 (기본 90)")
    p.add_argument("--min-turnover", type=float, default=None,
                   help="후속 당일 누적 체결대금 하한(원). 0 = 끔")
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
    g.add_argument("--max-consecutive-losses", type=int, default=3,
                   help="연속 손절 상한(킬 트리거). 기본 3")
    g.add_argument("--max-positions", type=int, default=2,
                   help="동시 보유 가상 포지션 상한. 기본 2")
    g.add_argument("--fill-basis", choices=("through", "touch"), default="through",
                   help="가상 체결 회계 가정: through(보수·기본)|touch(낙관)")
    g.add_argument("--cost-rate", type=float, default=0.0,
                   help="왕복 비용률(수익률에서 차감). 0 = 끔")
    g.add_argument("--stop-file", default="data/pair_STOP",
                   help="존재 시 수동 킬스위치. 기본 data/pair_STOP")
    g.add_argument("--no-stop-file", action="store_true",
                   help="수동 정지 파일 감시 끄기")
    g.add_argument("--position-log", default="data/pair_positions.log",
                   help="가상 포지션 로그(MAE·지연·체결·손익). 기본 data/pair_positions.log")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = resolve_pairs(args)
    if not pairs:
        raise SystemExit("감시할 쌍이 없다. --pairs 또는 워치리스트를 확인하라.")
    detector = PairDetector(pairs, build_config(args))
    guard = build_guard(args)

    # 소켓 직결 모드 — DB 폴링을 안 거친다(저장은 collector 가 담당).
    if args.live:
        return run_live(args, detector, guard)

    conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
    # 후속주 구독 커버리지 경고 (오늘 ticks 에 후속주가 있는지).
    if args.replay is None:
        _warn_missing(conn, detector)

    if args.replay:
        return run_replay(args, conn, detector, guard)
    return run_realtime(args, conn, detector, guard)


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
