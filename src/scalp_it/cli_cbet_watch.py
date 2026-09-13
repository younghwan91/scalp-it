"""`scalp-cbet-watch` — 종가베팅 전진 기록. 오늘 후보를 남기고, 지난 기록을 채점한다.

종가베팅은 이 프로젝트에서 시험한 7개 중 **유일하게 채택된 전략**이다
(`docs/research/manju/06-closebet-preregistration.md`). 11년 4,000여 건에서
34bp 차감 후 평균 **+2.73%**, 승률 61%, 손익비 2.62, 음수 해 0/11.

그런데 채택은 **전진 기록 착수**를 뜻하지 실전 투입이 아니다. 이 프로젝트는
백테스트를 믿었다가 두 번 틀렸다 (Minervini 분할 미조정 아티팩트, 스윙 ①층
중복매수). 남은 의심은 백테스트로 못 푼다. **오늘부터 기록을 남기는 것** 이
이 도구의 존재 이유이고, 값어치는 기능이 아니라 도는 데 있다.

규칙은 사전등록 확정본 그대로다. 여기서 파라미터를 만지면 그 순간 전진 기록의
자격이 사라진다 — 기록 중에 규칙을 바꾸면 그것은 더 이상 손대지 않은 창이 아니다.

    유니버스   당일 거래대금 >= 5000 (백만원 = 50억)
    스캔       tv_ratio >= 3.0 · close_pos >= 0.8
    순위       tv_ratio 내림차순, 하루 상위 2건
    진입       D 종가        청산  D+1 시가        비용  왕복 34bp

**운영 갭 — 이 도구는 사후 기록이지 실시간 신호가 아니다.**
백테스트는 종가 기준 피처로 종가에 산다고 가정한다. 그러나 실제 매매에서는
**15:20 동시호가가 시작되기 전에 주문을 내야 하고, 그 시점에는 종가를 모른다.**
실무에서는 15:25 경의 **예상체결가** 를 종가 대용으로 써야 하며, 그때 계산한
``close_pos``·``tv_ratio``·``ret_day`` 는 확정 종가로 계산한 값과 다르다.
이 CLI 는 확정 종가로 계산하므로 **장 마감 후에만 정확하다.** 분봉 수집이 끝나면
15:20 시점 가격과 종가의 괴리를 측정해 이 갭의 크기를 검증할 예정이며,
그 전까지 전진 기록의 수익률은 **갭만큼 낙관 방향** 임을 감안해서 읽어야 한다.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .adapters.db import connect, load_ohlcv, load_stock_master
from .closebet.features import close_bet_features
from .report.forward_log import append_forward_log, read_forward_log

#: 사전등록 확정 규칙. **바꾸지 마라.** 바꾸면 기록의 의미가 사라진다.
STRATEGY = "closebet"
RULE_VERSION = "v1-2026-08-17"
ADV_FLOOR = 5000.0  # 당일 거래대금 하한 (백만원 = 50억)
TV_RATIO_MIN = 3.0
CLOSE_POS_MIN = 0.8
TOP_N = 2
# 고정 가정값(값은 바꾸지 않는다). 일자별 거래세 스케줄은 krx_quant_core.costs 참고.
COST_ROUND_TRIP = 0.0034
ADV_WINDOW = 20

#: 11년 백테스트 기대치. 전진 결과가 여기서 얼마나 벗어나는지가 이 도구의 산출물이다.
EXPECTED = {"mean": 0.0307, "net34": 0.0273, "win_rate": 0.610}

_GAP_NOTE = (
    "※ 확정 종가로 계산한 **사후 기록** 이다. 실매매는 15:20 동시호가 전에 "
    "예상체결가로 주문해야 하므로 실시간 신호가 아니다."
)


def _payoff(returns: pd.Series) -> float:
    """평균이익 / 평균손실. 한쪽이 비면 정의할 수 없다."""
    wins, losses = returns[returns > 0], returns[returns < 0]
    if wins.empty or losses.empty:
        return float("nan")
    avg_loss = abs(losses.mean())
    return float("nan") if avg_loss == 0 else float(wins.mean() / avg_loss)


def select_candidates(ohlcv: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """``asof`` 하루치에 확정 규칙을 적용해 상위 ``TOP_N`` 건을 돌려준다.

    Returns:
        ``[code, date, close, trade_value, tv_ratio, close_pos, ret_day]``.
        조건을 만족하는 종목이 없거나 그날 봉이 없으면 빈 프레임.
    """
    columns = ["code", "date", "close", "trade_value", "tv_ratio", "close_pos", "ret_day"]
    if ohlcv.empty:
        return pd.DataFrame(columns=columns)

    features = close_bet_features(ohlcv, adv_window=ADV_WINDOW)
    day = ohlcv[ohlcv["date"] == asof][["code", "date", "close", "trade_value"]]
    if day.empty:
        return pd.DataFrame(columns=columns)

    merged = day.merge(
        features[["code", "date", "tv_ratio", "close_pos", "ret_day"]],
        on=["code", "date"],
        how="left",
    )
    hit = merged[
        (merged["trade_value"] >= ADV_FLOOR)
        & (merged["tv_ratio"] >= TV_RATIO_MIN)
        & (merged["close_pos"] >= CLOSE_POS_MIN)
    ]
    # 동률은 종목코드로 깬다 — 같은 입력에 같은 출력이어야 기록이 재현된다.
    ordered = hit.sort_values(["tv_ratio", "code"], ascending=[False, True])
    return ordered.head(TOP_N)[columns].reset_index(drop=True)


def next_trading_day(ohlcv: pd.DataFrame, asof: pd.Timestamp) -> pd.Timestamp:
    """``asof`` 다음 거래일. 달력에 없으면 영업일로 근사한다 (오늘이 마지막일 때)."""
    if not ohlcv.empty:
        later = ohlcv.loc[ohlcv["date"] > asof, "date"]
        if len(later):
            return pd.Timestamp(later.min())
    return pd.Timestamp(asof) + pd.tseries.offsets.BDay(1)


def build_records(
    candidates: pd.DataFrame, names: dict[str, str], *, logged_at: str
) -> list[dict]:
    """후보 프레임을 전진 기록 레코드로 바꾼다.

    ``layer`` 는 ``strategy`` 와 같은 값을 넣는다. :data:`~scalp_it.report.forward_log.LOG_KEY`
    가 ``layer`` 로 중복을 판정하므로, 이 별칭이 있어야 기존 append 로직을
    고치지 않고 (logged_at, strategy, code, event_date) 중복 제거가 성립한다.
    """
    return [
        {
            "logged_at": logged_at,
            "strategy": STRATEGY,
            "layer": STRATEGY,
            "code": str(row["code"]),
            "name": names.get(str(row["code"]), ""),
            "event_date": str(pd.Timestamp(row["date"]).date()),
            "close": float(row["close"]),
            "trade_value": float(row["trade_value"]),
            "tv_ratio": float(row["tv_ratio"]),
            "close_pos": float(row["close_pos"]),
            "ret_day": None if pd.isna(row["ret_day"]) else float(row["ret_day"]),
            "rule_version": RULE_VERSION,
        }
        for _, row in candidates.iterrows()
    ]


def format_candidates(records: list[dict], exit_date: pd.Timestamp) -> str:
    """사람이 읽는 후보 표."""
    header = (
        "| 종목코드 | 종목명 | 종가 | 거래대금(억) | tv_ratio | close_pos | 등락률 | 목표 청산일 |\n"
        "|---|---|---:|---:|---:|---:|---:|---|"
    )
    rows = [
        "| {code} | {name} | {close:,.0f} | {tv:,.0f} | {ratio:.2f} | {pos:.2f} "
        "| {ret} | {exit_date} (시가) |".format(
            code=rec["code"],
            name=rec["name"],
            close=rec["close"],
            tv=rec["trade_value"] / 100.0,  # 백만원 -> 억원
            ratio=rec["tv_ratio"],
            pos=rec["close_pos"],
            ret="—" if rec["ret_day"] is None else f"{rec['ret_day']:+.2%}",
            exit_date=str(pd.Timestamp(exit_date).date()),
        )
        for rec in records
    ]
    return "\n".join([header, *rows])


def _next_open(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """``[code, date, next_open]`` — 각 (종목, 날짜) 의 **다음 거래일 시가**."""
    if ohlcv.empty:
        return pd.DataFrame(columns=["code", "date", "next_open"])
    out = ohlcv.sort_values(["code", "date"])[["code", "date", "open"]].copy()
    out["next_open"] = out.groupby("code", sort=False)["open"].shift(-1)
    return out[["code", "date", "next_open"]]


def score(log: pd.DataFrame, ohlcv: pd.DataFrame, log_path: str) -> str:
    """전진 기록을 익일 시가로 채점한 보고서 문자열."""
    lines = [f"# 종가베팅 전진 기록 채점 — {log_path}", "", _GAP_NOTE, ""]
    if log.empty:
        return "\n".join([*lines, f"전진 기록이 비어 있다: {log_path}"])

    scored = log.copy()
    scored["code"] = scored["code"].astype(str)
    scored["_date"] = pd.to_datetime(scored["event_date"])
    nxt = _next_open(ohlcv)
    if nxt.empty:
        scored["next_open"] = float("nan")
    else:
        nxt = nxt.rename(columns={"date": "_date"})
        nxt["code"] = nxt["code"].astype(str)
        scored = scored.merge(nxt, on=["code", "_date"], how="left")

    scored["gross"] = (scored["next_open"] - scored["close"]) / scored["close"]
    done = scored[scored["gross"].notna()].copy()
    pending = len(scored) - len(done)

    lines.append(f"기록 {len(scored)}건 · 채점 {len(done)}건 · 미채점 {pending}건 (익일 시가 미도래)")
    lines.append("")
    if done.empty:
        lines.append("채점 가능한 기록이 아직 없다. 기록을 계속 쌓아라.")
        return "\n".join(lines)

    gross = done["gross"]
    mean, net = float(gross.mean()), float(gross.mean() - COST_ROUND_TRIP)
    win = float((gross > 0).mean())
    lines += [
        f"평균 **{mean:+.2%}** · 34bp 차감 후 **{net:+.2%}** · 중앙값 {gross.median():+.2%}",
        f"승률 {win:.1%} · 손익비 {_payoff(gross):.2f}",
        "",
        "## 백테스트 기대치와 나란히",
        "",
        "| 지표 | 전진 | 백테스트(11년) | 차이 |",
        "|---|---:|---:|---:|",
        f"| 평균 | {mean:+.2%} | {EXPECTED['mean']:+.2%} | {mean - EXPECTED['mean']:+.2%} |",
        f"| 34bp 차감 후 | {net:+.2%} | {EXPECTED['net34']:+.2%} | {net - EXPECTED['net34']:+.2%} |",
        (f"| 승률 | {win:.1%} | {EXPECTED['win_rate']:.1%} | "
         f"{(win - EXPECTED['win_rate']) * 100:+.1f}p |"),
        "",
        f"표본 {len(done)}건. 이 수가 수백 건이 되기 전에는 차이를 해석하지 마라.",
        "",
        "| 기록일 | 이벤트일 | 종목코드 | 종목명 | 진입가 | 청산가 | 수익률 | 34bp 후 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    ordered = done.sort_values(["event_date", "tv_ratio"], ascending=[True, False])
    lines += [
        f"| {row['logged_at']} | {row['event_date']} | {row['code']} | "
        f"{row.get('name', '')} | {row['close']:,.0f} | {row['next_open']:,.0f} | "
        f"{row['gross']:+.2%} | {row['gross'] - COST_ROUND_TRIP:+.2%} |"
        for _, row in ordered.iterrows()
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="종가베팅 전진 기록 — 오늘 후보를 남기고 지난 기록을 채점한다"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--start", default=None,
                        help="가격 로드 시작일. 생략 시 --asof 에서 60거래일 전")
    parser.add_argument("--log-path", default="data/cbet_forward_log.jsonl")
    parser.add_argument("--score", action="store_true",
                        help="기록하지 않고, 익일 시가가 나온 과거 기록만 채점한다")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    return parser


def _score_main(args: argparse.Namespace) -> int:
    log = read_forward_log(args.log_path)
    if not log.empty and "strategy" in log.columns:
        log = log[log["strategy"] == STRATEGY]
    if log.empty:
        print(score(pd.DataFrame(), pd.DataFrame(), args.log_path))
        return 0

    # 기록된 이벤트일 전체 + 익일을 덮는 창이면 충분하다.
    start = str(pd.to_datetime(log["event_date"]).min().date())
    end = str((pd.Timestamp.today() + pd.Timedelta(days=1)).date())
    conn = connect(args.dsn)
    try:
        ohlcv = load_ohlcv(conn, start, end)
    finally:
        conn.close()
    print(score(log, ohlcv, args.log_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.score:
            return _score_main(args)
    except Exception as exc:  # noqa: BLE001 — DSN 미설정·드라이버별 오류를 한데 잡는다
        print(f"DB 접속·로드 실패: {exc}", file=sys.stderr)
        return 1

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(pd.Timestamp.today().date())
    # 워밍업 20거래일 + 여유. 공휴일 때문에 영업일 60 은 실제 거래일 55 안팎이다.
    start = args.start or str((asof - pd.tseries.offsets.BDay(60)).date())

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1
    try:
        ohlcv = load_ohlcv(conn, start, str(asof.date()))
        master = load_stock_master(conn)
    finally:
        conn.close()

    names = dict(zip(master["code"].astype(str), master["name"])) if len(master) else {}
    candidates = select_candidates(ohlcv, asof)

    print(f"# 종가베팅 후보 — {asof.date()}  (규칙 {RULE_VERSION})")
    print()
    print(_GAP_NOTE)
    print()
    if candidates.empty:
        print("**오늘은 후보 없음.**  0 건은 실패가 아니다 — 재료가 없는 날이 정답인 날이다.")
        return 0

    logged_at = str(pd.Timestamp.today().date())
    records = build_records(candidates, names, logged_at=logged_at)
    exit_date = next_trading_day(ohlcv, asof)
    print(format_candidates(records, exit_date))
    print()

    written = append_forward_log(args.log_path, records)
    print(f"기록 {written}건 추가 (중복 {len(records) - written}건 건너뜀) → {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
