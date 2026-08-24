"""`scalp-pead-report` — PEAD 전진 기록 성과 채점.

``data/pead_forward_log.jsonl`` 을 읽어, 각 분기 포지션의 손익을 진입가 대비로
매긴다. 목표청산일(+40거래일)이 지난 포지션은 **청산가로 확정(realized)**, 아직
보유 중이면 최근 종가로 **미실현(unrealized)** 평가한다. 롱/숏/롱숏 합산과 누적,
그리고 백테스트 기대(분기 net +4.47%)를 나란히 표로 낸다.

손익 정의는 전진 기록과 같다 — 롱은 (현재/청산가 - 진입가)/진입가, 숏은 그 부호를
뒤집는다. 롱숏 분기수익(spread)은 롱 평균 + 숏 평균(둘 다 이미 '이익' 부호)이며
백테스트 net +4.47% 와 직접 비교 가능한 값이다. gross(비용 전)와 net(왕복 68bp
+ 숏차입 차감)을 함께 보인다.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np
import pandas as pd

from .adapters.db import connect, load_prices
from .cli_pead_watch import BORROW_40D, COST_ROUND_TRIP, EXPECTED, STRATEGY
from .report.forward_log import read_forward_log


def _price_asof(prices: pd.DataFrame, code: str, date: pd.Timestamp) -> float:
    """code 의 date 이하 마지막 종가. 없으면 NaN."""
    if prices.empty:
        return float("nan")
    sub = prices[(prices["code"] == code) & (prices["date"] <= date)]
    if sub.empty:
        return float("nan")
    return float(sub.sort_values("date")["close"].iloc[-1])


def evaluate_log(
    log: pd.DataFrame, prices: pd.DataFrame, asof: pd.Timestamp
) -> pd.DataFrame:
    """각 로그 포지션에 손익을 붙인다.

    반환 컬럼: period, side, code, name, entry_open, exit_date_planned, status
    ('realized'|'held'|'no_price'), exit_price, pnl_gross(부호적용), pnl_net.
    """
    if log.empty:
        return pd.DataFrame(
            columns=[
                "period", "side", "code", "name", "entry_open",
                "exit_date_planned", "status", "exit_price", "pnl_gross", "pnl_net",
            ]
        )
    df = log.copy()
    df["code"] = df["code"].astype(str)
    df["exit_dt"] = pd.to_datetime(df["exit_date_planned"])
    prices = prices.copy()
    if not prices.empty:
        prices["code"] = prices["code"].astype(str)
        prices["date"] = pd.to_datetime(prices["date"])

    out = []
    for r in df.itertuples():
        realized = r.exit_dt <= asof
        px_date = r.exit_dt if realized else asof
        px = _price_asof(prices, r.code, px_date)
        if np.isnan(px):
            status, pnl_gross, pnl_net = "no_price", np.nan, np.nan
        else:
            raw = (px - r.entry_open) / r.entry_open
            pnl_gross = raw if r.side == "long" else -raw
            # 비용: 롱숏 왕복 68bp 를 다리마다 절반씩, 숏은 차입비 추가
            cost = COST_ROUND_TRIP / 2 + (BORROW_40D if r.side == "short" else 0.0)
            pnl_net = pnl_gross - cost
            status = "realized" if realized else "held"
        out.append(
            {
                "period": r.period,
                "side": r.side,
                "code": r.code,
                "name": getattr(r, "name", ""),
                "entry_open": r.entry_open,
                "exit_date_planned": r.exit_date_planned,
                "status": status,
                "exit_price": px,
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
            }
        )
    return pd.DataFrame(out)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """분기별 롱/숏 평균과 롱숏 spread(=롱+숏). 비용 전/후."""
    valid = scored[scored["status"].isin(["realized", "held"])]
    rows = []
    for period, g in valid.groupby("period"):
        longs = g[g["side"] == "long"]
        shorts = g[g["side"] == "short"]
        status = "realized" if (g["status"] == "realized").all() else "held(일부미실현)"
        lg, ln = longs["pnl_gross"].mean(), longs["pnl_net"].mean()
        sg, sn = shorts["pnl_gross"].mean(), shorts["pnl_net"].mean()
        rows.append(
            {
                "period": period,
                "status": status,
                "n_long": len(longs),
                "n_short": len(shorts),
                "long_gross": lg,
                "short_gross": sg,
                "longshort_gross": (0.0 if np.isnan(lg) else lg)
                + (0.0 if np.isnan(sg) else sg),
                "long_net": ln,
                "short_net": sn,
                "longshort_net": (0.0 if np.isnan(ln) else ln)
                + (0.0 if np.isnan(sn) else sn),
            }
        )
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def format_report(scored: pd.DataFrame, summary: pd.DataFrame, asof: pd.Timestamp) -> str:
    lines = [
        f"# PEAD 전진 기록 성과 (기준일 {asof.date()})",
        "",
    ]
    if scored.empty:
        lines.append("전진 기록이 비어 있다 (아직 기록된 분기 없음).")
        return "\n".join(lines)

    n_real = (scored["status"] == "realized").sum()
    n_held = (scored["status"] == "held").sum()
    n_np = (scored["status"] == "no_price").sum()
    lines.append(
        f"- 포지션 {len(scored)}건: 청산확정 {n_real} · 보유중 {n_held} · "
        f"가격없음 {n_np}."
    )
    lines.append(
        f"- 백테스트 기대: 분기 롱숏 net **+{EXPECTED['net_longshort']:.2%}** "
        f"(t={EXPECTED['t']})."
    )
    lines.append("")
    lines.append("## 분기별 성과")
    lines.append("")
    lines.append(
        "| 분기 | 상태 | 롱n | 숏n | 롱 gross | 숏 gross | 롱숏 gross | "
        "롱숏 net | vs 기대 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for x in summary.itertuples():
        diff = x.longshort_net - EXPECTED["net_longshort"]
        lines.append(
            f"| {x.period} | {x.status} | {x.n_long} | {x.n_short} | "
            f"{x.long_gross:+.2%} | {x.short_gross:+.2%} | {x.longshort_gross:+.2%} | "
            f"{x.longshort_net:+.2%} | {diff:+.2%} |"
        )
    lines.append("")

    # 누적(분기 net 단순 합 = 시장중립 분기수익 누적)
    cum_net = summary["longshort_net"].sum()
    cum_gross = summary["longshort_gross"].sum()
    lines.append(
        f"- **누적 롱숏: gross {cum_gross:+.2%} · net {cum_net:+.2%}** "
        f"({len(summary)}개 분기)."
    )
    lines.append(
        "- 청산확정 전 분기는 미실현 참고치다 — 목표청산(+40거래일) 도래 시 확정된다."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PEAD 전진 기록 성과 채점 — 실현/미실현 손익, 백테스트 기대와 비교"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--log-path", default="data/pead_forward_log.jsonl")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp(pd.Timestamp.today().date())
    )
    log = read_forward_log(args.log_path)
    if not log.empty and "strategy" in log.columns:
        log = log[log["strategy"] == STRATEGY]

    if log.empty:
        print(format_report(pd.DataFrame(), pd.DataFrame(), asof))
        return 0

    codes = [str(c) for c in log["code"].unique()]
    start = str((pd.to_datetime(log["entry_date"]).min() - pd.Timedelta(days=5)).date())
    end = str(asof.date())
    try:
        conn = connect(args.dsn)
        prices = load_prices(conn, start, end)
        prices = prices[prices["code"].astype(str).isin(codes)]
    except Exception as exc:  # noqa: BLE001
        print(f"DB 로드 실패 (가격 없이 진행): {exc}", file=sys.stderr)
        prices = pd.DataFrame(columns=["code", "date", "open", "close", "trade_value"])
    else:
        with contextlib.suppress(Exception):
            conn.close()

    scored = evaluate_log(log, prices, asof)
    summary = summarize(scored)
    print(format_report(scored, summary, asof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
