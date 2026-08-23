"""`scalp-pead-watch` — PEAD 전진 기록. 최신 분기의 롱숏을 확정해 로그로 남긴다.

PEAD(공시 후 어닝 서프라이즈 드리프트)는 이 프로젝트에서 **백테스트가 구조적으로
성립한 첫 팩터이자, 강건성까지 통과해 실전 채택된 전략**이다
(`docs/research/manju/23`·`24`). 11년에 걸쳐 롱숏 시장중립, 50억 유동성서
34bp+숏차입 차감 후 **net +4.47%**(t=6.15), 500억서도 net>0(t=2.54).

그러나 채택은 **전진 기록 착수**를 뜻하지 실전 성과의 증명이 아니다. 이 프로젝트는
백테스트를 믿었다가 두 번 틀렸다(분할 미조정, 중복매수). walk-forward 도 과거를
잘라 쓴 것이라 '손 안 댄 창'이 아니다. **오늘부터 실제 포트폴리오를 로그로 남겨
6개월·1년 뒤 실현 성과를 채점하는 것** 이 이 도구의 존재 이유다.

규칙은 배포 스펙 확정본 그대로다. 여기서 파라미터를 만지면 그 순간 전진 기록의
자격이 사라진다.

    유니버스   진입 직전 20거래일 평균 거래대금 >= 5000 (백만원 = 50억), 스팩 제외,
               진입일 상한가(체결불가) 제외
    시그널     영업이익 YoY 서프라이즈 SUE = (op_income - op_income_prior)/|op_income_prior|
    포트폴리오 분기내 데실. **롱 = Q10(서프라이즈 최상), 숏 = Q1(최하).** 각 동일가중
    진입       avail 익일 시가        보유  +40거래일(분기 리밸)        비용  왕복 68bp+숏차입

분기 리밸이라 매일 돌 필요는 없다. 새 분기 실적 avail 이 뜨면 새 롱숏이 형성되고,
그때 한 번 기록하면 된다. 이미 기록한 분기는 조용히 건너뛴다(append-only).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adapters.db import (
    connect,
    load_daily_bars_raw,
    load_earnings,
    load_prices,
    load_stock_master,
)
from .report.forward_log import append_forward_log, read_forward_log

#: 배포 스펙 확정 규칙. **바꾸지 마라.** 바꾸면 기록의 의미가 사라진다.
STRATEGY = "pead"
RULE_VERSION = "v1-2026-08-23"
DEPLOY_LIQ = 5000.0  # 진입 직전 20거래일 평균 거래대금 하한 (백만원 = 50억)
LIQ_WINDOW = 20
HOLD_DAYS = 40  # 보유 거래일 (분기 리밸)
N_DECILES = 10
MIN_NAMES = 30  # 한 분기가 포트폴리오를 형성하려면 유니버스 >= 이 수 (데실당 3+)
LIMIT_UP = 1.29  # 진입일 시가/전일종가 >= 이 값이면 상한가 체결불가로 제외
COST_ROUND_TRIP = 0.0068  # 롱숏 왕복 68bp
BORROW_40D = 0.005  # 숏 차입 ~2개월

#: 배포 스펙 기대치 (24번). 전진 결과가 여기서 얼마나 벗어나는지가 산출물이다.
EXPECTED = {"net_longshort": 0.0447, "t": 6.15}

#: PEAD 전진 기록의 중복 판정 키. ``logged_at`` 을 넣지 않는다 — 같은 분기를
#: 다른 날 다시 돌려도 재기록하면 안 되기 때문이다.
DEDUP_KEY = ("strategy", "period", "code", "side")


@dataclass
class Portfolio:
    """최신 avail 분기의 확정 롱숏."""

    period: str
    avail_date: pd.Timestamp
    longs: pd.DataFrame
    shorts: pd.DataFrame


def _build_events(
    earn: pd.DataFrame,
    adj: pd.DataFrame,
    raw: pd.DataFrame,
    spac: set[str],
) -> pd.DataFrame:
    """(code, period) 마다 진입가·유동성·상한가·SUE 를 계산한다.

    ``scripts/pead_events.build_events`` 의 배포에 필요한 부분만 옮긴 것이다.
    진입 = avail_date 초과 첫 거래일 시가(수정주가). 유동성 = 진입 직전 20거래일
    평균 거래대금. 상한가 = 원주가 익일 시가 / 전일 종가.
    """
    adj_by_code: dict[str, tuple] = {}
    for code, g in adj.sort_values(["code", "date"]).groupby("code", sort=False):
        adj_by_code[str(code)] = (
            g["date"].values,
            g["open"].values.astype(float),
            g["trade_value"].values.astype(float),
        )
    raw_by_code: dict[str, tuple] = {}
    if raw is not None and not raw.empty:
        for code, g in raw.sort_values(["code", "date"]).groupby("code", sort=False):
            raw_by_code[str(code)] = (
                g["date"].values,
                g["open"].values.astype(float),
                g["close"].values.astype(float),
            )

    rows = []
    for r in earn.itertuples(index=False):
        code = str(r.code)
        if code not in adj_by_code or pd.isna(r.avail_date):
            continue
        avail = np.datetime64(pd.Timestamp(r.avail_date), "ns")
        dates, opens, tvs = adj_by_code[code]
        ei = int(np.searchsorted(dates, avail, side="right"))
        if ei >= len(dates):
            continue  # avail 익일 거래일이 아직 없다 (진입가 미형성)
        entry_open = float(opens[ei])
        entry_date = pd.Timestamp(dates[ei])
        lo = max(0, ei - LIQ_WINDOW)
        liq = float(np.nanmean(tvs[lo:ei])) if ei - lo >= LIQ_WINDOW else np.nan

        oi, oip = r.op_income, r.op_income_prior
        has_sig = pd.notna(oi) and pd.notna(oip) and oip != 0
        sue_raw = (float(oi) - float(oip)) / abs(float(oip)) if has_sig else np.nan

        entry_limit_up = False
        if code in raw_by_code:
            rd, ro, rc = raw_by_code[code]
            rei = int(np.searchsorted(rd, avail, side="right"))
            if 0 < rei < len(rd):
                prev_close = rc[rei - 1]
                if prev_close > 0 and ro[rei] / prev_close >= LIMIT_UP:
                    entry_limit_up = True

        rows.append(
            {
                "code": code,
                "period": str(r.period),
                "avail_date": pd.Timestamp(r.avail_date),
                "entry_date": entry_date,
                "entry_open": entry_open,
                "liq": liq,
                "sue_raw": sue_raw,
                "entry_limit_up": entry_limit_up,
            }
        )
    ev = pd.DataFrame(
        rows,
        columns=[
            "code", "period", "avail_date", "entry_date", "entry_open",
            "liq", "sue_raw", "entry_limit_up",
        ],
    )
    ev["spac"] = ev["code"].isin(spac)
    return ev


def _plan_exit_date(entry_date: pd.Timestamp, calendar: np.ndarray) -> pd.Timestamp:
    """진입일 + ``HOLD_DAYS`` 거래일. 달력을 벗어나면 영업일로 근사한다."""
    idx = int(np.searchsorted(calendar, np.datetime64(entry_date, "ns"), side="left"))
    tgt = idx + HOLD_DAYS
    if 0 <= tgt < len(calendar):
        return pd.Timestamp(calendar[tgt])
    return entry_date + pd.tseries.offsets.BDay(HOLD_DAYS)


def select_portfolio(
    earn: pd.DataFrame,
    adj: pd.DataFrame,
    raw: pd.DataFrame,
    meta: pd.DataFrame,
    spac: set[str],
    asof: pd.Timestamp,
) -> Portfolio | None:
    """최신 형성 가능한 avail 분기의 롱(Q10)·숏(Q1) 을 확정한다.

    형성 가능 = 진입가가 이미 존재(entry_date <= asof)하고 유니버스가
    ``MIN_NAMES`` 이상인 최대 period. 그런 분기가 없으면 ``None``.
    """
    ev = _build_events(earn, adj, raw, spac)
    uni = ev[
        ~ev["spac"]
        & ev["sue_raw"].notna()
        & ~ev["entry_limit_up"]
        & (ev["liq"] >= DEPLOY_LIQ)
        & ev["entry_date"].notna()
        & (ev["entry_date"] <= asof)
    ].copy()
    if uni.empty:
        return None
    counts = uni.groupby("period").size()
    cand = counts[counts >= MIN_NAMES].index
    if len(cand) == 0:
        return None
    period = max(cand)
    cur = uni[uni["period"] == period].copy()

    r = cur["sue_raw"].rank(method="first")
    cur["decile"] = pd.qcut(r, N_DECILES, labels=range(1, N_DECILES + 1)).astype(int)

    name_map = dict(zip(meta["code"].astype(str), meta["name"])) if len(meta) else {}
    sec_map = (
        dict(zip(meta["code"].astype(str), meta["sector"]))
        if "sector" in meta.columns else {}
    )
    mkt_map = (
        dict(zip(meta["code"].astype(str), meta["market"]))
        if "market" in meta.columns else {}
    )
    cur["name"] = cur["code"].map(name_map).fillna("")
    cur["sector"] = cur["code"].map(sec_map)
    cur["market"] = cur["code"].map(mkt_map)

    calendar = np.unique(adj["date"].values)  # 종목 중복 제거한 거래일 달력
    cur["exit_date_planned"] = cur["entry_date"].map(
        lambda d: _plan_exit_date(d, calendar)
    )

    longs = cur[cur["decile"] == N_DECILES].sort_values("sue_raw", ascending=False)
    shorts = cur[cur["decile"] == 1].sort_values("sue_raw", ascending=True)
    return Portfolio(
        period=period,
        avail_date=cur["avail_date"].max(),
        longs=longs.reset_index(drop=True),
        shorts=shorts.reset_index(drop=True),
    )


def build_records(portfolio: Portfolio, *, logged_at: str) -> list[dict]:
    """확정 포트폴리오를 전진 기록 레코드로 바꾼다.

    ``layer`` 는 ``pead:{period}`` 로 넣어 기존 채점 도구와 호환시키되, 중복
    판정은 :data:`DEDUP_KEY`(``logged_at`` 제외)로 한다.
    """
    records = []
    for side, frame in (("long", portfolio.longs), ("short", portfolio.shorts)):
        for x in frame.itertuples():
            records.append(
                {
                    "logged_at": logged_at,
                    "strategy": STRATEGY,
                    "layer": f"{STRATEGY}:{portfolio.period}",
                    "period": portfolio.period,
                    "side": side,
                    "code": str(x.code),
                    "name": x.name,
                    "market": None if pd.isna(x.market) else x.market,
                    "sector": None if pd.isna(x.sector) else x.sector,
                    "sue_raw": float(x.sue_raw),
                    "liq": float(x.liq),
                    "entry_date": str(pd.Timestamp(x.entry_date).date()),
                    "entry_open": float(x.entry_open),
                    "exit_date_planned": str(pd.Timestamp(x.exit_date_planned).date()),
                    "rule_version": RULE_VERSION,
                }
            )
    return records


def format_portfolio(portfolio: Portfolio) -> str:
    """사람이 읽는 롱숏 표."""
    lines = [
        f"# PEAD 전진 기록 — 분기 {portfolio.period} (avail {portfolio.avail_date.date()})",
        "",
        f"- 유니버스 유동성 >= {DEPLOY_LIQ / 100:.0f}억, 스팩·진입일 상한가 제외.",
        (f"- **롱 = Q10({len(portfolio.longs)}종목), 숏 = Q1({len(portfolio.shorts)}종목)**, "
         "각 동일가중. 진입: avail 익일 시가. 보유: +40거래일(분기 리밸)."),
        "",
    ]

    def dump(frame: pd.DataFrame, title: str) -> None:
        lines.append(f"## {title} ({len(frame)}종목)")
        lines.append("")
        lines.append(
            "| # | 코드 | 이름 | 시장 | SUE | 20일거래대금(억) | 진입일 | 기준가 | 목표청산 |"
        )
        lines.append("|---|---|---|---|---:|---:|---|---:|---|")
        for i, x in enumerate(frame.itertuples(), 1):
            lines.append(
                f"| {i} | {x.code} | {x.name} | {'' if pd.isna(x.market) else x.market} | "
                f"{x.sue_raw:+.2f} | {x.liq / 100:.0f} | "
                f"{pd.Timestamp(x.entry_date).date()} | {x.entry_open:,.0f} | "
                f"{pd.Timestamp(x.exit_date_planned).date()} |"
            )
        lines.append("")

    dump(portfolio.longs, "롱 (Q10) — 매수")
    dump(portfolio.shorts, "숏 (Q1) — 대차 매도")
    return "\n".join(lines)


def evaluate_holdings(log: pd.DataFrame, adj: pd.DataFrame, asof: pd.Timestamp) -> str:
    """보유 중(목표청산 미도래) 포지션의 진입가 대비 현재 평가."""
    if log.empty:
        return "보유 중 포지션 없음 (전진 기록이 비어 있다)."
    held = log.copy()
    held["code"] = held["code"].astype(str)
    held = held[pd.to_datetime(held["exit_date_planned"]) > asof]
    if held.empty:
        return "보유 중 포지션 없음 (모두 목표청산일 지남)."

    if adj.empty:
        last_close = pd.Series(dtype=float)
    else:
        recent = adj[adj["date"] <= asof]
        last_close = (
            recent.sort_values("date").groupby("code")["close"].last()
        )
    held["cur"] = held["code"].map(last_close)
    scored = held[held["cur"].notna()].copy()
    if scored.empty:
        return "보유 중 포지션이 있으나 현재가를 못 붙였다 (daily_bars 미도래)."

    raw = (scored["cur"] - scored["entry_open"]) / scored["entry_open"]
    scored["pnl"] = np.where(scored["side"] == "long", raw, -raw)

    longs = scored[scored["side"] == "long"]["pnl"]
    shorts = scored[scored["side"] == "short"]["pnl"]
    lines = [
        f"## 보유 중 포지션 평가 (기준 {asof.date()}, 최근 종가 대비 진입가)",
        "",
        (f"보유 {len(scored)}건 (롱 {len(longs)} · 숏 {len(shorts)}) · "
         f"미평가 {len(held) - len(scored)}건 (현재가 없음)"),
    ]
    if not longs.empty:
        lines.append(f"- 롱 평균 {longs.mean():+.2%} (승률 {(longs > 0).mean():.0%})")
    if not shorts.empty:
        lines.append(f"- 숏 평균 {shorts.mean():+.2%} (승률 {(shorts > 0).mean():.0%})")
    lines.append(
        f"- 롱숏 합산(동일가중) 평균 {scored['pnl'].mean():+.2%} "
        "— **비용 미차감, 미실현.** 목표청산 전이라 참고치다."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PEAD 전진 기록 — 최신 분기 롱숏을 확정해 로그로 남긴다"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--log-path", default="data/pead_forward_log.jsonl")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument(
        "--dry-run", action="store_true", help="화면에만 출력하고 로그에 쓰지 않는다"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asof = (
        pd.Timestamp(args.asof) if args.asof else pd.Timestamp(pd.Timestamp.today().date())
    )
    # 최신 분기 avail(수개월 전)까지 덮으려면 넉넉히. 20거래일 워밍업 + 40 보유 여유.
    start = str((asof - pd.tseries.offsets.BDay(400)).date())
    end = str(asof.date())

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001 — DSN 미설정·드라이버별 오류를 한데 잡는다
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1
    try:
        earn = load_earnings(conn)
        adj = load_prices(conn, start, end)
        raw = load_daily_bars_raw(conn, start, end)
        meta = load_stock_master(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 로드 실패: {exc}", file=sys.stderr)
        conn.close()
        return 1
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    spac = set(meta.loc[meta["name"].str.contains("스팩|기업인수목적", na=False), "code"])

    portfolio = select_portfolio(earn, adj, raw, meta, spac, asof)
    log = read_forward_log(args.log_path)
    if not log.empty and "strategy" in log.columns:
        pead_log = log[log["strategy"] == STRATEGY]
    else:
        pead_log = pd.DataFrame()

    if portfolio is None:
        print(f"# PEAD 전진 기록 — {asof.date()}")
        print()
        print("형성 가능한 avail 분기가 없다 (진입가 미형성 또는 유니버스 부족).")
        print()
        print(evaluate_holdings(pead_log, adj, asof))
        return 0

    print(format_portfolio(portfolio))

    already = (
        not pead_log.empty
        and "period" in pead_log.columns
        and portfolio.period in set(pead_log["period"].astype(str))
    )
    if already:
        print(f"이미 기록된 분기 **{portfolio.period}** → 건너뜀 (append-only).")
    else:
        records = build_records(portfolio, logged_at=str(pd.Timestamp.today().date()))
        if args.dry_run:
            print(f"--dry-run: {len(records)}건 기록 예정 (쓰지 않음).")
        else:
            written = append_forward_log(args.log_path, records, key=DEDUP_KEY)
            print(
                f"기록 {written}건 추가 (중복 {len(records) - written}건 건너뜀) "
                f"→ {args.log_path}"
            )
            pead_log = read_forward_log(args.log_path)
            pead_log = pead_log[pead_log["strategy"] == STRATEGY]

    print()
    print(evaluate_holdings(pead_log, adj, asof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
