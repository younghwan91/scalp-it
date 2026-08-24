"""`scalp-book-report` — PEAD + 저변동성 결합 북 대시보드.

32번 결합 검증에서 채택된 **역변동성 결합 북**(PEAD 0.70 / 저변동 0.30)의 오늘자
현황을 한 화면에 모은다. 두 전략의 현재 롱숏 포지션·종목수·기준가와, 결합 비중을
적용한 **자본 100 기준 종목별 배분·합산 익스포저**를 보여준다.

선정은 각 전략 배포 스펙 함수를 그대로 재사용한다 — PEAD 는
:func:`scalp_it.cli_pead_watch.select_portfolio`, 저변동은
:func:`scalp_it.lowvol.select_lowvol_portfolio`. DB 는 읽기 전용.

결합 비중 근거(32번): 두 전략 net 월수익 표준편차의 역수 가중(risk parity). 전 표본
w_PEAD≈0.70·w_저변동≈0.30 에서 결합 샤프 1.57 > 단일 1.39·1.08, MDD -16%→-12%.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import pandas as pd

from .adapters.db import (
    connect,
    load_daily_bars_raw,
    load_earnings,
    load_prices,
    load_stock_master,
)
from .cli_pead_watch import Portfolio as PeadPortfolio
from .cli_pead_watch import select_portfolio as select_pead
from .lowvol import LowvolPortfolio, select_lowvol_portfolio

#: 32번 결합 검증에서 확정된 역변동성 결합 비중(전 표본). 배포 북의 자본 배분.
COMBO_W_PEAD = 0.70
COMBO_W_LOWVOL = 0.30


def _name_notional(portfolio, side_attr: str, book_weight: float) -> pd.DataFrame:
    """전략의 한 다리를 자본 100 기준 종목별 노출로. 롱:숏=1:1 (시장중립).

    각 전략은 결합 자본의 ``book_weight`` 를 받고, 그 안에서 롱·숏에 절반씩
    (gross = book_weight, 롱 = 숏 = book_weight/2), 다리 내 동일가중.
    """
    frame = getattr(portfolio, side_attr)
    n = len(frame)
    if n == 0:
        return pd.DataFrame(columns=["code", "name", "notional"])
    per = (book_weight / 2.0) / n * 100.0
    return pd.DataFrame({
        "code": frame["code"].astype(str).values,
        "name": frame["name"].values if "name" in frame.columns else "",
        "notional": per,
    })


def format_dashboard(
    pead: PeadPortfolio | None,
    lowvol: LowvolPortfolio | None,
    as_of: str,
    top: int = 10,
) -> str:
    lines = [
        f"# 결합 북 대시보드 (PEAD + 저변동성) — 기준일 {as_of}",
        "",
        (f"- 결합 비중(역변동성, 32번): **PEAD {COMBO_W_PEAD:.0%} / "
         f"저변동 {COMBO_W_LOWVOL:.0%}**. 결합 샤프 1.57 (단일 1.39·1.08), MDD -16%→-12%."),
        ("- 각 전략 시장중립(롱 총액 = 숏 총액). 자본 100 기준: "
         f"PEAD 롱 {COMBO_W_PEAD/2*100:.0f}·숏 {COMBO_W_PEAD/2*100:.0f}, "
         f"저변동 롱 {COMBO_W_LOWVOL/2*100:.0f}·숏 {COMBO_W_LOWVOL/2*100:.0f} "
         "(합산 gross 200, net ≈ 0)."),
        "",
    ]

    # 전략별 요약
    lines.append("## 전략별 현재 포지션")
    lines.append("")
    lines.append("| 전략 | 리밸/분기 | 롱 종목 | 숏 종목 | 결합 비중 |")
    lines.append("|---|---|---:|---:|---:|")
    if pead is not None:
        lines.append(
            f"| PEAD | {pead.period} (avail {pead.avail_date.date()}) | "
            f"{len(pead.longs)} | {len(pead.shorts)} | {COMBO_W_PEAD:.0%} |"
        )
    else:
        lines.append("| PEAD | (형성 불가) | - | - | - |")
    if lowvol is not None:
        lines.append(
            f"| 저변동 | {lowvol.period} (신호 {lowvol.asof_date.date()}) | "
            f"{len(lowvol.longs)} | {len(lowvol.shorts)} | {COMBO_W_LOWVOL:.0%} |"
        )
    else:
        lines.append("| 저변동 | (형성 불가) | - | - | - |")
    lines.append("")

    # 합산 익스포저 (종목별 노출 합)
    long_parts, short_parts = [], []
    if pead is not None:
        long_parts.append(_name_notional(pead, "longs", COMBO_W_PEAD))
        short_parts.append(_name_notional(pead, "shorts", COMBO_W_PEAD))
    if lowvol is not None:
        long_parts.append(_name_notional(lowvol, "longs", COMBO_W_LOWVOL))
        short_parts.append(_name_notional(lowvol, "shorts", COMBO_W_LOWVOL))

    def agg(parts):
        if not parts:
            return pd.DataFrame(columns=["code", "name", "notional"])
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby("code", as_index=False).agg(
            name=("name", "first"), notional=("notional", "sum"))
        return g.sort_values("notional", ascending=False).reset_index(drop=True)

    long_book = agg(long_parts)
    short_book = agg(short_parts)

    # 롱·숏 양쪽에 동시 등장하는 종목(상계) 표시
    overlap = set(long_book["code"]) & set(short_book["code"])

    lines.append("## 합산 익스포저 (자본 100 기준 종목별 노출)")
    lines.append("")
    lines.append(
        f"- 롱 {len(long_book)}종목(합 {long_book['notional'].sum():.0f}) · "
        f"숏 {len(short_book)}종목(합 {short_book['notional'].sum():.0f}) · "
        f"롱↔숏 상계 종목 {len(overlap)}개."
    )
    lines.append("")

    def dump_book(book: pd.DataFrame, title: str) -> None:
        lines.append(f"### {title} — 상위 {min(top, len(book))} (노출순)")
        lines.append("")
        lines.append("| # | 코드 | 이름 | 노출(자본%) |")
        lines.append("|---|---|---|---:|")
        for i, x in enumerate(book.head(top).itertuples(), 1):
            mark = " ⚠상계" if x.code in overlap else ""
            lines.append(f"| {i} | {x.code} | {x.name}{mark} | {x.notional:.2f} |")
        lines.append("")

    dump_book(long_book, "롱 북")
    dump_book(short_book, "숏 북")

    lines.append("## 운영 메모")
    lines.append("- PEAD 는 분기 리밸(+40거래일), 저변동은 월 리밸(+21거래일) — 각자 스케줄.")
    lines.append("- 상계(⚠) 종목은 한쪽 전략 롱·다른쪽 숏 — 순노출만 실제 집행하면 비용 절감.")
    lines.append("- 상세 종목표: `scalp-pead-portfolio`, `scalp-lowvol-portfolio`.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PEAD + 저변동성 결합 북 대시보드 — 현재 포지션·비중·합산 익스포저"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument("--top", type=int, default=10, help="합산 북 표시 상위 N (기본 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asof = (pd.Timestamp(args.asof) if args.asof
            else pd.Timestamp(pd.Timestamp.today().date()))
    start = str((asof - pd.tseries.offsets.BDay(400)).date())
    end = str(asof.date())

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1
    try:
        earn = load_earnings(conn)
        adj = load_prices(conn, start, end)
        raw = load_daily_bars_raw(conn, start, end)
        meta = load_stock_master(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 로드 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    spac = set(meta.loc[meta["name"].str.contains("스팩|기업인수목적", na=False), "code"])
    pead = select_pead(earn, adj, raw, meta, spac, asof)
    lowvol = select_lowvol_portfolio(adj, meta, asof)

    print(format_dashboard(pead, lowvol, str(asof.date()), top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
