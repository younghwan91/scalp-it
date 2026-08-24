"""`scalp-lowvol-portfolio` — 오늘 실제로 매매할 저변동성 롱숏 포트폴리오.

31번 다중팩터 배치에서 채택된 저변동성 팩터의 배포 산출물이다. 리밸일(오늘 이하 최신
거래일) 기준으로 폴드내 데실을 매겨 **롱 = Q10(저변동)·숏 = Q1(고변동)** 을 확정하고,
종목코드·이름·시장·연율변동성·데실·기준가·비중(동일가중 %)을 표로 낸 뒤
``data/lowvol_portfolio_YYYY-MM.csv`` 로 저장한다. PEAD 포트폴리오 CLI 와 대칭.

선정 로직은 :mod:`scalp_it.lowvol` 의 :func:`select_lowvol_portfolio` 한 곳에서만
정의된다(결합 대시보드 :mod:`scalp_it.cli_book_report` 와 공유). DB 는 읽기만 하고,
쓰는 것은 ``data/`` CSV 뿐이다.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import pandas as pd

from .adapters.db import connect, load_prices, load_stock_master
from .lowvol import (
    BORROW_21D,
    COST_ROUND_TRIP,
    DEPLOY_LIQ,
    EXPECTED,
    HOLD_DAYS,
    RULE_VERSION,
    LowvolPortfolio,
    build_frame,
    select_lowvol_portfolio,
)


def format_report(portfolio: LowvolPortfolio, frame: pd.DataFrame, as_of: str) -> str:
    cost_pct = (COST_ROUND_TRIP + BORROW_21D) * 100
    lines = [
        (f"# 저변동성 실전 포트폴리오 — 리밸 {portfolio.period} "
         f"(신호일 {portfolio.asof_date.date()}, 기준일 {as_of})"),
        "",
        (f"- 유니버스: 리밸일 직전 20거래일 평균 거래대금 >= {DEPLOY_LIQ / 100:.0f}억, "
         "스팩·리츠 제외."),
        (f"- **롱 = Q10 저변동({len(portfolio.longs)}종목), 숏 = Q1 고변동"
         f"({len(portfolio.shorts)}종목)**, 각 동일가중(시장중립: 롱 총액 = 숏 총액)."),
        (f"- 진입: 리밸일 익일 시가. 보유: +{HOLD_DAYS}거래일(월 리밸). "
         f"비용: 왕복 68bp + 숏차입(월 ≈{cost_pct:.2f}%)."),
        (f"- 규칙 버전 {RULE_VERSION}. 백테스트 기대 월 롱숏 net "
         f"+{EXPECTED['net_longshort']:.2%} (t={EXPECTED['t']})."),
        "",
    ]

    def dump(side: str, title: str) -> None:
        sub = frame[frame["side"] == side]
        wt = sub["weight_pct"].iloc[0] if len(sub) else 0.0
        lines.append(f"## {title} ({len(sub)}종목, 각 {wt:.2f}%)")
        lines.append("")
        lines.append(
            "| # | 코드 | 이름 | 시장 | 데실 | 연율변동성% | 기준가 | 비중% | 20일대금(억) |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for i, x in enumerate(sub.itertuples(), 1):
            lines.append(
                f"| {i} | {x.code} | {x.name} | {x.market} | {x.decile} | "
                f"{x.vol_ann_pct:.1f} | {x.close:,.0f} | {x.weight_pct:.2f} | "
                f"{x.liq_eok:.0f} |"
            )
        lines.append("")

    dump("롱", "롱 (Q10 저변동) — 매수")
    dump("숏", "숏 (Q1 고변동) — 대차 매도")
    lines.append("## 실전 메모")
    lines.append("- 엣지 = 저변동 롱 + 고변동(복권주) 숏. 500억서도 net +3.07%(t=2.77, 31번).")
    lines.append("- 고변동 소형주 숏은 대차 회수·숏스퀴즈 위험 — 개별 대차 가능/수수료 확인.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="저변동성 실전 롱숏 포트폴리오 생성 — 표 출력 + CSV 저장"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument("--out-dir", default="data", help="CSV 저장 디렉터리 (기본 data/)")
    parser.add_argument("--no-save", action="store_true", help="CSV 저장 없이 화면 출력만")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asof = (pd.Timestamp(args.asof) if args.asof
            else pd.Timestamp(pd.Timestamp.today().date()))
    start = str((asof - pd.tseries.offsets.BDay(140)).date())
    end = str(asof.date())

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1
    try:
        adj = load_prices(conn, start, end)
        meta = load_stock_master(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 로드 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    portfolio = select_lowvol_portfolio(adj, meta, asof)
    if portfolio is None:
        print(f"# 저변동성 실전 포트폴리오 — {asof.date()}")
        print()
        print("형성 가능한 리밸 시점이 없다 (거래일·이력 부족 또는 유니버스 부족).")
        return 0

    frame = build_frame(portfolio)
    print(format_report(portfolio, frame, str(asof.date())))

    if not args.no_save:
        from pathlib import Path

        out = Path(args.out_dir) / f"lowvol_portfolio_{portfolio.period}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print(f"저장: {out} ({len(frame)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
