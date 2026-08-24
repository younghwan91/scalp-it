"""`scalp-pead-portfolio` — 오늘 실제로 매매할 PEAD 롱숏 포트폴리오.

24번 강건성 통과(풀 롱숏, 실전 유니버스 50억) 후 배포 산출물. 최신 avail 분기
기준으로 분기내 데실을 매겨 롱(Q10)·숏(Q1) 을 확정하고, **종목코드·이름·시장·
SUE·데실·진입기준가(avail 익일 시가)·비중(동일가중 %)** 표로 출력한 뒤
``data/pead_portfolio_YYYYQ.csv`` 로 저장한다.

포트폴리오 선정 로직은 전진 기록 CLI(:mod:`scalp_it.cli_pead_watch`)와 **완전히
같은 함수**(``select_portfolio``)를 쓴다 — 배포 스펙이 한 곳에서만 정의되도록.
여기에 롱 다리의 악재성 공시 리스크 플래그(:mod:`scalp_it.pead_risk`)를 덧댄다.
DB 는 읽기만 하고, 쓰는 것은 ``data/`` CSV 뿐이다.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import pandas as pd

from . import pead_risk
from .adapters.db import (
    connect,
    load_daily_bars_raw,
    load_earnings,
    load_prices,
    load_stock_master,
)
from .cli_pead_watch import (
    DEPLOY_LIQ,
    HOLD_DAYS,
    RULE_VERSION,
    Portfolio,
    select_portfolio,
)


def build_frame(portfolio: Portfolio, risk: pead_risk.RiskScan) -> pd.DataFrame:
    """롱숏을 하나의 매매 지시 프레임으로. 비중=동일가중, 롱에 리스크 플래그."""
    n_long = len(portfolio.longs)
    n_short = len(portfolio.shorts)
    rows = []
    for side, frame, n in (("롱", portfolio.longs, n_long), ("숏", portfolio.shorts, n_short)):
        weight = 100.0 / n if n else 0.0
        for x in frame.itertuples():
            code = str(x.code)
            flags = risk.hits.get(code, []) if side == "롱" else []
            flag_txt = ""
            if flags:
                top = flags[0]
                flag_txt = f"악재:{top.report_nm[:20]}({top.rcept_dt})"
            rows.append(
                {
                    "side": side,
                    "code": code,
                    "name": x.name,
                    "market": "" if pd.isna(x.market) else x.market,
                    "decile": int(x.decile),
                    "sue_raw": round(float(x.sue_raw), 4),
                    "entry_date": str(pd.Timestamp(x.entry_date).date()),
                    "entry_open": round(float(x.entry_open), 1),
                    "exit_date_planned": str(pd.Timestamp(x.exit_date_planned).date()),
                    "weight_pct": round(weight, 3),
                    "liq_eok": round(float(x.liq) / 100, 0),
                    "risk_flag": flag_txt,
                }
            )
    return pd.DataFrame(rows)


def format_report(
    portfolio: Portfolio, frame: pd.DataFrame, risk: pead_risk.RiskScan, as_of: str
) -> str:
    lines = [
        (f"# PEAD 실전 포트폴리오 — 분기 {portfolio.period} "
         f"(avail {portfolio.avail_date.date()}, 기준일 {as_of})"),
        "",
        (f"- 유니버스: 진입 직전 20거래일 평균 거래대금 >= {DEPLOY_LIQ / 100:.0f}억, "
         "스팩·진입일 상한가 제외."),
        (f"- **롱 = Q10({len(portfolio.longs)}종목), 숏 = Q1({len(portfolio.shorts)}종목)**, "
         "각 동일가중(시장중립: 롱 총액 = 숏 총액)."),
        (f"- 진입: avail 익일 시가. 보유: +{HOLD_DAYS}거래일(분기 리밸). "
         "비용: 왕복 68bp + 숏차입."),
        f"- 규칙 버전 {RULE_VERSION}. 백테스트 기대 분기 net +4.47% (t=6.15).",
        "",
    ]

    def dump(side: str, title: str) -> None:
        sub = frame[frame["side"] == side]
        lines.append(f"## {title} ({len(sub)}종목, 각 {sub['weight_pct'].iloc[0]:.2f}%)")
        lines.append("")
        hdr = "| # | 코드 | 이름 | 시장 | 데실 | SUE | 기준가 | 비중% | 20일대금(억) |"
        if side == "롱":
            hdr += " 리스크 |"
        lines.append(hdr)
        sep = "|---|---|---|---|---:|---:|---:|---:|---:|"
        if side == "롱":
            sep += "---|"
        lines.append(sep)
        for i, x in enumerate(sub.itertuples(), 1):
            row = (
                f"| {i} | {x.code} | {x.name} | {x.market} | {x.decile} | "
                f"{x.sue_raw:+.2f} | {x.entry_open:,.0f} | {x.weight_pct:.2f} | "
                f"{x.liq_eok:.0f} |"
            )
            if side == "롱":
                row += f" {x.risk_flag or '-'} |"
            lines.append(row)
        lines.append("")

    dump("롱", "롱 (Q10) — 매수")
    dump("숏", "숏 (Q1) — 대차 매도")

    lines.append("## 리스크 오버레이 (롱 다리)")
    lines.append("")
    lines.append(f"- {pead_risk.freshness_note(risk, as_of)}")
    if risk.flagged_codes:
        lines.append(f"- **악재 플래그 {len(risk.flagged_codes)}종목** — 롱 진입 재검토:")
        for code in sorted(risk.flagged_codes):
            hit = risk.hits[code][0]
            nm = frame.loc[frame["code"] == code, "name"]
            nm = nm.iloc[0] if len(nm) else ""
            lines.append(f"  - {code} {nm}: {hit.report_nm[:40]} ({hit.rcept_dt})")
    else:
        lines.append("- 악재 플래그 없음 (위 커버리지 한계 감안).")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PEAD 실전 롱숏 포트폴리오 생성 — 표 출력 + CSV 저장"
    )
    parser.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD. 생략 시 오늘")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument(
        "--out-dir", default="data", help="CSV 저장 디렉터리 (기본 data/)"
    )
    parser.add_argument(
        "--dart-db", default=pead_risk.DEFAULT_DB, help="DART 공시 SQLite 경로"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="CSV 저장 없이 화면 출력만"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp(pd.Timestamp.today().date())
    )
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

    spac = set(
        meta.loc[meta["name"].str.contains("스팩|기업인수목적", na=False), "code"]
    )
    portfolio = select_portfolio(earn, adj, raw, meta, spac, asof)
    if portfolio is None:
        print(f"# PEAD 실전 포트폴리오 — {asof.date()}")
        print()
        print("형성 가능한 avail 분기가 없다 (진입가 미형성 또는 유니버스 부족).")
        return 0

    long_codes = [str(c) for c in portfolio.longs["code"]]
    # 최신 악재만 의미 있다 — 진입 180일 전부터. 오래된 이력은 현재 리스크가 아니다.
    since = (portfolio.avail_date - pd.Timedelta(days=180)).strftime("%Y%m%d")
    risk = pead_risk.scan_longs(long_codes, args.dart_db, since=since)
    frame = build_frame(portfolio, risk)

    print(format_report(portfolio, frame, risk, str(asof.date())))

    if not args.no_save:
        from pathlib import Path

        out = Path(args.out_dir) / f"pead_portfolio_{portfolio.period}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print(f"저장: {out} ({len(frame)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
