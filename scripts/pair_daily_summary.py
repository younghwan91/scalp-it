#!/usr/bin/env python3
"""짝꿍 dry-run 일일 성과 집계 — data/pair_positions.log 를 읽어 그날(또는 --date)의
가상 포지션을 **체결 가정(fill_basis)별로 따로** 요약한다.

설계 메모(회계 결함 수정본)
==========================

* **가정별 그룹핑**: 가드가 알림 하나당 회계 포지션(through)과 그림자(touch)를 독립
  수명으로 굴려 청산 로그를 **가정마다 한 줄씩** 남긴다. 그래서 여기서는 로그의
  ``fill_basis`` 로 그룹핑한다. 예전처럼 ``filled_touch``/``filled_through`` 불리언으로
  거르면 through ⊆ touch 라 두 열의 모집단이 항상 같아져 낙관 시나리오가 측정되지 않는다.
* **미체결은 청산이 아니다**: ``status=="closed"`` 만 승률·손익·MAE/MFE 에 넣고, 미체결은
  건수(체결률)로만 따로 보고한다. 미체결의 mae/mfe 는 갱신된 적 없는 0.0 이라 섞이면
  중앙값이 리스크를 작아 보이게 왜곡한다.
* **세전/세후 구분**: 로그의 ``pnl`` 은 가드 ``cost_rate`` 를 이미 뺀 값이라 설정에 따라
  의미가 흔들린다. 여기서는 ``exit_price/entry-1`` 로 **세전(gross) 수익률을 직접 다시
  계산**하고, 비용 시나리오 3개(34/64/94bp, ``scripts/verify_pair.py`` 와 같은 상수)를
  거기서 차감해 병기한다. 필드명에 gross/net 을 명시해 헷갈리지 않게 한다.
* **재실행 안전**: 결과를 ``data/pair_daily_summary.jsonl`` 에 날짜 키로 upsert 하고,
  사람이 읽는 ``.log`` 도 같은 날짜 줄을 지우고 다시 쓴다(중복 append 금지).
  며칠치 합산은 jsonl 을 읽어 ``by_basis[...]["gross"]["total_krw"]`` 등을 더하면 된다.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

POS_LOG = Path("data/pair_positions.log")
TXT_LOG = Path("data/pair_daily_summary.log")
JSONL = Path("data/pair_daily_summary.jsonl")

# 왕복 비용 시나리오 — scripts/verify_pair.py 의 COST_LO/BASE/HI 와 동일한 상수.
COSTS = {"net34": 0.0034, "net64": 0.0064, "net94": 0.0094}
DEFAULT_NOTIONAL = 100_000.0


def _notional(row: dict[str, Any]) -> float:
    """로그 한 줄에서 명목금액 역산(pnl_krw = pnl × notional). 불가하면 기본값."""
    pnl, krw = row.get("pnl"), row.get("pnl_krw")
    if pnl and krw is not None:
        return abs(krw / pnl)
    return DEFAULT_NOTIONAL


def _gross(row: dict[str, Any]) -> float | None:
    """세전 수익률 — 비용 설정에 흔들리지 않게 청산가/진입가에서 직접 계산."""
    entry, exit_px = row.get("entry"), row.get("exit_price")
    if not entry or exit_px is None:
        return row.get("pnl")   # 구버전 로그 대비(청산가 미기록)
    return exit_px / entry - 1.0


def _scenario(rets: list[float], notionals: list[float], cost: float) -> dict[str, Any]:
    nets = [r - cost for r in rets]
    wins = sum(1 for v in nets if v > 0)
    return {
        "n": len(nets),
        "wins": wins,
        "win_rate_pct": round(wins / len(nets) * 100, 2),
        "mean_pct": round(st.mean(nets) * 100, 4),
        "total_krw": round(sum(v * n for v, n in zip(nets, notionals, strict=True)), 2),
    }


def summarize(rows: list[dict[str, Any]], day: str) -> dict[str, Any]:
    """가정별 요약 dict. 며칠치를 기계적으로 합산할 수 있게 전부 수치로만 담는다."""
    by_basis: dict[str, dict[str, Any]] = {}
    bases = sorted({str(r.get("fill_basis") or "through") for r in rows})
    for basis in bases:
        grp = [r for r in rows if str(r.get("fill_basis") or "through") == basis]
        closed = [r for r in grp if r.get("status") == "closed"]
        unfilled = [r for r in grp if r.get("status") == "unfilled"]
        rets = [g for g in (_gross(r) for r in closed) if g is not None]
        notionals = [_notional(r) for r in closed if _gross(r) is not None]
        mae = [r["mae"] * 100 for r in closed if r.get("mae") is not None]
        mfe = [r["mfe"] * 100 for r in closed if r.get("mfe") is not None]
        entry: dict[str, Any] = {
            "n_alerts": len(grp),
            "n_closed": len(closed),
            "n_unfilled": len(unfilled),
            "fill_rate_pct": round(len(closed) / len(grp) * 100, 2) if grp else None,
            "exit_reasons": dict(Counter(r.get("exit_reason") for r in closed)),
            "median_mae_pct": round(st.median(mae), 4) if mae else None,
            "median_mfe_pct": round(st.median(mfe), 4) if mfe else None,
        }
        if rets:
            entry["gross"] = _scenario(rets, notionals, 0.0)
            entry["net"] = {k: _scenario(rets, notionals, c) for k, c in COSTS.items()}
        by_basis[basis] = entry
    return {
        "date": day,
        "generated_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
        "n_rows": len(rows),
        "cost_scenarios": COSTS,
        "by_basis": by_basis,
    }


def human_line(summary: dict[str, Any]) -> str:
    """사람이 읽는 한 줄. 손익은 전부 세전(gross)/세후(net) 라벨을 붙여 구분한다."""
    parts = [f"[{summary['date']}]"]
    labels = {"through": "through(보수)", "touch": "touch(낙관)"}
    for basis, e in summary["by_basis"].items():
        head = (f"{labels.get(basis, basis)}: 청산{e['n_closed']}건 "
                f"미체결{e['n_unfilled']}건")
        g = e.get("gross")
        if g:
            head += (f" 승률{g['win_rate_pct']:.0f}% 세전{g['mean_pct']:+.3f}%"
                     f"({g['total_krw']:+.0f}원)")
            nets = " ".join(f"{k}{e['net'][k]['mean_pct']:+.3f}%" for k in COSTS)
            head += f" 세후[{nets}]"
        if e["median_mae_pct"] is not None:
            head += (f" MAE중앙{e['median_mae_pct']:+.2f}%"
                     f" MFE중앙{e['median_mfe_pct']:+.2f}%")
        if e["exit_reasons"]:
            head += f" 사유{e['exit_reasons']}"
        parts.append(head)
    return " | ".join(parts)


def _upsert_jsonl(summary: dict[str, Any]) -> None:
    """날짜 키 upsert — 같은 날 두 번 돌려도 줄이 늘지 않는다."""
    rows = []
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date") != summary["date"]:
                rows.append(rec)
    rows.append(summary)
    rows.sort(key=lambda r: str(r.get("date")))
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with JSONL.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def _upsert_text(day: str, line: str) -> None:
    """사람용 .log 도 같은 날짜 줄을 덮어쓴다(예전엔 무조건 append 라 중복됐다)."""
    kept = []
    if TXT_LOG.exists():
        kept = [x for x in TXT_LOG.read_text(encoding="utf-8").splitlines()
                if x.strip() and not x.startswith(f"[{day}]")]
    kept.append(line)
    TXT_LOG.parent.mkdir(parents=True, exist_ok=True)
    TXT_LOG.write_text("\n".join(kept) + "\n", encoding="utf-8")


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else str(date.today())  # noqa: DTZ011
    rows: list[dict[str, Any]] = []
    shadows: list[dict[str, Any]] = []
    if not POS_LOG.exists():
        print(f"[{day}] 포지션 로그 없음")
        return 0
    for line in POS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # close(체결 후 청산) + unfilled(미체결 만료) 둘 다 모집단. 구버전 로그는
        # 미체결도 event="close" 라 status 로 한 번 더 갈라낸다.
        # 그림자(국면 차단 중 "샀으면 어땠나")는 실제 매매가 아니다 — 섞으면
        # 그날 성적이 거짓이 된다. 따로 센다.
        if r.get("blocked"):
            shadows.append(r)
            continue
        if r.get("event") in ("close", "unfilled") and str(r.get("ts", "")).startswith(day):
            rows.append(r)
    same_day = [r for r in shadows
                if r.get("event") in ("close", "unfilled")
                and str(r.get("ts", "")).startswith(day)]
    if same_day:
        pnl = sum(float(r.get("pnl_krw") or 0.0) for r in same_day)
        win = sum(1 for r in same_day if float(r.get("pnl_krw") or 0.0) > 0)
        print(f"[{day}] 안 샀지만 기록 {len(same_day)}건 (국면 차단·실계좌 무관) — "
              f"샀다면 승 {win} · 합계 {pnl:+,.0f}원")
    if not rows:
        print(f"[{day}] 청산 0건")
        return 0

    summary = summarize(rows, day)
    line = human_line(summary)
    print(line)
    _upsert_text(day, line)
    _upsert_jsonl(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
