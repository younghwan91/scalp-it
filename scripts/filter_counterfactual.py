#!/usr/bin/env python3
"""필터배제 신호 반사실 — "안 산 게 맞았나"를 틱으로 되돌려 본다.

라이브 감지기는 방아쇠를 통과한 신호를 세 관문(roundtrip_bp·depth_short·min_sl_ticks)
에서 배제하고 `data/pair_alerts.log` 에 통지 한 줄만 남긴다. 그 배제가 옳았는지는
아무도 안 봤다 — 09-09 저널에서 배제 건이 진입 건의 20배가 넘는데, 문턱(20→30bp)은
"표본을 늘리자"는 감으로 옮겼다. 이 스크립트는 배제된 각 신호를 그 종목 체결 흐름에
얹어 라이브 청산 규칙 근사(재난선 −2% · 타임스탑 180s · 트레일 3%/1.5% · 약손절 −0.5%)
로 되돌려 보고, 잰 왕복충격 구간별로 "샀으면 얼마였나"를 낸다.

같은 규칙을 **통과한** 신호(알림 JSON 레코드, rt_bp 있음)에도 똑같이 돌려 대조군으로
둔다 — 실거래 결과가 아니라 같은 시뮬레이터로 비교해야 필터 효과만 남는다.

한계(정직하게): 대장이탈 청산은 못 흉내 낸다(대장 상태 미기록). 체결은 신호 직후
첫 틱에 진입가로 즉시 체결로 본다(라이브 지연 0~1초 실측). 그러니 여기 숫자는
"필터가 거른 신호의 결이 다른가"를 보는 용도지 손익 예측이 아니다.

읽기만 한다 — `ticks` 만 읽고, 로그 파일만 읽는다. 산출물은
`data/filter_counterfactual_<from>_<to>.md` + stdout.

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/filter_counterfactual.py --from 2026-08-25 --to 2026-09-09
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tick_sanity import _load_code_day, _load_env

ALERTS = "data/pair_alerts.log"
DETECT_LOG = "data/pair_detect.log"
DISASTER = -0.02
HOLD_SECS = 180
TRAIL_ARM = 0.03
TRAIL = 0.015
STOP_LOSS = -0.005
COST_BP = 64.0            # pair_daily_summary 의 net64 와 같은 기준
BUCKETS = [(0, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, float("inf"))]

_NOTICE = re.compile(
    r"\[(\d\d:\d\d:\d\d)\] 짝꿍 필터배제\((\w+)\)\s+\S+\((\d+)\)\s+→\s+\S+\((\d+)\)\s+진입 ([\d,]+)"
)
_RT = re.compile(r"왕복충격 ([\d.]+)bp > ([\d.]+)bp")


@dataclass
class Signal:
    day: str
    hms: str
    leader: str
    follower: str
    entry: float
    reason: str          # roundtrip_bp | depth_short | min_sl_ticks | passed
    rt_bp: float         # inf = depth_short, nan = 못 잼


def _parse_notice(text: str) -> tuple | None:
    m = _NOTICE.search(text)
    if not m:
        return None
    hms, reason, leader, follower, entry = m.groups()
    if reason not in ("roundtrip_bp", "depth_short", "min_sl_ticks"):
        return None
    rt = float("nan")
    mr = _RT.search(text)
    if mr:
        rt = float(mr.group(1))
    elif reason == "depth_short":
        rt = float("inf")
    return hms, reason, leader, follower, float(entry.replace(",", "")), rt


def load_sessions(detect_log: str) -> list[list[tuple]]:
    """배제 통지를 `pair_detect.log` 에서 세션 단위로 읽는다 — 통지엔 날짜가 없다.

    `# 시황 — 날짜` 머리말은 stdout(블록 버퍼)이고 통지는 stderr 라 파일 안에서
    순서가 섞여 날짜 표지로 못 쓴다. 대신 통지 시각이 거꾸로 가는 곳을 세션 경계로
    잡고, 각 세션의 날짜는 :func:`date_sessions` 가 틱 DB 와 대조해 정한다.
    """
    sessions: list[list[tuple]] = []
    cur: list[tuple] = []
    prev = None
    for line in pathlib.Path(detect_log).read_text(encoding="utf-8", errors="replace").splitlines():
        if "필터배제" not in line:
            continue
        p = _parse_notice(line)
        if not p:
            continue
        hms = p[0]
        if prev is not None and hms < prev:
            sessions.append(cur)
            cur = []
        cur.append(p)
        prev = hms
    if cur:
        sessions.append(cur)
    return sessions


def _bdays(day_from: str, day_to: str) -> list[str]:
    import datetime as dt
    d = dt.date.fromisoformat(day_from)
    end = dt.date.fromisoformat(day_to)
    out = []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _has_tick_near(conn, code: str, day: str, hms: str, slack_s: int = 120) -> bool:
    from scalp_it.intraday.store import _placeholder
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        f"select 1 from ticks where code = {ph} and ts >= {ph}::timestamp - interval '{slack_s} seconds'"
        f" and ts <= {ph}::timestamp + interval '{slack_s} seconds' limit 1",
        (code, f"{day} {hms}", f"{day} {hms}"),
    )
    hit = cur.fetchone() is not None
    cur.close()
    return hit


def date_sessions(conn, sessions: list[list[tuple]], day_from: str, day_to: str,
                  cache: dict) -> list[Signal]:
    """세션마다 "그 세션 통지들이 그 시각에 틱을 가진" 영업일을 고른다(날짜는 단조증가).

    같은 날 재시작은 허용(직전 세션과 같은 날짜 가능). 후보 중 적중률이 가장 높은 날,
    동률이면 이른 날. 적중 0 인 세션은 버린다(날짜를 못 정하니 쓰지 않는다).
    """
    days = _bdays(day_from, day_to)
    out: list[Signal] = []
    lo = 0
    for sess in sessions:
        best, best_score = None, 0.0
        for j in range(lo, len(days)):
            day = days[j]
            hits = 0
            for p in sess:
                k = (p[3], day, p[0])
                if k not in cache:
                    cache[k] = _has_tick_near(conn, p[3], day, p[0])
                hits += cache[k]
            score = hits / len(sess)
            if score > best_score:
                best, best_score = j, score
        if best is None or best_score < 0.5:
            continue
        lo = best
        day = days[best]
        for hms, reason, leader, follower, entry, rt in sess:
            out.append(Signal(day, hms, leader, follower, entry, reason, rt))
    return out


def load_passed(alerts: str, day_from: str, day_to: str) -> list[Signal]:
    """통과한 신호(알림 JSON, rt_bp 기록됨) — 대조군."""
    out: list[Signal] = []
    for line in pathlib.Path(alerts).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "ts" not in d or "follower" not in d or d.get("rt_bp") is None:
            continue
        day = d["ts"][:10]
        if day_from <= day <= day_to:
            out.append(Signal(day, d["ts"][11:19], d["leader"], d["follower"],
                              float(d["entry"]), "passed", float(d["rt_bp"])))
    return out


def simulate(ts_sec: np.ndarray, price: np.ndarray, t0: int, entry: float,
             use_stop: bool) -> dict | None:
    """신호 시각 이후 체결로 라이브 청산 근사. 진입은 t0 이후 첫 틱에 entry 로 즉시 체결."""
    i0 = int(np.searchsorted(ts_sec, t0, side="left"))
    if i0 >= ts_sec.shape[0]:
        return None
    t_end = ts_sec[i0] + HOLD_SECS
    peak = entry
    armed = False
    mfe = mae = 0.0
    exit_px, reason = None, "시간청산"
    last = entry
    for i in range(i0, ts_sec.shape[0]):
        if ts_sec[i] > t_end:
            break
        p = price[i]
        if not np.isfinite(p):
            continue
        last = p
        r = p / entry - 1
        mfe, mae = max(mfe, r), min(mae, r)
        if r <= DISASTER:
            exit_px, reason = p, "재난선"
            break
        if use_stop and r <= STOP_LOSS:
            exit_px, reason = p, "손절"
            break
        peak = max(peak, p)
        if not armed and r >= TRAIL_ARM:
            armed = True
        if armed and p <= peak * (1 - TRAIL):
            exit_px, reason = p, "트레일"
            break
    if exit_px is None:
        exit_px = last
    return {"ret": exit_px / entry - 1, "mfe": mfe, "mae": mae, "reason": reason}


def _fmt(v: float, pct: bool = True) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:+.3f}" if pct else f"{v:.1f}"


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:g}~{hi:g}bp" if np.isfinite(hi) else f"{lo:g}bp+"


def summarize(rows: list[dict], key: str) -> dict:
    if not rows:
        return {"n": 0}
    r = np.array([x[key] for x in rows])
    mfe = np.array([x["mfe"] for x in rows])
    mae = np.array([x["mae"] for x in rows])
    bp = np.array([x["rt_bp"] for x in rows])
    fin_bp = bp[np.isfinite(bp)]
    net = r - COST_BP / 1e4
    net_imp = net - np.where(np.isfinite(bp), bp, 0.0) / 1e4
    reasons: dict[str, int] = {}
    for x in rows:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    return {
        "n": len(rows), "win": float((r > 0).mean()),
        "mean": float(r.mean()), "median": float(np.median(r)),
        "net": float(net.mean()), "net_imp": float(net_imp.mean()),
        "mfe": float(np.median(mfe)), "mae": float(np.median(mae)),
        "bp_med": float(np.median(fin_bp)) if fin_bp.size else float("nan"),
        "reasons": reasons,
    }


def table(groups: list[tuple[str, dict]], title: str) -> list[str]:
    out = [f"### {title}", "",
           "| 구간 | n | 승률 | 평균 | 중앙 | net64 | net64+충격 | MFE중앙 | MAE중앙 | 청산사유 |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for name, s in groups:
        if s["n"] == 0:
            out.append(f"| {name} | 0 | | | | | | | | |")
            continue
        rs = " ".join(f"{k}{v}" for k, v in sorted(s["reasons"].items(), key=lambda kv: -kv[1]))
        out.append(
            f"| {name} | {s['n']} | {s['win'] * 100:.0f}% | {_fmt(s['mean'])} | {_fmt(s['median'])}"
            f" | {_fmt(s['net'])} | {_fmt(s['net_imp'])} | {_fmt(s['mfe'])} | {_fmt(s['mae'])} | {rs} |"
        )
    out.append("")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="day_from", required=True)
    ap.add_argument("--to", dest="day_to", required=True)
    ap.add_argument("--alerts", default=ALERTS)
    ap.add_argument("--detect-log", default=DETECT_LOG)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    sessions = load_sessions(args.detect_log)
    signals = (date_sessions(conn, sessions, args.day_from, args.day_to, {})
               + load_passed(args.alerts, args.day_from, args.day_to))
    if not signals:
        print("신호 없음")
        return 1
    cache: dict[tuple[str, str], tuple | None] = {}
    rows: list[dict] = []
    missing = 0
    for s in signals:
        k = (s.follower, s.day)
        if k not in cache:
            cache[k] = _load_code_day(conn, s.follower, s.day)
        data = cache[k]
        if data is None:
            missing += 1
            continue
        ts_sec, hms, price, *_ = data
        idx = np.searchsorted(hms, s.hms, side="left")
        if idx >= hms.shape[0]:
            missing += 1
            continue
        t0 = int(ts_sec[idx])
        for use_stop, tag in ((False, "nostop"), (True, "stop")):
            sim = simulate(ts_sec, price, t0, s.entry, use_stop)
            if sim is None:
                missing += 1
                break
            rows.append({**sim, "variant": tag, "rt_bp": s.rt_bp, "kind": s.reason,
                         "day": s.day, "hms": s.hms, "follower": s.follower, "entry": s.entry})

    n_days = sorted({s.day for s in signals})
    lines = [f"# 필터배제 반사실 — {args.day_from} ~ {args.day_to} ({len(n_days)}일)", "",
             ("> 배제된 신호를 그 종목 체결에 얹어 라이브 청산 근사(재난선 −2%·타임스탑 180s·"
              "트레일 +3%/1.5%·[stop 변형만] 약손절 −0.5%)로 되돌린 값. 대장이탈은 흉내 못 냄. "
              "진입은 신호 직후 첫 틱에 진입가 즉시 체결 가정. **손익 예측이 아니라 필터가 거른 "
              "신호의 결이 다른지 보는 용도.**"), "",
             (f"- 신호 {len(signals)}건 (배제 {sum(s.reason != 'passed' for s in signals)} / "
              f"통과 {sum(s.reason == 'passed' for s in signals)}), 틱 없음·장마감 {missing}건"),
             "- 날짜별: " + ", ".join(f"{d} {sum(s.day == d for s in signals)}건" for d in n_days),
             (f"- net64 = 수익 − {COST_BP:.0f}bp. net64+충격 = 거기서 잰 왕복충격(bp)까지 뺀 값"
              "(1,000만원 시장가 왕복 가정. 1주 실주문엔 과대)."), ""]

    for tag, title in (("nostop", "약손절 없음(대장 굳힘 중 손절 잠금 상태)"),
                       ("stop", "약손절 −0.5% 항상 켬")):
        sub = [r for r in rows if r["variant"] == tag]
        lines.append(f"## {title}")
        lines.append("")
        groups = [("통과(≤한도)", summarize([r for r in sub if r["kind"] == "passed"], "ret"))]
        for lo, hi in BUCKETS:
            g = [r for r in sub if r["kind"] == "roundtrip_bp" and lo < r["rt_bp"] <= hi]
            groups.append((f"배제 {bucket_label(lo, hi)}", summarize(g, "ret")))
        groups.append(("배제 depth_short", summarize([r for r in sub if r["kind"] == "depth_short"], "ret")))
        groups.append(("배제 min_sl_ticks", summarize([r for r in sub if r["kind"] == "min_sl_ticks"], "ret")))
        lines += table(groups, "왕복충격 구간별")
        # 누적: 한도를 X 로 올리면 추가로 들어오는 집합
        cum = []
        for cap in (30, 40, 50, 60, 100):
            g = [r for r in sub if r["kind"] == "roundtrip_bp" and r["rt_bp"] <= cap]
            cum.append((f"한도 {cap}bp 면 추가 진입", summarize(g, "ret")))
        lines += table(cum, "한도를 올렸을 때 추가로 들어오는 배제 건(누적)")

    # 개별 건 — 눈으로 보게
    lines.append("## 배제 건 개별(약손절 없음 변형, 왕복충격 오름차순)")
    lines.append("")
    lines.append("| 날짜 | 시각 | 종목 | 진입가 | 사유 | 충격bp | 수익 | MFE | MAE | 청산 |")
    lines.append("|---|---|---|--:|---|--:|--:|--:|--:|---|")
    ind = sorted((r for r in rows if r["variant"] == "nostop" and r["kind"] != "passed"),
                 key=lambda r: (r["rt_bp"] if np.isfinite(r["rt_bp"]) else 1e9))
    for r in ind:
        bp = "∞" if np.isinf(r["rt_bp"]) else ("—" if not np.isfinite(r["rt_bp"]) else f"{r['rt_bp']:.1f}")
        lines.append(f"| {r['day']} | {r['hms']} | {r['follower']} | {r['entry']:,.0f} | {r['kind']} | {bp}"
                     f" | {_fmt(r['ret'])} | {_fmt(r['mfe'])} | {_fmt(r['mae'])} | {r['reason']} |")
    lines.append("")

    text = "\n".join(lines)
    out = args.out or f"data/filter_counterfactual_{args.day_from.replace('-', '')}_{args.day_to.replace('-', '')}.md"
    pathlib.Path(out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
