"""만쥬식 매매일지 — 로그를 사람이 복기할 수 있는 글로 바꾼다(문서 44 §10-2).

만쥬는 일지를 실력 유지의 **필수 요건**으로 말한다:

> "매매일지와 복기가 중단되면 실력 유지가 어렵고 언제든 무너질 수 있다" [02:47]
> **진입 이유** 명시 — 예: "대장주 상한가 도달 시 2·3등주 짝꿍매매 진입" [01:30]
> **손절/청산 사유** 명시 — 예: "프로그램 매수를 보고 반등을 기대했으나 예상과 다른
> 하락으로 비중 축소 후 전량 손절" [02:08]
> **주요 매매 종목 중심** — 사소한 매매를 다 적기보다 큰 손익이 났거나 복기 가치가
> 높은 핵심 종목 위주로 [00:26]

우리는 `data/pair_alerts.log`·`pair_positions.log` 에 전부 기록하고 있었지만, 그건
**기계가 읽는 형식**이라 사람이 복기할 수 없다. 숫자 스무 개가 나열된 JSON 을 보고
"왜 샀더라"를 떠올릴 수는 없다. 그 간극이 이 모듈이 메우는 자리다.

만쥬가 고수와 하수를 가른 기준도 같은 것이었다(§9-1) — 고수는 "왜 샀는가"에 시장
상황·핵심 이슈·수급 배경·매도 타이밍까지 시나리오로 답하고, 하수는 "그냥 갈 것
같아서"라고 답한다. 일지는 그 답을 강제로 남기게 하는 장치다.

**국면을 반드시 같이 적는다.** 만쥬 기준 쉬는 날(거래대금 5조 미만)에 나온 결과를
활황장 결과와 섞어 읽으면 판단이 무너진다.

DB·시계를 모른다 — 로그 레코드를 받아 글을 만들 뿐이다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "JournalEntry",
    "build_entries",
    "render_journal",
    "why_entered",
    "why_exited",
]

#: 청산 사유별 복기 문장. 만쥬 일지가 "무엇을 기대했고 무엇이 어긋났나"를 적는 형식이라
#: 사유 코드가 아니라 그 판단을 문장으로 남긴다.
_EXIT_STORY = {
    "대장이탈": "대장이 상한가를 지키지 못해 **매수 이유가 사라졌다** → 손익 무관 전량 청산",
    "익절": "후속주 탄력이 목표에 닿아 수익 실현",
    "손절": "진입 직후 역행해 약손절선에서 잘라냄",
    "재난선": "재난 방지선까지 밀려 즉시 탈출",
    "시간청산": "보유 시간 안에 시세 분출이 나오지 않아 미련 없이 정리(만쥬 타임아웃 룰)",
    "미체결": "지정가에 안 붙어 진입 자체가 성립하지 않음",
    "미체결(대장이탈)": "체결 전에 대장이 무너져 살 이유가 사라짐 → 주문 소멸",
    "마감청산": "장 마감으로 정리(오버나이트 금지)",
    "미체결취소": "제한 시간 안에 안 붙어 주문 취소",
    "마감미체결취소": "장 마감까지 안 붙어 주문 취소",
    "킬스위치취소": "킬스위치 발동으로 대기 주문 취소",
}

#: 방아쇠 종류 → 사람 말. 무엇을 보고 들어갔는지가 복기의 출발점이다.
_TRIGGER_STORY = {
    "상한가터치": "대장이 상한가에 닿는 순간",
    "상한가근접": "대장이 상한가로 치고 들어가는 구간(잠김 직전)",
    "상한가잠김": "대장이 상한가를 굳힌 뒤",
}


@dataclass(frozen=True)
class JournalEntry:
    """일지 한 줄. 사람이 읽는 문장과 기계가 세는 숫자를 같이 담는다."""

    ts: str
    leader: str
    leader_name: str
    follower: str
    follower_name: str
    why_entry: str
    why_exit: str
    exit_reason: str
    entry: float | None = None
    exit_price: float | None = None
    pnl: float | None = None
    pnl_krw: float | None = None
    mae: float | None = None
    mfe: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:+.2f}%"


def why_entered(alert: dict[str, Any]) -> str:
    """진입 이유 한 문장 — 만쥬 일지 형식("대장 상한가 도달 시 2·3등주 진입")."""
    lead = alert.get("leader_name") or alert.get("leader") or "대장"
    fol = alert.get("follower_name") or alert.get("follower") or "후속주"
    trig = alert.get("trigger")
    story = _TRIGGER_STORY.get(str(trig), "") if trig else ""
    if story:
        # 방아쇠 이름을 그대로 남긴다 — 터치였는지 근접이었는지가 복기의 출발점이고,
        # 며칠치를 grep 으로 세려면 코드 문자열이 문장 안에 있어야 한다.
        head = f"{story}({trig})"
    else:
        # 방아쇠가 기록되지 않은 옛 알림 — 서지(급등) 방아쇠였다.
        surge = alert.get("leader_surge")
        head = f"대장 서지 {_pct(surge)}" if surge is not None else "대장 급등"
    bits = [f"{head} — 대장 {lead}, 같은 섹터 후속주 {fol} 매수"]
    sec = alert.get("sector")
    rank = alert.get("fol_rank")
    if sec or rank:
        where = f"섹터 {sec}" if sec else "같은 섹터"
        order = f" 등락률 {rank}위" if rank else ""
        bits.append(f"({where}{order})")
    ratio = alert.get("buy_ratio")
    if isinstance(ratio, (int, float)) and ratio >= 0:
        bits.append(f"· 후속 매수체결비율 {ratio:.0%}")
    lock = alert.get("lock_krw")
    if isinstance(lock, (int, float)) and lock > 0:
        bits.append(f"· 대장 상한가 매수벽 {lock / 1e8:.1f}억")
    return " ".join(bits)


def why_exited(close: dict[str, Any]) -> str:
    """청산 사유 한 문장 — 무엇을 기대했고 무엇이 어긋났나."""
    reason = str(close.get("exit_reason") or "").strip()
    story = _EXIT_STORY.get(reason)
    if story is None:
        story = f"{reason or '사유 미기록'} 으로 청산" if reason else "사유 미기록"
    mae, mfe = close.get("mae"), close.get("mfe")
    if mae is not None and mfe is not None:
        story += f" · 최대순행 {_pct(mfe)} / 최대역행 {_pct(mae)}"
    return story


def _notice_entry(text: str) -> JournalEntry | None:
    """배제 통지 한 줄을 일지 항목으로. **왜 안 샀는지도 복기 대상이다.**"""
    m = re.search(r"필터배제\(([^)]+)\)\s*(\S+)\((\d+)\)\s*→\s*(\S+)\((\d+)\)", text)
    if not m:
        return None
    kind, lead_name, lead, fol_name, fol = m.groups()
    ts_m = re.search(r"\[(\d{2}:\d{2}:\d{2})]", text)
    return JournalEntry(
        ts=ts_m.group(1) if ts_m else "",
        leader=lead, leader_name=lead_name, follower=fol, follower_name=fol_name,
        why_entry=f"방아쇠는 통과했으나 {kind} 필터에 걸려 진입하지 않음",
        why_exit="진입하지 않았다 — 걸린 이유가 옳았는지 나중에 확인할 것",
        exit_reason="배제",
    )


def build_entries(alerts: Iterable[dict[str, Any]],
                  closes: Iterable[dict[str, Any]],
                  notices: Iterable[dict[str, Any]] | None = None) -> list[JournalEntry]:
    """알림·청산·배제 로그를 일지 항목으로 엮는다.

    알림과 청산은 ``alert_ts`` + 종목으로 맞춘다. 아직 안 닫힌 진입도 남긴다 —
    장중에도 읽는 문서라서다.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for c in closes:
        if c.get("event") not in (None, "close", "unfilled"):
            continue
        key = (str(c.get("alert_ts") or c.get("ts") or ""), str(c.get("follower") or ""))
        by_key[key] = c

    out: list[JournalEntry] = []
    for a in alerts:
        ts = str(a.get("ts") or "")
        fol = str(a.get("follower") or "")
        c = by_key.get((ts, fol), {})
        out.append(JournalEntry(
            ts=ts,
            leader=str(a.get("leader") or ""),
            leader_name=str(a.get("leader_name") or ""),
            follower=fol,
            follower_name=str(a.get("follower_name") or ""),
            why_entry=why_entered(a),
            why_exit=why_exited(c) if c else "아직 청산되지 않음",
            exit_reason=str(c.get("exit_reason") or "미청산") if c else "미청산",
            entry=a.get("entry"),
            exit_price=c.get("exit_price"),
            pnl=c.get("pnl"), pnl_krw=c.get("pnl_krw"),
            mae=c.get("mae"), mfe=c.get("mfe"),
        ))

    for n in notices or ():
        text = n.get("notice") if isinstance(n, dict) else None
        if not text:
            continue
        e = _notice_entry(str(text))
        if e is not None:
            out.append(e)
    return out


def render_journal(entries: Sequence[JournalEntry], *, day: str,
                   regime: str = "", note: str = "") -> str:
    """사람이 읽는 일지. 만쥬가 말한 "주요 종목 중심"이라 손익 큰 순으로 세운다."""
    lines = [f"# 매매일지 {day}", ""]
    if regime:
        # 국면을 안 적으면 나중에 결과 해석이 무너진다 — 쉬라고 한 날의 성적을
        # 활황장 성적과 섞어 읽게 된다.
        lines += [f"**시황**: {regime}", ""]
    if note:
        lines += [note, ""]

    traded = [e for e in entries if e.exit_reason != "배제"]
    skipped = [e for e in entries if e.exit_reason == "배제"]

    if not traded:
        lines += [("진입 **0건**. 만쥬 방아쇠(대장 상한가)는 하루 2~3건짜리 희소 "
                   "이벤트라 신호가 없는 날은 정상이다."), ""]
    else:
        total = sum(e.pnl_krw or 0.0 for e in traded)
        wins = sum(1 for e in traded if (e.pnl_krw or 0.0) > 0)
        lines += [f"진입 **{len(traded)}건** · 승 {wins} · 누적 {total:+,.0f}원", ""]
        # 복기 가치 순 = 손익 절대값 순(만쥬 §10-2 "큰 손익이 난 종목 위주로").
        for e in sorted(traded, key=lambda x: -abs(x.pnl_krw or 0.0)):
            pnl = f"{e.pnl_krw:+,.0f}원" if e.pnl_krw is not None else "—"
            lines += [
                f"## {e.ts}  {e.follower_name or e.follower}  {pnl}",
                f"- **왜 샀나** — {e.why_entry}",
                f"- **왜 나왔나** — {e.why_exit}",
                "",
            ]

    if skipped:
        lines += [f"## 배제된 신호 {len(skipped)}건", ""]
        for e in skipped:
            lines += [f"- {e.ts} {e.follower_name or e.follower} — {e.why_entry}"]
        lines += [""]
    return "\n".join(lines)
