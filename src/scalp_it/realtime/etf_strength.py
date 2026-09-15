"""81번 ETF 체결강도 규칙 — 소켓 실시간 그림자(주문 없음).

규칙 정본은 `docs/research/manju/81-etf-strength-scalp-preregistration.md` §3 이고, 판정은
DB 리플레이(`scripts/etf_strength_81.py`)가 한다. 이 모듈은 같은 규칙을 장중 소켓에서 8셀
동시에 돌려 가상 체결을 남긴다 — 채택 당일 실주문으로 뒤집을 준비이자, 리플레이와의
괴리를 보는 장치다. **주문 경로에는 한 줄도 닿지 않는다.**

- 신호: 스프레드 1틱(ETF 호가단위 2,000원 미만 1원·이상 5원) 그리고 체결강도(t) − 체결강도(t−30초) ≥ 문턱.
- 진입: 신호 초 + 1초 이후 첫 갱신의 매도1호가.
- 청산: 진입 뒤 체결강도 최고치 − s 이하가 된 갱신의 매수1호가, 또는 보유 T초.
- 종목·셀당 포지션 1개, 신호 시각 기준 60초 쿨다운. 진입 가능 09:05~15:10.

**리플레이와 알려진 차이**: 청산·진입은 갱신(틱·호가)이 올 때만 판정한다. VI 등으로 체결이 끊기면
시간청산이 재개 후 첫 갱신에서 난다(09-15 067290 VI: T=60 셀이 294초 보유). 리플레이는 T초 시점의
전방채움 호가로 청산한다 — 실제로는 그 호가에 못 판다. 판정은 리플레이가 하므로 이 차이는 패리티
리포트에서 드러낸다.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

__all__ = ["CELLS", "Cell", "StrengthShadow", "etf_tick", "load_thresholds"]

COST = 0.00015 * 2
COOLDOWN = 60
LOOKBACK = 30
WINDOW = (time(9, 5), time(15, 10))


def etf_tick(price: float) -> float:
    return 1.0 if price < 2000 else 5.0


@dataclass(frozen=True)
class Cell:
    q: float
    sdrop: float
    T: int

    @property
    def key(self) -> str:
        return f"q{self.q}_s{self.sdrop:g}_T{self.T}"


@dataclass
class _Pos:
    signal_sec: int
    entry: float | None = None
    entry_sec: int | None = None
    entry_ts: datetime | None = None
    peak: float | None = None


class StrengthShadow:
    def __init__(self, thresholds: dict[tuple[str, float], float], *, cells: Iterable[Cell],
                 sink: Callable[[dict[str, Any]], None]) -> None:
        self.thresholds = dict(thresholds)
        self.cells = tuple(cells)
        self.codes = {c for c, _ in self.thresholds}
        self._sink = sink
        self._book: dict[str, tuple[float, float]] = {}
        self._hist: dict[str, tuple[list[int], list[float]]] = {}
        self._pos: dict[tuple[str, str], _Pos] = {}
        self._free: dict[tuple[str, str], int] = {}

    def open_count(self) -> int:
        return sum(1 for p in self._pos.values() if p.entry is not None)

    def on_quote(self, code: str, ts: datetime, bid1: float | None, ask1: float | None) -> None:
        if code in self.codes and bid1 and ask1:
            self._book[code] = (float(bid1), float(ask1))
            self._step(code, ts, None)

    def on_tick(self, code: str, ts: datetime, price: float, *, strength: float | None,
                best_bid: float | None, best_ask: float | None) -> None:
        if code not in self.codes:
            return
        if best_bid and best_ask:
            self._book[code] = (float(best_bid), float(best_ask))
        self._step(code, ts, strength)

    # ------------------------------------------------------------------
    def _strength_ago(self, code: str, sec: int) -> float | None:
        secs, vals = self._hist.get(code, ([], []))
        i = bisect_right(secs, sec - LOOKBACK)
        return vals[i - 1] if i else None

    def _step(self, code: str, ts: datetime, strength: float | None) -> None:
        sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        book = self._book.get(code)
        if strength is not None:
            secs, vals = self._hist.setdefault(code, ([], []))
            if secs and secs[-1] == sec:
                vals[-1] = strength
            else:
                secs.append(sec)
                vals.append(strength)
        for cell in self.cells:
            key = (code, cell.key)
            pos = self._pos.get(key)
            if pos is not None and book is not None:
                self._manage(code, cell, key, pos, ts, sec, book, strength)
                continue
            if pos is not None or strength is None or book is None:
                continue
            if not (WINDOW[0] <= ts.time() < WINDOW[1]) or sec < self._free.get(key, -1):
                continue
            thr = self.thresholds.get((code, cell.q))
            ago = self._strength_ago(code, sec)
            bid, ask = book
            if thr is None or ago is None or ask - bid > etf_tick(ask) * 1.01:
                continue
            if strength - ago >= thr:
                self._pos[key] = _Pos(signal_sec=sec)
                self._free[key] = sec + COOLDOWN

    def _manage(self, code: str, cell: Cell, key: tuple[str, str], pos: _Pos, ts: datetime,
                sec: int, book: tuple[float, float], strength: float | None) -> None:
        bid, ask = book
        if pos.entry is None:
            if sec >= pos.signal_sec + 1:
                pos.entry, pos.entry_sec, pos.entry_ts = ask, sec, ts
                last = self._hist[code][1][-1] if self._hist.get(code) else strength
                pos.peak = last
            return
        reason = None
        if strength is not None:
            pos.peak = strength if pos.peak is None else max(pos.peak, strength)
            if strength <= pos.peak - cell.sdrop:
                reason = "강도꺾임"
        if reason is None and sec >= pos.entry_sec + cell.T:
            reason = "시간"
        if reason is None:
            return
        self._sink({
            "code": code, "cell": cell.key, "signal_sec": pos.signal_sec,
            "entry_ts": pos.entry_ts.isoformat() if pos.entry_ts else None, "exit_ts": ts.isoformat(),
            "entry": pos.entry, "exit": bid, "reason": reason,
            "ret": bid / pos.entry - 1 - COST,
        })
        del self._pos[key]

    def flush(self, ts: datetime) -> None:
        """장 마감 — 들고 있는 그림자를 마지막 매수1호가로 닫는다."""
        for (code, ck), pos in list(self._pos.items()):
            book = self._book.get(code)
            if pos.entry is not None and book is not None:
                self._sink({"code": code, "cell": ck, "signal_sec": pos.signal_sec,
                            "entry_ts": pos.entry_ts.isoformat() if pos.entry_ts else None,
                            "exit_ts": ts.isoformat(), "entry": pos.entry, "exit": book[0],
                            "reason": "마감", "ret": book[0] / pos.entry - 1 - COST})
            del self._pos[(code, ck)]


CELLS = tuple(Cell(q, s, T) for q in (0.95, 0.99) for s in (3.0, 10.0) for T in (60, 300))


def load_thresholds(conn: Any, codes: Iterable[str], day: str, *, cache: str = "data/etf81",
                    root: str | None = None) -> dict[tuple[str, float], float]:
    """직전 최대 5거래일 dstr30 분위수 — 리플레이 스크립트의 가공을 **그대로** 불러 쓴다.

    같은 규칙을 두 번 구현하면 패리티가 깨지는 순간을 못 본다. 그래서 초 격자 가공은
    `scripts/etf_strength_81.py::build_day`(→ 80번 `build_code`) 하나만 쓴다.
    """
    import importlib.util
    from pathlib import Path

    import pandas as pd

    base = Path(root) if root else Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("etf_strength_81", base / "scripts" / "etf_strength_81.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    codes = tuple(codes)
    cur = conn.cursor()
    cur.execute("select distinct ts::date from ticks where code = any(%s) and ts < %s::date "
                "and ts >= %s::date - 14 order by 1", (list(codes), day, day))
    prev_days = [r[0].isoformat() for r in cur.fetchall()][-5:]
    frames: dict[str, list] = {}
    for d in prev_days:
        data = mod.build_day(conn, d, codes, Path(cache))
        for code, (feat, _p) in (data or {}).items():
            frames.setdefault(code, []).append(feat)
    out: dict[tuple[str, float], float] = {}
    for code, fs in frames.items():
        x = pd.concat(fs).dstr30.dropna()
        if len(x):
            for q in {c.q for c in CELLS}:
                out[(code, q)] = float(x.quantile(q))
    return out

