# ruff: noqa: DTZ001, DTZ011 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""실행기 → krx-quant-core ``PositionBook`` 장부 기록(부가 기능, 결정에 무관).

``book`` 을 안 주면 아무것도 달라지지 않고(기존 test_order_executor 전부가 그 증거),
주면 매수·불타기·분할·전량 청산 체결이 장부와 journal 에 남는다. 장부가 예외를 던져도
실행기는 멀쩡히 돈다 — 장부는 기록이지 판단이 아니다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from krx_quant_core.execution.book import PositionBook
from test_order_executor import FakeClient, _alert, _last

from scalp_it.realtime.order_executor import OrderExecutor

T0 = datetime(2026, 8, 25, 9, 30, 0)


def _t(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def test_book_records_buy_and_full_exit(tmp_path):
    journal = tmp_path / "positions" / "2026-08-25.jsonl"
    book = PositionBook(journal=journal, market_of=lambda c: "KOSDAQ")
    ex = OrderExecutor(FakeClient(), qty=1, book=book)
    ex.on_signal(_alert(ts=T0, entry=10000, tp=10050))
    ex.on_tick("294570", _t(2), 9999)                  # 관통 → 매수 체결
    h = book.position("294570")
    assert h is not None and h.qty == 1 and h.avg_price == 10000
    ex.on_tick("294570", _t(5), 10050)                 # 익절 접수
    assert book.position("294570") is not None         # 접수 ≠ 체결
    ex.on_tick("294570", _t(6), 10051)                 # 매도 체결
    assert _last(ex).status == "exited"
    assert book.position("294570") is None
    assert book.n_round_trips == 1
    rows = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines()]
    assert [(r["side"], r["qty"], r["price"]) for r in rows] == [
        ("buy", 1, 10000), ("sell", 1, 10050)]
    # journal 재생으로 같은 장부가 복원된다
    again = PositionBook.load(journal, market_of=lambda c: "KOSDAQ")
    assert again.n_round_trips == 1 and again.realized_krw == book.realized_krw


def test_book_records_partial_and_remainder():
    book = PositionBook()
    ex = OrderExecutor(FakeClient(), qty=4, partial_exit_ratio=0.5,
                       partial_exit_gain=0.002, book=book)
    ex.on_signal(_alert(ts=T0, entry=10000, tp=10050))
    ex.on_tick("294570", _t(2), 9999)                  # 4주 체결
    ex.on_tick("294570", _t(5), 10025)                 # 2주 분할 접수
    ex.on_tick("294570", _t(6), 10030)                 # 분할 체결
    assert book.position("294570").qty == 2
    ex.on_tick("294570", _t(8), 10050)                 # 잔량 익절 접수
    ex.on_tick("294570", _t(9), 10060)                 # 잔량 체결
    assert book.position("294570") is None
    assert book.n_round_trips == 2


def test_book_records_pyramid_add():
    book = PositionBook()
    ex = OrderExecutor(FakeClient(), qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1, book=book)
    ex.on_signal(_alert(ts=T0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", _t(1), 9999)                  # 체결
    ex.on_tick("294570", _t(5), 10300)                 # 불타기 접수
    assert book.position("294570").qty == 1
    ex.on_tick("294570", _t(6), 10299)                 # 불타기 체결
    h = book.position("294570")
    assert h.qty == 2 and h.avg_price == 10150


class _BrokenBook:
    def __init__(self):
        self.calls = 0

    def apply(self, fill):
        self.calls += 1
        raise RuntimeError("disk full")


def test_book_failure_never_breaks_executor():
    notices: list[str] = []
    plain = OrderExecutor(FakeClient(), qty=1)
    broken = _BrokenBook()
    ex = OrderExecutor(FakeClient(), qty=1, book=broken, notice=notices.append)
    for e in (plain, ex):
        e.on_signal(_alert(ts=T0, entry=10000, tp=10050))
        e.on_tick("294570", _t(2), 9999)
        e.on_tick("294570", _t(5), 10050)
        e.on_tick("294570", _t(6), 10051)
    assert broken.calls == 2
    assert _last(ex).status == _last(plain).status == "exited"
    assert ex.summary() == plain.summary()


def test_no_book_is_default():
    ex = OrderExecutor(FakeClient(), qty=1)
    assert ex._book is None


# --------------------------------------------------------------------- CLI 배선

def _cron_executor(monkeypatch, tmp_path, *, real=True):
    from test_cli_pair_detect_wiring import CRON_FLAGS, _FakeOrderClient

    from scalp_it.cli_pair_detect import build_order_executor, build_parser, resolve_pairs

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args([*CRON_FLAGS, *(["--confirm-real"] if real else [])])
    return build_order_executor(args, resolve_pairs(args), client=_FakeOrderClient())


def test_live_order_executor_gets_dated_journal_book(monkeypatch, tmp_path):
    from datetime import date

    ex = _cron_executor(monkeypatch, tmp_path)
    assert isinstance(ex._book, PositionBook)
    assert ex._book._journal_path == (
        __import__("pathlib").Path("data/positions") / f"{date.today().isoformat()}.jsonl")


def test_existing_journal_is_replayed_and_bad_journal_never_blocks_startup(monkeypatch, tmp_path):
    from datetime import date

    j = tmp_path / "data" / "positions" / f"{date.today().isoformat()}.jsonl"
    j.parent.mkdir(parents=True)
    j.write_text(json.dumps({"ord_no": "B1", "code": "294570", "side": "buy", "qty": 2,
                             "price": 10000, "ts": "2026-08-25T09:30:00"}) + "\n",
                 encoding="utf-8")
    ex = _cron_executor(monkeypatch, tmp_path)
    assert ex._book.position("294570").qty == 2           # 재시작해도 장부가 이어진다
    j.write_text("{broken\n", encoding="utf-8")
    ex = _cron_executor(monkeypatch, tmp_path)
    assert ex is not None and ex._book is None             # 장부만 끄고 실행기는 뜬다


def test_no_live_order_no_executor_no_book(monkeypatch, tmp_path):
    from scalp_it.cli_pair_detect import build_order_executor, build_parser

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--take-profit", "0"])
    assert build_order_executor(args, []) is None
    assert not (tmp_path / "data").exists()


def test_dry_run_executor_gets_no_book(monkeypatch, tmp_path):
    """--live-order 만(dry-run)이면 장부를 열지 않는다 — 모의 체결이 실계좌 journal 처럼 쌓이면 안 된다."""
    ex = _cron_executor(monkeypatch, tmp_path, real=False)
    assert ex is not None and ex._book is None
    assert not (tmp_path / "data" / "positions").exists()
