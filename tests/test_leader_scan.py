"""장중 시장 전체 대장 스캔(만쥬 §2 대장 낙점) 단위테스트.

이 모듈이 있는 이유는 실측 한 줄이다:

    최근 7거래일 상한가 52건 중 우리 유니버스(44종목) 안에 있던 것 **1건 = 2%**

전날 거래대금으로 45종목을 미리 고르면 오늘 상한가로 가는 종목과 거의 안 겹친다.
방아쇠를 아무리 정확히 맞춰도 대장을 안 보고 있으면 신호가 안 뜬다.
"""

from scalp_it.realtime.leader_scan import (
    ScanRow,
    leader_candidates,
    parse_scan_rows,
    rank_body,
    sector_followers,
)

# 실제 ka10027 응답에서 따온 형태(2026-08-27 코스닥). 종목코드에 거래소 접미사가 붙고
# 숫자는 부호 붙은 문자열로 온다 — 둘 다 파싱에서 걸리면 스캔이 통째로 죽는다.
RAW = [
    {"stk_cd": "043200_AL", "stk_nm": "파루", "cur_prc": "+1331", "flu_rt": "+29.98",
     "now_trde_qty": "4193984", "sel_req": "0", "buy_req": "130290", "cntr_str": "150.25"},
    {"stk_cd": "457600_AL", "stk_nm": "벡트", "cur_prc": "+5010", "flu_rt": "+27.64",
     "now_trde_qty": "391802", "sel_req": "77038", "buy_req": "3534", "cntr_str": "98.10"},
    {"stk_cd": "111111_AL", "stk_nm": "보통주", "cur_prc": "+1100", "flu_rt": "+10.00",
     "now_trde_qty": "100000", "sel_req": "5000", "buy_req": "4000", "cntr_str": "80.00"},
]


def test_parse_strips_exchange_suffix_and_signs():
    """``043200_AL`` · ``+29.98`` 을 못 읽으면 스캔 전체가 조용히 빈다."""
    rows = parse_scan_rows(RAW)
    assert [r.code for r in rows] == ["043200", "457600", "111111"]
    assert rows[0].flu_rt == 29.98
    assert rows[0].price == 1331
    assert rows[0].buy_qty == 130290


def test_parse_skips_rows_without_code():
    assert parse_scan_rows([{"stk_nm": "이름만"}]) == []


def test_locked_reads_the_limit_up_wall():
    """매도 잔량 0 + 매수 잔량 = 상한가 잠김. 만쥬가 창에서 보는 그 값이다."""
    rows = parse_scan_rows(RAW)
    assert rows[0].locked is True       # 파루 — 매도 0, 매수 13만
    assert rows[1].locked is False      # 벡트 — 매도 7.7만


def test_leader_candidates_take_the_twenty_percent_movers():
    """만쥬: (비공개 회신 — 공개본에서 뺐다)
    got = leader_candidates(parse_scan_rows(RAW), min_flu_rt=20.0)
    assert [r.code for r in got] == ["043200", "457600"]
    assert got[0].flu_rt > got[1].flu_rt, "등락률 내림차순이어야 한다"


def test_min_buy_qty_only_binds_on_locked_names():
    """안 잠긴 종목은 상한가 호가가 없어 잔량을 잴 수 없다 — 조건을 걸면 안 된다."""
    got = leader_candidates(parse_scan_rows(RAW), min_flu_rt=20.0, min_buy_qty=100_000)
    assert [r.code for r in got] == ["043200", "457600"], "벡트는 안 잠겨 통과해야 한다"
    got = leader_candidates(parse_scan_rows(RAW), min_flu_rt=20.0, min_buy_qty=200_000)
    assert [r.code for r in got] == ["457600"], "파루는 벽이 얇아 빠져야 한다"


# ----- 짝꿍 선정 --------------------------------------------------------------

def _rows(*specs):
    return [ScanRow(code=c, name=c, price=1000, flu_rt=f, sell_qty=s, buy_qty=b,
                    cntr_str=0.0, volume=0.0) for c, f, s, b in specs]


SECTORS = {"L": "전기/전자", "A": "전기/전자", "B": "전기/전자",
           "C": "전기/전자", "X": "화학"}


def test_followers_are_the_second_and_third_in_the_same_sector():
    rows = _rows(("L", 29.9, 0, 5e5), ("A", 18.0, 100, 100), ("B", 12.0, 100, 100),
                 ("C", 5.0, 100, 100), ("X", 25.0, 100, 100))
    assert sector_followers(rows, "L", SECTORS, rank_min=2, rank_max=3) == ["A", "B"]


def test_followers_exclude_other_sectors():
    rows = _rows(("L", 29.9, 0, 5e5), ("X", 25.0, 100, 100))
    assert sector_followers(rows, "L", SECTORS) == []


def test_followers_exclude_names_already_locked():
    """상한가에 붙어 있으면 살 수가 없다 — 만쥬가 사는 건 아직 안 간 2·3등주다."""
    rows = _rows(("L", 29.9, 0, 5e5), ("A", 29.9, 0, 4e5), ("B", 12.0, 100, 100))
    assert sector_followers(rows, "L", SECTORS) == ["B"], "잠긴 A 는 빠지고 B 가 남는다"


def test_followers_rank_includes_the_leader():
    """순위는 대장을 포함해 매긴다 — 감지기 `_dynamic_pairs` 와 같은 규칙이라야 한다."""
    rows = _rows(("L", 29.9, 0, 5e5), ("A", 18.0, 100, 100), ("B", 12.0, 100, 100))
    # 대장이 1위이므로 2~3위는 A·B. 대장을 빼고 매겼다면 A 가 1위라 rank_min=2 에 걸려
    # A 가 사라졌을 것이다.
    assert sector_followers(rows, "L", SECTORS, rank_min=2, rank_max=3) == ["A", "B"]


def test_unknown_sector_yields_nothing():
    assert sector_followers(_rows(("Z", 29.9, 0, 1)), "Z", SECTORS) == []


def test_rank_body_includes_limit_up_names():
    """이미 잠긴 대장도 짝꿍매매의 방아쇠다 — 빼면 안 된다."""
    assert rank_body()["updown_incls"] == "1"
    assert rank_body()["mrkt_tp"] == "101"
    assert rank_body("000")["mrkt_tp"] == "000"


# ----- 자리 회전: 무엇을 내려도 되는가 -----------------------------------------

class _Pos:
    def __init__(self, leader, follower):
        self.leader, self.follower = leader, follower


class _Guard:
    def __init__(self, *pairs):
        self.open_positions = [_Pos(l, f) for l, f in pairs]


def test_open_positions_are_never_dropped():
    """구독을 끊으면 틱이 안 와 청산 판정이 멈춘다 — 대장이탈도 손절도 못 하고 물린다."""
    from scalp_it.cli_pair_detect import _stale_scan_codes

    hot = {"LEAD": 0.0, "FOL": 0.0, "COLD": 0.0}
    got = _stale_scan_codes(hot, base=set(), guard=_Guard(("LEAD", "FOL")),
                            keep_after=100.0)
    assert got == ["COLD"], "포지션이 걸린 대장·후속은 남아야 한다"


def test_preopen_universe_is_never_dropped():
    """장전 유니버스는 스캐너가 붙인 게 아니라 내릴 권한이 없다."""
    from scalp_it.cli_pair_detect import _stale_scan_codes

    hot = {"BASE": 0.0, "SCAN": 0.0}
    assert _stale_scan_codes(hot, base={"BASE"}, guard=None, keep_after=100.0) == ["SCAN"]


def test_recently_hot_names_stay():
    """아직 상위에 보이는 종목을 내리면 곧바로 다시 붙이느라 자리만 낭비한다."""
    from scalp_it.cli_pair_detect import _stale_scan_codes

    hot = {"WARM": 150.0, "COLD": 10.0}
    assert _stale_scan_codes(hot, base=set(), guard=None, keep_after=100.0) == ["COLD"]


def test_nothing_to_drop_is_empty_not_error():
    from scalp_it.cli_pair_detect import _stale_scan_codes

    assert _stale_scan_codes({}, base=set(), guard=None, keep_after=100.0) == []


# ----- 실전 테마가 있으면 27종 섹터보다 먼저 쓴다 ------------------------------

def _det(themes=None, sectors=None, universe=("L", "A", "B", "X")):
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    return PairDetector([], DetectorConfig(),
                        sectors=sectors or {c: "전기/전자" for c in universe},
                        universe=universe, themes=themes)


def test_peers_fall_back_to_sector_without_themes():
    """테마 스냅샷은 전 종목의 25% 만 덮는다 — 없으면 섹터로 떨어져야 한다."""
    d = _det()
    assert set(d.peers("L")) == {"L", "A", "B", "X"}


def test_peers_prefer_the_real_theme_over_the_27_sector():
    """만쥬: "27종 표준 산업분류로는 실전 테마를 절대 잡아낼 수 없습니다"."""
    d = _det(themes={"L": ["T1"], "A": ["T1"], "B": ["T2"], "X": ["T2"]})
    assert set(d.peers("L")) == {"A"}, "같은 테마만 무리로 본다"


def test_peers_use_sector_when_the_leader_has_no_theme():
    """대장이 테마에 없으면 섹터로 — 섹터보다 나쁠 일은 없어야 한다."""
    d = _det(themes={"A": ["T1"], "B": ["T1"]})
    assert set(d.peers("L")) == {"L", "A", "B", "X"}


def test_peers_use_sector_when_the_theme_has_nobody_else():
    """혼자 있는 테마는 짝꿍이 없다 — 그 경우도 섹터로 떨어져야 한다."""
    d = _det(themes={"L": ["T9"]})
    assert set(d.peers("L")) == {"L", "A", "B", "X"}


def test_theme_index_rebuilds_when_codes_are_added_intraday():
    """장중에 붙은 종목이 테마 무리에 안 들어가면 짝꿍 후보가 못 된다."""
    d = _det(themes={"L": ["T1"]})
    assert set(d.peers("L")) == {"L", "A", "B", "X"}      # 아직 T1 동료가 없다
    d.add_codes(["NEW"], sectors={"NEW": "화학"}, prev_close={"NEW": 1000.0},
                themes={"NEW": ["T1"]})
    assert set(d.peers("L")) == {"NEW"}


# ----- 새 대장을 받아들이는 시간 창 -------------------------------------------

def test_adoption_window_matches_manju_morning():
    """만쥬: "오전 09:00~09:30 사이 ... 대장 후보로 즉시 낙점"."""
    from scalp_it.realtime.leader_scan import in_adoption_window

    assert in_adoption_window("0905", "0900-0930") is True
    assert in_adoption_window("0900", "0900-0930") is True
    assert in_adoption_window("0930", "0900-0930") is True
    assert in_adoption_window("0931", "0900-0930") is False
    assert in_adoption_window("1400", "0900-0930") is False


def test_empty_window_is_always_open():
    """창을 안 걸면 종일 새 대장을 받는다(기본값)."""
    from scalp_it.realtime.leader_scan import in_adoption_window

    assert in_adoption_window("1400", "") is True
    assert in_adoption_window("1400", "   ") is True


def test_colon_form_is_accepted():
    from scalp_it.realtime.leader_scan import in_adoption_window

    assert in_adoption_window("09:05", "09:00-09:30") is True


def test_malformed_window_does_not_block_the_day():
    """오타 하나로 하루를 날리지 않는다 — 형식이 틀리면 막지 않는다."""
    from scalp_it.realtime.leader_scan import in_adoption_window

    assert in_adoption_window("1400", "쓰레기") is True


# ----- 감시 구독과 주문 허용은 다른 목록이다 -----------------------------------

def test_intraday_codes_reach_the_order_whitelist():
    """구독만 늘리면 신호는 나는데 주문이 전부 차단된다 — 2026-08-27 실측 사고."""
    from scalp_it.adapters.kiwoom_order import OrderGuardConfig
    from scalp_it.cli_pair_detect import _allow_orders

    class _Client:
        def __init__(self):
            self.guard = OrderGuardConfig(whitelist=frozenset({"000001"}))

        def allow_codes(self, codes):
            self.guard = self.guard.with_allowed(set(codes))

    class _Exec:
        def __init__(self, c):
            self._client = c

    c = _Client()
    _allow_orders(_Exec(c), ["000002", "000003"])
    assert c.guard.whitelist == frozenset({"000001", "000002", "000003"})


def test_allow_orders_is_a_noop_without_an_executor():
    """dry-run·리플레이엔 실행기가 없다 — 거기서 터지면 안 된다."""
    from scalp_it.cli_pair_detect import _allow_orders

    _allow_orders(None, ["000002"])          # 예외 없이 지나가야 한다


def test_with_allowed_keeps_the_original_entries():
    from scalp_it.adapters.kiwoom_order import OrderGuardConfig

    g = OrderGuardConfig(whitelist=frozenset({"000001"}))
    assert g.with_allowed({"000002"}).whitelist == frozenset({"000001", "000002"})
    assert g.whitelist == frozenset({"000001"}), "원본은 그대로여야 한다(frozen)"


# ----- 종목코드 정규화: 알파벳이 든 코드를 다른 회사로 바꾸면 안 된다 ------------

def test_letters_inside_the_code_survive():
    """``0155E0``(해치텍) → ``001550``(조비) 로 바뀌던 버그. 실주문이 걸린 경로다.

    KRX 코드에는 가운데 알파벳이 들어가는 것이 실재한다(2026-08-27 기준 55종목).
    이게 뭉개지면 보유종목 차단이 **진짜 보유분을 놓치고 엉뚱한 종목을 막는다**.
    """
    from scalp_it.adapters.kiwoom_order import normalize_code

    assert normalize_code("0155E0") == "0155E0"
    assert normalize_code("0220W0") == "0220W0"


def test_kt00018_letter_prefix_is_still_stripped():
    """계좌 조회 응답은 ``A000010`` 형식이다 — 접두 한 글자는 그대로 떼야 한다."""
    from scalp_it.adapters.kiwoom_order import normalize_code

    assert normalize_code("A000010") == "000010"
    assert normalize_code(" a000010 ") == "000010"


def test_short_numeric_codes_are_zero_padded():
    from scalp_it.adapters.kiwoom_order import normalize_code

    assert normalize_code("10") == "000010"
    assert normalize_code(43200) == "043200"


def test_empty_code_stays_empty():
    from scalp_it.adapters.kiwoom_order import normalize_code

    assert normalize_code("") == "" and normalize_code(None) == ""


def test_blocklist_matches_the_real_holding_not_a_lookalike():
    """보유종목 차단이 실제로 그 종목을 막는지 — 이게 이 함수의 존재 이유다."""
    from scalp_it.adapters.kiwoom_order import OrderGuardConfig

    g = OrderGuardConfig(whitelist=frozenset()).with_blocked({"0155E0"})
    assert "0155E0" in g.blocklist
    assert "001550" not in g.blocklist, "엉뚱한 회사를 막으면 안 된다"
