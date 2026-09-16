"""95g번 매치드 널 입력 — **진입 시각과 gross 환산**이 그쪽 규격과 맞는지.

시각이 밀리면 널의 분(minute) 셀이 어긋나고, 비용을 두 번 빼면 alpha 가 통째로 음수로 나온다.
"""
import importlib.util
import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "null_export.py"
spec = importlib.util.spec_from_file_location("null_export95", _p)
N = importlib.util.module_from_spec(spec)
sys.modules["null_export95"] = N
spec.loader.exec_module(N)


def test_seconds_convert_to_kst_clock_from_nine():
    assert N._hhmmss(0) == "09:00:00"
    assert N._hhmmss(3600) == "10:00:00"
    assert N._hhmmss(22799) == "15:19:59"


def test_tick_bins_cover_everything_and_do_not_overlap():
    names = []
    for v in (0.0, 3.0, 6.99, 7.0, 11.99, 12.0, 500.0):
        hit = [n for lo, hi, n in N.TICK_BINS if lo <= v < hi]
        assert len(hit) == 1, f"{v} 가 {len(hit)}개 구간에 걸린다"
        names.append(hit[0])
    assert names[:3] == ["tick_narrow"] * 3 and names[3:5] == ["tick_mid"] * 2
