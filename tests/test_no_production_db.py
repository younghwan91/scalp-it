import os


def test_tests_never_see_production_dsn():
    # conftest 의 autouse 가드가 빠지면 start_run 이 운영 kqc_runs 에 쓴다(2026-10-12).
    assert "KR_QUANT_DB" not in os.environ
