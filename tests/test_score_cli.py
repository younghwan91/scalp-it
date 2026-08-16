import json
import sqlite3

import pytest

from swing_it.score.cli import main


@pytest.fixture()
def workspace(tmp_path):
    """가격 DB · 뉴스 DB 를 함께 깐다. 어느 쪽도 네트워크를 쓰지 않는다."""
    price_path = tmp_path / "prices.db"
    conn = sqlite3.connect(price_path)
    conn.executescript(
        """
        CREATE TABLE daily_bars_adjusted (
            code TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, trade_value REAL, source TEXT
        );
        """
    )
    rows = []
    for code in ("005930", "000660"):
        for day in range(1, 6):
            # 5일차만 거래대금 4배로 튄다 → window=3 에서 이벤트가 된다
            trade_value = 24000.0 if day == 5 else 6000.0
            rows.append((code, f"2026-01-0{day}", 100.0, 110.0, 90.0, 105.0,
                         1000, trade_value, "test"))
    conn.executemany(
        "insert into daily_bars_adjusted values (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    # 계획 2 의 정규화 스키마 — articles + article_codes.
    # n3 은 종목 매핑에 실패한 기사다. 매핑 실패를 사전 필터의 no_code 로
    # 세려면 빈 code 행이 있어야 한다 (조인에서 통째로 빠지면 안 센다).
    news_path = tmp_path / "news.db"
    conn = sqlite3.connect(news_path)
    conn.executescript(
        """
        CREATE TABLE articles (
            news_id  TEXT PRIMARY KEY,
            date     TEXT NOT NULL,
            outlet   TEXT,
            title    TEXT,
            body     TEXT,
            keywords TEXT
        );
        CREATE TABLE article_codes (
            news_id    TEXT NOT NULL,
            code       TEXT NOT NULL,
            match_kind TEXT NOT NULL,
            PRIMARY KEY (news_id, code)
        );
        INSERT INTO articles VALUES
            ('n1','2026-01-05','한국경제','삼성전자 HBM 대규모 공급 계약 체결',
             '삼성전자가 미국 A사와 3년간 2조원 규모의 공급 계약을 맺었다.','삼성전자,HBM'),
            ('n2','2026-01-05','매일경제','SK하이닉스 청주 증설 결정 발표',
             'SK하이닉스가 청주 M15X 라인 증설을 이사회에서 결정했다.','SK하이닉스,증설'),
            ('n3','2026-01-05','더벨','종목 매핑에 실패한 기사 제목입니다',
             '어느 종목인지 특정되지 않은 시황 기사 본문이다.','시황'),
            ('n4','2026-01-02','서울경제','거래대금이 튀지 않은 날의 기사 제목',
             '평범한 날의 평범한 기사 본문이 여기에 들어간다.','삼성전자');
        INSERT INTO article_codes VALUES
            ('n1','005930','name'),
            ('n2','000660','name'),
            ('n3','','none'),
            ('n4','005930','name');
        """
    )
    conn.commit()
    conn.close()
    return {"prices": str(price_path), "news": str(news_path), "root": tmp_path}


def _run(workspace, *extra):
    out = workspace["root"] / "report.json"
    code = main(
        [
            "--db", workspace["prices"],
            "--news-db", workspace["news"],
            "--cache-db", str(workspace["root"] / "llm.db"),
            "--start", "2026-01-01",
            "--end", "2026-01-31",
            "--provider", "fake",
            "--event-window", "3",
            "--out", str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_cli_runs_end_to_end_with_fake_provider(workspace):
    code, report = _run(workspace)
    assert code == 0
    assert report["prefilter"]["n_input"] == 4
    assert report["prefilter"]["no_code"] == 1
    assert report["prefilter"]["no_volume_event"] == 1
    assert report["prefilter"]["n_kept"] == 2
    assert report["batch"]["n_ok"] == 2
    assert report["batch"]["n_failed"] == 0


def test_cli_records_model_and_prompt_version(workspace):
    _, report = _run(workspace)
    assert report["model"] == "fake-1"
    assert report["prompt_version"]
    assert report["provider"] == "fake"


def test_cli_second_run_is_all_cache(workspace):
    _run(workspace)
    _, report = _run(workspace)
    assert report["batch"]["n_called"] == 0
    assert report["batch"]["n_cached"] == 2


def test_cli_limit_caps_calls(workspace):
    _, report = _run(workspace, "--limit", "1")
    assert report["batch"]["n_called"] == 1


def test_report_contains_no_article_text(workspace):
    """results/ 는 커밋될 수 있다. 기사 본문이 새면 안 된다."""
    _, report = _run(workspace)
    blob = json.dumps(report, ensure_ascii=False)
    assert "삼성전자가 미국 A사와" not in blob
    assert "n1" not in blob
