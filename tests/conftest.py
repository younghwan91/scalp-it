import sqlite3

import pytest


@pytest.fixture(autouse=True)
def _no_production_db(monkeypatch):
    """테스트는 운영 DB 에 닿지 않는다.

    ``scripts/_envload.py`` 가 ``../quant-airflow/.env`` 를 읽어 ``KR_QUANT_DB`` 가 늘 잡혀 있다.
    krx-quant-core[db](psycopg) 를 깐 뒤로 ``start_run`` 이 그 DSN 으로 운영 ``kqc_runs`` 에 실제로
    썼다(2026-10-12, test_pair_sweep_engine 의 임시 레포가 repo='repo' 14행). DB 가 필요한 테스트는
    sqlite 경로 등을 ``monkeypatch.setenv`` 로 직접 준다.
    """
    monkeypatch.delenv("KR_QUANT_DB", raising=False)


@pytest.fixture()
def news_db(tmp_path):
    """계획 2 가 만드는 news.db 와 같은 스키마의 sqlite fixture."""
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE articles (
            news_id TEXT PRIMARY KEY, date TEXT, outlet TEXT, title TEXT,
            body TEXT, keywords TEXT, orgs TEXT, url TEXT
        );
        CREATE TABLE article_codes (news_id TEXT, code TEXT, PRIMARY KEY (news_id, code));
        CREATE TABLE daily_news (
            date TEXT, code TEXT, n_articles INTEGER, n_outlets INTEGER,
            article_ratio REAL, adv REAL, tv_ratio REAL, hotness REAL,
            "rank" INTEGER, in_ops INTEGER, primary_news_id TEXT,
            PRIMARY KEY (date, code)
        );
        INSERT INTO articles VALUES
            ('n1','2026-08-14','한국경제','체코 원전 2기 최종 계약',
             '두산에너빌리티가 체코 두코바니 원전 2기 주기기 공급 계약을 체결했다.',
             '원전,수주','두산에너빌리티','https://x/1'),
            ('n2','2026-08-14','더벨','반도체 감산 루머',
             '업계에 따르면 감산 논의가 있었다는 이야기가 돈다.',
             '반도체','','https://x/2');
        INSERT INTO daily_news VALUES
            ('2026-08-14','034020',7,5,4.2,8420.0,4.6,0.81,1,1,'n1'),
            ('2026-08-14','005930',3,2,1.8,61000.0,2.1,0.44,2,0,'n2'),
            ('2026-08-14','000660',1,1,0.9,5200.0,1.2,0.20,3,0,NULL),
            ('2026-08-13','034020',2,2,1.1,8300.0,1.0,0.15,1,1,'n1');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture()
def cron_args():
    """cron 실행줄의 플래그를 그대로 파싱한다 — 문서와 배선이 갈리는 걸 막는다.

    경로는 레포 루트 기준으로 잡는다(CWD 와 무관해야 어디서 pytest 를 돌려도 같다).
    """
    import re
    import shlex
    from pathlib import Path

    from scalp_it.cli_pair_detect import build_parser

    sh = (Path(__file__).resolve().parent.parent / "scripts" / "pair_detect_daily.sh").read_text(
        encoding="utf-8"
    )
    m = re.search(r"scalp-pair-detect \\\n(.*?)>> data/pair_detect\.log", sh, re.DOTALL)
    assert m, "cron 실행줄을 못 찾았다"
    flags = (m.group(1).replace("\\\n", " ")
             .replace('"$(date \'+%H:%M\')-15:20"', "09:00-15:20"))
    return build_parser().parse_args(shlex.split(flags))
