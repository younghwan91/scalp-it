# scalp-it — Claude 작업 규칙

국장 일중 틱·호가 수집, 짝꿍(만쥬) 감지기, 미시구조·RL 연구 레포다. 현물 일중 연구는 102번 FINAL과 전진 최종 판정 9/9 기각으로
닫혀 있다. 지금 실제로 도는 것은 dry-run 감지기와 수집 크론이다. 상태는 자주 바뀌므로 수치는 여기 적지 않고 아래 문서를 본다.

## 먼저 읽을 문서 (최신순)
- `docs/TRADING_CRITERIA.md`: 날짜별 결정이 맨 위에 쌓이는 살아 있는 문서. 현재 판정 상태는 여기서 본다.
- `docs/research/manju/NEXT.md` 머리말과 `99-where-we-stand.md`: 무엇이 닫혔고 왜 닫혔는지(비용 벽 표).
- `docs/PROMOTION_CHECKLIST.md`: 실주문 재개나 라이브 규칙 변경을 논할 때. A·B 게이트 11개를 전부 통과해야 승격한다.
- `docs/PAIR_TRADING_RUNBOOK.md`: 감지기를 직접 다룰 때. `scripts/README.md`: 스크립트 색인.
- `docs/archive/README.md` "반드시 기억할 실패 다섯": 새 아이디어를 내기 전에 본다.
- 낡은 문서: `docs/STATUS.md`(2026-08-29)와 README의 운영표·"실주문 1주" 문구는 옛 내용이다. 운영의 정본은
  `deploy/crontab.simnode`와 `scripts/pair_detect_daily.sh`다.

## 명령
```bash
uv sync --extra dev
uv run pytest -q                      # 약 1,600개, 45초. scalp-it에는 CI 테스트가 없어서 푸시 전에 직접 돌린다
uv run ruff check <바꾼 파일>          # 레포 전체에는 기존 위반이 수백 건 있다. 대량 자동 수정 금지
uv run python scripts/forward_verdicts.py           # 전진 최종 판정 모아보기 (읽기 전용)
uv run --extra pg python scripts/<x>.py ...         # DB 쓰는 스크립트. DSN은 ../quant-airflow/.env의 KR_QUANT_DB
deploy/research_run.sh uv run python <학습·대규모 스크립트>  # 무거운 연구는 반드시 이 래퍼로 (라이브와 같은 호스트)
```
- `.venv`를 activate하지 않는다. `VIRTUAL_ENV`가 아직 옛 경로 `~/Documents/git/...`를 가리킨다. 항상 `uv run`을 쓴다.
- RL·밴딧 연구 스크립트(`forward_gate.py`, `tiger_86/`, `collect_health.py`)는 전용 venv `~/of80/.rlvenv/bin/python`으로
  돈다. 학습 산출물은 `~/of80/*`에 있다.
- **워크트리는 `~/git/<이름>`처럼 `quant-airflow`의 형제 위치에 만든다.** `scripts/_envload.py`가
  `../quant-airflow/.env`를 import 시점에 읽기 때문이다. `.claude/worktrees/`나 다른 깊이에 만들면 pytest 22건이
  `FileNotFoundError: quant-airflow/.env`로 실패한다(실패 21, 수집 오류 1). 위치 탓이지 내가 바꾼 코드 탓이 아니다.

## 라이브 운영: 이 PC(simnode)가 장중 라이브 호스트다 (2026-10-07부터)
- 평일 08:40~15:45에는 크론이 `~/git/scalp-it` 체크아웃으로 감지기·수집·DART를 돌린다. 그 시간에는 이 체크아웃에서
  브랜치 전환, pull, `uv sync`, `.venv` 변경을 하지 않는다. 작업은 워크트리에서 한다.
- 키움 App key 1개는 WebSocket 세션 1개다. 장중에 `scalp-pair-detect`, `scalp-tick-collect`, `scalp-intraday`처럼
  키움을 쓰는 명령을 손으로 띄우지 않는다. 같은 키로 붙으면 라이브 소켓이 밀려나고, 틱은 소급 수집이 불가능하다.
- 감지기는 2026-09-16부터 dry-run이다(`--confirm-real` 없음). `--confirm-real`을 다시 넣는 일은 PROMOTION_CHECKLIST를
  통과하고 사용자가 승인한 뒤에만 한다.
- 크론을 바꿀 때는 `deploy/crontab.simnode`를 고쳐 커밋한다. 07:50 동기화가 이 파일로 설치하므로 `crontab -e`로 고친 내용은 덮인다.
- `pair_detect_daily.sh`의 플래그는 테스트가 고정한다(`tests/conftest.py`, `test_profit_giveback.py`,
  `test_cli_pair_detect_wiring.py`). 연속손절 킬 `2`는 만쥬 지시서 §5의 값이므로 표본을 늘리려고 완화하지 않는다.
- 라이브 셸 스크립트는 경로를 `${BASH_SOURCE[0]}`에서 유도한다. `cd` 실패는 명시적으로 죽게 둔다. 2026-09-11 레포 이동 때
  `set -e` 없는 런처가 빈 자격증명으로 떠서 하루를 날렸다.
- 프로세스를 찾거나 죽일 때 `pgrep -f "[s]calp-pair-detect"`처럼 대괄호 트릭을 쓰거나 PID를 명시한다. 맨 패턴은 내 셸
  명령줄까지 잡는다. 죽이기 전에 `/proc/<pid>/cmdline`을 확인한다.
- DB: `KR_QUANT_DB`는 고정 호스트가 아니라 지금 PRIMARY를 가리킨다. 컨테이너 이름은 역할과 반대로 붙어 있으니
  `select pg_is_in_recovery()`(false = PRIMARY)로 확인한다. `scripts/db_guard.sh`는 split-brain을 만들므로 크론에 되살리지 않는다.
- 수집기(`realtime/collector.py`)의 DB·소켓 경로를 고치면 `--dsn` 없이 환경변수로 붙은 경우까지 테스트한다
  (`test_realtime_collector.py`, `test_collector_reconnect.py`). 10-12 OOM 재시작 때 이 경로만 재연결이 안 됐다(72da0fc).

## 연구 규율
- 탐색 → 사전등록(`docs/research/manju/NNN-*-preregistration.md`) → 잠금 → 전진 → 한 번 판정. 탐색 결과로 바로
  라이브 코드를 짜지 않는다. 기각된 것을 살리려고 파라미터를 바꾸지 않는다.
- **잠긴 사전등록 문서는 고치지 않는다. 새 번호로 쓴다.** `research/runs/<study>/PREREG.lock`의 sha256이 문서와 어긋나면
  krx-quant-core가 최종 평가를 `RunRefused`로 거부한다. 잠긴 문서, `*.lock`, `OOS.json`, `RUNS.jsonl`, `TRIALS.jsonl`은
  `.claude/settings.json`에서 편집을 거부하게 막아 두었다. 잠금은 `uv run kqc prereg lock <label> <doc>`로 하고, 잠근 문서는 deny 목록에 추가한다.
- 원장(`RUNS.jsonl`, `TRIALS.jsonl`)은 코드(`start_run`, `kqc nightly`, 연구 스크립트의 시행 기록기)만 쓴다. 손으로 고치지 않는다.
- 새 아이디어는 먼저 `99-where-we-stand.md`의 비용 벽과 기각 목록에 비춰 본다. 상따·낙주·호가흐름(80)·수초 보유(95)·
  급등주 일중 롱(97·102)은 이미 기각됐다. 백로그에 "신규"로 다시 올린 사고가 있었다(2026-09-06).
- 순위·필터는 **전일** 값으로 매긴다. 당일 거래대금 순위를 쓰면 look-ahead가 생겨 부호까지 뒤집혔다(97번).
- 대용 지수가 유망해 보여도 거래 가능한 실물로 다시 재기 전에는 믿지 않는다(인버스 ETF −16.6bp가 실물에서 −5.0bp, t=−1.59).
- "표본이 없다"고 말하기 전에 `minute_bars`(분봉 크론 약 1,400종목)를 `information_schema`로 확인한다.

## 데이터 함정
- `minute_bars`는 기업행동 뒤 재수집 구간이 키움 수정주가로 덮여 있다. 일봉과 섞을 때 전일종가는 그날 **분봉 15:30**으로
  잡는다. 연속 시계열은 `daily_bars_adjusted`를 쓰고, 실제 호가 단위는 틱만 믿는다(c03baff, 5f4a8b0, 833ddf7).
- 수집 불량일은 `research/bad_days.json`으로 캐시에서만 뺀다. 원본 행은 지우지 않는다. 사람이 적은 항목은 덮어쓰지 않는다.

## 코드 규칙
- 시장 규칙, 주문 가드, 체결 판정, 검증 통계의 정본은 `krx-quant-core`다(README "공통 코어 의존성" 표). 여기에 사본을 두지
  않는다. daytrade-it과 겹치는 새 로직은 코어로 보낸다. 코어를 올릴 때는 `pyproject.toml` 핀을 바꾸고 `uv lock`을 한 다음
  전체 테스트를 돌린다.
- 계좌에 묶인 것(`HELD_STOCK_CODES`)과 주문 HTTP 전송(`KiwoomOrderClient`)은 이 레포에 남긴다.
- 실시간 피처는 백테스트와 **같은 함수**로 계산한다(PROMOTION_CHECKLIST 8번 피처 패리티).

## Git
- 커밋 신원은 레포 설정(`chyohw97@gmail.com`)을 그대로 쓴다. 세션 컨텍스트의 회사 이메일로 `git -c user.email=...`를
  쓰지 않는다. `.githooks/pre-commit`이 막는다(2026-08-29 사고). 새 클론에서는 `git config core.hooksPath .githooks`를 실행한다.
- 커밋 메시지는 한국어 conventional 형식을 쓴다. 예: `fix(realtime): ...`, `research(106): ...`, `docs(108): ...`, `chore(cron): ...`
- 데이터(`data/`, csv, parquet, jsonl)는 커밋하지 않는다. 예외는 `research/runs/*/RUNS.jsonl`과 연구 원장이다.
