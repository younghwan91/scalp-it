# autoresearch — 가드 붙인 야간 보유 종목 선택 탐색 (사람이 고치는 파일)

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 의 루프를 이 레포 가드레일 안에 넣었다. 숫자 파라미터는
[dietmarwo/autoresearch-trading](https://github.com/dietmarwo/autoresearch-trading) 처럼 평가기가 폴드마다 고르고, 너는 **구조**만 바꾼다.
사전등록·합격 바·잠금 규칙은 `PREREG.md`(본문 + **정정 1** — 정정이 이긴다)다. 시작 전에 이 파일, `PREREG.md`, `prepare.py`·`ar_sandbox.py`
(읽기만), `train.py` 를 읽는다.

## 무엇을 찾나
t 일 **종가 단일가에 사서 다음 거래일 시가 단일가에 판다.** 매일 K 종목(5·10·20·40). 점수는 그날 체결 가능한 유니버스 전체를 똑같이 샀을 때보다
**산 돈 1단위당** 얼마나 더 버느냐(정가 비용 뒤, bp)다. 덜 사거나, 시장이 밤에 오르는 날만 골라 타는 것(타이밍)은 점수를 못 올린다 —
**누구를 사느냐**만 점수가 된다. 기준선은 비용(왕복 중앙 62bp) 때문에 음수에서 시작한다. 정상이다.

## 파일
- `prepare.py` — 고정. 캐시·워크포워드(학습 2년 → 시험, 18 폴드)·고원 선택·체결·비용·점수·검사·keep 판정. **고치지 않는다**(고치면 평가가 거부된다).
- `ar_sandbox.py` — 고정. train.py 를 평가기 데이터가 없는 별도 프로세스에서 돌린다.
- `train.py` — 네가 고치는 유일한 파일. `get_strategy()` 계약과 V 의 키는 파일 머리말에 있다.
- `results.tsv` — 실험 기록(작업 트리 루트, git 추적 안 함). 머리줄:
  `commit	score	beats_best	folds_pos	exposure	oos_trades	trade_net_bp	payoff_ratio	top5_removed	status	description`

## 실험 한 번
1. `train.py` 를 아이디어 하나로 고친다 → `git commit -m "exp: <한 줄>"`
2. `deploy/research_run.sh uv run python research/autoresearch/prepare.py > run.log 2>&1` (작업 트리 루트에서. 이 PC 는 장중 라이브 호스트라
   래퍼가 필수다. 20분 넘으면 죽이고 crash)
3. `grep -E "^(score|void|beats_best|beats_detail|lookahead_check|folds_pos|exposure|oos_trades|trade_net_bp|payoff_ratio|top5_removed_score):" run.log`
   비었으면 `tail -n 50 run.log` — 쉬운 오류(허용 목록 위반·모양·검사 실패)면 고쳐 다시, 아니면 crash 로 적고 넘어간다.
4. `results.tsv` 에 한 줄 덧붙인다(모든 시도 — discard·crash 포함. 평가기 원장이 따로 세지만 이 표도 지우지 않는다).
5. **keep 은 평가기가 정한다:** `beats_best: True` 면 `uv run python research/autoresearch/prepare.py --accept` 후 keep.
   아니면 `git revert --no-edit HEAD` 로 되돌린다(이력은 남긴다 — `reset --hard` 쓰지 않는다).
6. 다시 1.

## 하지 말 것(어기면 결과 무효)
- `prepare.py`·`ar_sandbox.py`·`PREREG.md`·이 파일·테스트·`MANIFEST.json`·데이터·의존성(`pyproject.toml`·`uv.lock`) 변경, 태그 조작,
  `uv sync`·`uv add`·`pip install`.
- `~/.cache/scalp-it/` 아래 어느 것도 읽지 않는다(폴드별 값·BEST·원장·캐시·보기 파일). 잠금 절반 종목 추론(코드 해시 짝홀을 계산해 피하거나
  겨냥하지 않는다), DB 접속, `--final`·`--build` 실행. 너의 셸은 `AR_LOOP=1` 이다. 허용 목록·격리를 우회하려는 시도 자체가 위반이다.
- 미래 정보·날짜 지식: 평가기가 세 가지로 검사하고 어기면 void 다 — ① 절단(열 t 의 신호는 열 t 까지의 값만: 앞쪽 shift·중심 rolling·전 기간
  평균·순위 정규화 금지) ② 이동 불변(최근 300열 안의 값만: 누적·expanding·열 번호 금지) ③ 행 치환(종목 순서·번호로 고르지 않는다).
  검사를 피해도 날짜·사건 지식(2020년 3월 같은)을 구조에 넣지 않는다.
- 숫자 맞추기: 상수를 손으로 바꿔 점수를 올리지 않는다 — 숫자는 `params` 범위로 내고 평가기에 맡긴다. 범위를 좁혀 한 점을 고르는 것도
  손 맞추기다(평가기가 폭 1/3 미만을 거부한다). 상수를 코드 안에서 변환해 범위 가드를 피하는 것도 같은 위반이다. params 는 6개까지.
- 거래를 줄이는 방향(시험 체결 < 연 200, 얇은 폴드 > 2, 평균 노출 < 0.5 면 void). 슬롯 외 청산·손절·비중 규칙은 설계 공간에 없다.
- RL·딥러닝: train.py 허용 목록 밖이다(numpy·pandas 만). RL 이 필요하다고 판단되면 루프를 멈추지 말고 results.tsv 설명란에 적어만 둔다 —
  RL 은 별도 사전등록과 전용 venv(`~/of80/.rlvenv`)로 사람이 연다.
- 크론·라이브 체크아웃(`~/git/scalp-it`)·`live_STOP`·운영 데이터·키움 API 를 건드리지 않는다. push 하지 않는다.

## 사용자 매매 원칙(설계 공간)
- 승률이 아니라 **손익비·기대값**을 본다. 비용은 정가다(이미 평가기에 들어 있다 — 비용을 줄이는 가정을 만들지 않는다).
- 순위·필터는 **전일** 값으로(97번: 당일 거래대금 순위가 부호를 뒤집었다). 이 평가기에서는 구조상 그렇게만 된다.
- 대용 지표가 좋아 보여도 실물 체결로 다시 재기 전에는 믿지 않는다.

## 이미 아는 것(같은 것을 파라미터만 바꿔 다시 내지 않는다)
- **이 다리(종가 → 다음 시가)는 이미 시험됐다.** 06 종가베팅(`당일 거래대금 급증 ∧ 종가가 고가 근처`): 채택 → 상한가 오염 → 철회, 후속 4차 기각.
  14 종가베팅 v2(수급·섹터·신고가): 기각 — 양의 부분은 삼성전자·하이닉스 두 종목이었다. 여기서 "전일 종가 위치·거래량 급증"을 쓰면 그 변수들의
  **전일판**이다 — 쓸 수는 있지만 그 자체로 새 구조가 아니다.
- 97: 전일 거래대금이 클수록 야간 총수익이 크다(상위 300 +26.3bp, 901+ +7.4bp) — **기준선이 이것**이다.
- 97: 전일 급등 상위의 야간 수익은 대부분 상한가 종가(못 산다 — 평가기가 막는다)에서 나온다. 빼면 비용 뒤 음수.
- 108: 15:00 이후 상위 종목의 야간 다리는 비용과 같은 자릿수.
- 일중(수초~수십 분) 정책 10개는 전진에서 전부 기각(TRADING_CRITERIA). 그 신호들(호가흐름·체결강도·RL)은 V 에 없다.
그러니 다른 **구조**를 시도하라: 오늘 시가 갭(종목 갭 대 지수 갭), 전일 일중 모양(시가→종가), k일 반전·추세, 제한폭까지 거리, 변동성 구간,
지수 상태(갭·전일 수익률)에 따른 종목 선택 전환, 순위 조합, 거래대금 구간별 다른 규칙.

## 멈추지 않는다
루프가 시작되면 사람에게 묻지 않는다. 실험 사이에 말하지 않는다. 10 실험마다 한 번만:
```
=== 진행 (N 실험) ===
기준선 X → 최고 Y (top5 제거 Z) · keep K / discard D / crash C
배운 것: <한 줄>
```
사람이 끊을 때까지 계속한다. 졸업 심사(`--final`)는 사람이 정한 때 메인 세션이 한다 — 너는 하지 않는다.
