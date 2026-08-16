# 계획 4: 층별 모델 비교 + 근거 카드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 §3 의 세 층(①수급 단독 / ②+규칙 뉴스 / ③+LLM)을 **동일 표본·동일 폴드·동일 음성대조 널** 위에서 비교해 각 층이 값을 하는지 판정하고, PEAD 대비 판정과 사람이 읽는 근거 카드, 6개월 뒤 채점 가능한 전진 기록까지 산출한다.

**Architecture:** 층은 "다른 데이터셋"이 아니라 **하나의 데이터셋 위에서 쓰는 피처 컬럼의 부분집합**이다. 세 층이 같은 행 위에서 돌게 만드는 것이 이 계획의 전부라 해도 된다 — 표본이 다르면 층 비교는 아무것도 말해주지 않는다. 그래서 데이터셋 조립 시점에 세 층 피처를 모두 붙이고 하나라도 결측인 행을 통째로 떨군다. 폴드 경계는 한 번 계산해 세 층에 그대로 넘기고, 널은 `excess_return` 만으로 만들어지므로 세 층에서 자동으로 동일해진다(테스트로 강제한다).

**Tech Stack:** Python ≥3.10, uv, pandas ≥2.0, scikit-learn, numpy, sqlite3, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-08-17-swing-it-design.md`

**선행 계획:**
- 계획 1 `docs/superpowers/plans/2026-08-17-baseline-skeleton.md` (필수)
- 계획 2 `docs/superpowers/plans/2026-08-17-news-ingest.md` (필수 — `data/news.db` 가 존재해야 실행 가능)
- 계획 3 LLM 피처 (필수 — ③층)

## Global Constraints

- **수정주가만 사용한다.** 가격은 `daily_bars_adjusted` 에서만 읽는다. `daily_bars` 는 분할 미반영이며 절대 쓰지 않는다.
- **point-in-time 엄수.** 시점 t 의 피처는 t 시점에 알 수 있는 정보만 쓴다. 롤링 통계는 반드시 `.shift(1)` 후 계산한다. 계획 2 의 `add_hotness` 가 60일 baseline 을 shift(1) 로 이미 끝냈으므로 **계획 4 에서 뉴스 지표를 다시 롤링하지 않는다** — 이중 계산이다.
- **유동성 하한.** 이벤트 후보는 진입 직전 20거래일 평균 `trade_value` ≥ 5000 (백만원 = 50억원). 스펙 §5.
- **드리프트를 자르지 않는다.** 라벨은 D+21 ~ D+41 구간 보유. 중간 손절·이익실현 없음. 스펙 §2.
- **동시 보유 15종목 이상.** `top_n` 하한 15. 4~8 종목 집중은 이미 죽은 영역이다. 스펙 §2.
- **음성대조 없이 승격 없음.** 모든 성과 주장은 랜덤 진입 널 대비로 보고한다. 스펙 §2.
- **모델은 얕게 고정.** `GradientBoostingRegressor(max_depth=2, n_estimators=100)`. 층이 늘어도 여기서 올리지 않는다. 스펙 §6.
- **FAIL 을 숨기지 않는다.** 리포트는 FAIL 을 그대로 출력하고, 근거 카드에도 그 층의 게이트 판정을 함께 찍는다. 파라미터를 바꿔 재시도하는 경로는 만들지 않는다. 스펙 §10.
- **데이터 커밋 금지.** 결과물은 `data/`·`results/` 아래 두며 `.gitignore` 가 `data/`·`results/`·`*.db` 를 차단한다. 빅카인즈 재배포 금지·기사 저작권. 스펙 §8.
- **DB 는 읽기 전용.** TimescaleDB 도, `data/news.db` 도 읽기만 한다.

## 계획 1 에서 그대로 쓰는 것

이 계획은 계획 1 의 다음 이름들을 **깨지 않고 재사용**한다. 계획 1 테스트는 전부 계속 통과해야 한다.

| 이름 | 시그니처 |
|---|---|
| `swing_it.dataset.FEATURE_COLUMNS` | `list[str]` — 7개 수급·유동성 피처 |
| `swing_it.dataset.build_dataset` | `(prices, supply_demand, *, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` |
| `swing_it.model.baseline.walk_forward_predict` | `(ds, *, n_folds=4, min_train=200, seed=0) -> pd.DataFrame` |
| `swing_it.model.baseline.portfolio_returns` | `(pred, *, top_n=15) -> pd.DataFrame` |
| `swing_it.validate.gate.gate_report` | `(pred, *, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict` |
| `swing_it.validate.gate.random_null` | `(pred, *, top_n=15, n_trials=200, seed=0) -> np.ndarray` |
| `swing_it.validate.diagnostics.fold_null_table` | `(pred, *, top_n=15, n_trials=200, seed=0) -> pd.DataFrame` — 컬럼 `[fold, n_dates, strategy_mean, null_mean, percentile, beats_null]` |
| `swing_it.validate.diagnostics.untouched_window` | `(pred, *, fraction=0.25, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict` — `gate_report` 키 + `n_dates, start, end` |
| `swing_it.validate.diagnostics.placebo_shuffle` | `(pred, *, seed=0) -> pd.DataFrame` — 날짜 안에서 라벨만 섞은 프레임 |
| `swing_it.adapters.db.load_stock_master` | `(conn) -> pd.DataFrame` — 컬럼 `[code, name, market, sector, last_trade_date, is_delisted]` |

`build_dataset` 의 시그니처는 손대지 않는다. 층 확장은 **피처 목록을 인자로 받는 새 함수**(`build_layered_dataset`, `walk_forward_predict_features`)로 옆에 둔다. Task 3 은 `walk_forward_predict_features(ds, FEATURE_COLUMNS)` 가 `walk_forward_predict(ds)` 와 **행 단위로 동일한 결과**를 내는지 테스트로 고정한다 — 이게 계획 1 과의 호환 계약이다.

## 계획 1 이 이미 낸 결과 — 이 계획이 이어받는 것

계획 1 은 구현·실행이 끝났고 판정은 **FAIL** 이다
(`docs/research/2026-08-17-baseline-verdict.md`). 그 문서에서 계획 4 로 넘어온 것이 셋이다.

**1. 판정은 `gate_report` 하나로 하지 않는다.** 계획 1 의 전체 표본 판정은 PASS 였지만,
폴드별로 뜯으니 3/4 폴드에서 전략이 랜덤 널과 소수점 넷째 자리까지 같았고, 손 안 댄
최종창은 절대값이 음수였다. 전체 평균 하나로 판정했으면 아티팩트를 다음 층으로
올렸을 것이다. 그래서 **층별 판정은 `diagnostics.py` 의 4종 진단 종합**으로 한다(Task 4):

```
전체_널_대비        gate_report
모든_폴드_널_대비   fold_null_table — 모든 폴드가 beats_null 이어야 통과
손_안_댄_최종창     untouched_window(fraction=0.25) — net_mean > 0 이어야 통과
플라시보_FAIL       placebo_shuffle → gate_report 가 FAIL 이어야 통과 (누수 검출)
```

**2. 하방 회피 지표를 층 비교에 넣는다** (verdict 문서 시사점 2). 계획 1 폴드 3 이 보여준
것은 "상위를 잘 고르기"보다 "최악의 꼬리를 피하기"가 이 데이터에서 더 실현 가능하다는
것이다. 평균만 보면 그 축을 놓친다.

**3. 후보 풀 대비 `top_n` 비율을 리포트에 명시한다** (verdict 문서 시사점 3). 계획 1 폴드 3 은
하루 평균 24.1건 중 상위 15를 골라 **유니버스의 62%를 담고 있었다.** 그걸 "상위 15 선별"이라
부르면 안 된다. 뉴스 축이 들어오면 후보 풀이 좁아져야 정상이고, 좁아졌는지를 숫자로 보여야
한다. 이것이 계획 4 가 계획 1 의 구조적 결함을 고치는 지점이다.

**베이스라인 참고값:** 이벤트 25,559건 · 1,538종목 · 2017-01-06 ~ 2026-06-18 ·
`strategy_mean 0.594%` · `net_mean 0.134%` · `overall_verdict FAIL`.
층 비교 표본은 이보다 훨씬 작으므로 **직접 비교하지 않는다**(아래 「표본 정의」).

## 계획 2 에서 받는 것

계획 2 는 **인제스트에서 멈춘다.** 규칙 기반 뉴스 피처(`features/news.py`)는 계획 4 몫이다(Task 1). 계획 2 가 주는 것은 sqlite `data/news.db` 와 그 조회 함수다.

| 이름 | 계약 |
|---|---|
| `swing_it.news.store.load_daily_news(conn, start, end, *, ops_only=False)` | 컬럼 `[date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, hotness, rank, in_ops, primary_news_id]`. `date` `datetime64[ns]`, `in_ops` bool |
| `data/news.db` 테이블 | `articles`(PK `news_id`) / `article_codes`(PK `news_id, code`) / `daily_news`(PK `date, code`) |
| `articles` 컬럼 | `[news_id, date, outlet, title, body, keywords, orgs, url]`. `body` 는 **최대 200자** |
| 근거 카드 조인 | `daily_news.primary_news_id → articles.news_id` (그 종목·그 날 클러스터 대표 기사 1건) |

주의 두 가지:
- SQL 에서 `rank` 는 예약어라 `"rank"` 로 인용해야 한다. 계획 2 코드가 그렇게 쓰고 있다.
- `replace_daily_news` 는 날짜를 통째로 지우고 다시 넣는다(baseline 이 후행 데이터에 따라 바뀌므로). 계획 4 는 `data/news.db` 를 **읽기만** 하며, 재계산이 필요하면 `swing-news` 를 통째로 다시 돌린다.

## 계획 3 에서 받는 것

계획 3 Task 8 이 `swing_it.features.llm` 을 소유한다. **계획 4 는 소비만 한다 — 재정의하지 않는다.**

| 이름 | 계약 |
|---|---|
| `LLM_FEATURE_COLUMNS: list[str]` | `["llm_persistence", "llm_specificity", "llm_freshness", "llm_sentiment", "llm_scale"]` — `score.schema` 에서 재수출 |
| `LLM_FRAME_COLUMNS: list[str]` | `["code", "date", "material_type", "n_articles", *LLM_FEATURE_COLUMNS]` |
| `load_llm_features(cache_path="data/llm_features.db", *, news_path=None, model=None, prompt_version=None, articles=None) -> pd.DataFrame` | 컬럼 `LLM_FRAME_COLUMNS` |
| `available_models(conn) -> list[tuple[str, str, int]]` | `(model, prompt_version, n_ok)` |
| `fold_to_code_date(scores) -> pd.DataFrame` | 기사 단위 → (종목, 일자) 접기 |

접기 규칙은 계획 3 이 정한다: **대표 기사 한 건의 행을 통째로** 쓴다(`|llm_sentiment|` 최대 → 동률 시 `llm_specificity` 최대 → 동률 시 `article_id` 사전순). 컬럼별 max 를 따로 집계하면 A기사의 구체성과 B기사의 감정이 섞인, 실재하지 않는 기사가 만들어지기 때문이다.

### 계획 4 가 반환 컬럼 중 둘을 버린다

**`n_articles` 를 버린다.** 계획 2 의 규칙 뉴스 피처가 이미 `article_count` 로 같은 것을 준다(Task 1). 둘 다 넣으면 같은 축을 두 번 세는 셈이고, 더 나쁘게는 ②층과 ③층에 중복 투입돼 층 기여도 분리가 오염된다. **②층의 `article_count` 를 쓴다.** LLM 쪽 `n_articles` 는 조립 단계에서 드롭한다.

**`material_type` 을 모델 행렬에 넣지 않는다.** 9개 값을 원-핫으로 펼치면 LLM 몫이 5개에서 14개로 뛰어 스펙 §6 의 전체 10~20개 상한을 혼자 먹는다. 타깃 인코딩은 walk-forward 폴드 안에서 해야 하는데 이 표본 크기에서는 그 자체가 과적합 경로다. **근거 카드에 표시하는 문자열로만 딸려 보낸다** — 사람이 "이게 수주 건인지 루머인지"를 보는 데 쓴다.

따라서 ③층이 실제로 쓰는 LLM 피처는 `LLM_FEATURE_COLUMNS` 5개뿐이다.

## `portfolio_returns` 는 포트폴리오가 아니다 — 이 계획이 재지 않는 것

`portfolio_returns` 는 날짜마다 예측 상위 `top_n` 을 뽑아 그 초과수익을 평균한다. 보유가 40거래일이므로 **이대로 실제 담으면 동시 보유가 수백 종목이 된다.** 개인 계좌에서 불가능하고, 스펙 §2 의 "동시 보유 15종목 이상"은 지금 이름만 충족된다.

그럼에도 층 비교에는 이 지표를 그대로 쓴다. 여기서 묻는 것은 "이 모델로 얼마를 벌 수 있나"가 아니라 **"이 층의 시그널이 랜덤보다 나은가"** 이고, 그 질문에는 날짜별 상위 선택의 평균 초과수익이 타당한 측정치다. 세 층이 같은 왜곡을 공유하므로 층 간 비교도 성립한다.

**포지션 예산 시뮬레이션 — 동시 보유 상한을 두고 자리가 빌 때만 새 후보를 받는 실제 운용 재현 — 은 이 계획의 범위 밖이다.** 어떤 층이든 종합 판정을 통과해 승격을 검토하게 되면 **그 전에 별도 계획이 필요하다.** 승격 판정의 마지막 관문은 이 계획이 아니다. 여기서 구현하지 않는 이유는 범위가 커지기 때문이고, 하지 않기로 한 것이지 잊은 것이 아니다.

## 표본 정의 — 이 계획에서 가장 중요한 결정

비교 표본은 세 조건을 **모두** 만족하는 `(종목, 날짜)` 다:

1. 계획 1 `extract_events` 의 거래대금 급증 이벤트 (유동성 하한 통과)
2. 계획 2 `daily_news` 에 행이 있음 (그날 기사가 났고 핫함 상위 50 안에 들었음)
3. 계획 3 LLM 피처가 존재함

**결측 채우기는 하지 않는다.** 뉴스 행이 없는 이벤트를 `n_articles=0` 으로 채워 넣고 싶은 유혹이 있지만, 그 행은 어차피 3번에서 떨어진다. 채움 규칙을 두면 "언제 채웠고 언제 떨궜나"를 추적해야 하고, 그 복잡도가 값을 하지 않는다. 규칙은 하나다 — **어느 층 피처든 결측이면 그 행은 세 층 모두에서 빠진다.**

결과적으로 층 비교 표본은 계획 1 베이스라인 표본보다 훨씬 작다. 그래서 **①층의 여기 성과는 계획 1 의 `results/baseline.json` 숫자와 다르다.** 이건 버그가 아니다. 계획 4 리포트는 두 숫자를 나란히 찍되 "표본이 다르므로 직접 비교하지 말 것"을 명시하고, 층 간 비교에는 여기 숫자만 쓴다. 다른 표본 위의 두 숫자를 비교하는 것이 정확히 이 계획이 막으려는 오류다.

## `tv_multiple` 사전등록 절차 (Task 7 Step 1)

조건 1(거래대금 급증)과 조건 2(핫함 상위)의 교집합이 얼마나 될지는 실측해야 안다. 교집합이 수백 건이면 4폴드 walk-forward 가 성립하지 않고, 반대로 계획 1 처럼 하루 24건이면 상위 15가 풀의 62%라 "선별"이 아니다.

그래서 Task 7 은 **성과를 계산하기 전에** 표본 수와 하루 평균 풀 크기만 세는 `--dry-run` 을 먼저 돌린다. 그 숫자만 보고 `tv_multiple` 을 확정해 `results/preregistration.json` 에 기록하고, 그 다음 본 실행을 **한 번** 돌린다. 성과 숫자를 본 뒤에 파라미터를 바꾸는 경로는 없다.

기준은 **하루 평균 풀 크기**다. `top_n=15` 가 의미를 가지려면 풀이 그보다 충분히 커야 한다.

- 하루 평균 풀 ≥ 40건 **그리고** 총 이벤트 ≥ 2,000건 → `tv_multiple=3.0` (계획 1 과 동일한 이벤트 정의)
- 그 외 → `tv_multiple=1.0` (유동성 하한만 남기고 급증 조건 해제)

**`tv_multiple` 을 3.0 보다 올리는 선택지는 없다.** 계획 1 의 진단이 남긴 결론은 임계값을 조여 풀을 줄이라는 것이 아니라 **뉴스 축이 풀을 좁혀야 한다**는 것이었다. 조건 2(핫함 상위 50)가 이미 그 역할을 한다. 거래대금 임계값을 더 올리면 뉴스 축의 기여를 거래대금으로 대신 사버리는 셈이고, 층별 기여도 분리가 무너진다.

풀 크기가 어느 쪽으로 나오든 리포트는 `top_n / 평균 풀` 비율을 찍는다(Task 4 `selectivity`). 62% 같은 숫자가 다시 나오면 그 사실이 보여야 한다.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/swing_it/features/news.py` | `daily_news` → 규칙 기반 뉴스 피처 (순수 함수) |
| `src/swing_it/adapters/news_db.py` | **수정** — `load_article_cards` 추가 (뉴스 읽기 SQL 은 이 어댑터에만 둔다) |
| `src/swing_it/layers.py` | 층 정의 + 세 층 피처를 모두 붙인 단일 데이터셋 조립 |
| `src/swing_it/model/layered.py` | 피처 목록을 인자로 받는 walk-forward, 공유 폴드 경계, 최종 모델 적합 |
| `src/swing_it/validate/layer_verdict.py` | 층 하나의 종합 판정 — 진단 4종 + 하방 회피 + 선별도 |
| `src/swing_it/validate/compare.py` | 동일조건 층 비교 · 페어드 부트스트랩 · LLM 정당화 판정 |
| `src/swing_it/validate/pead.py` | 겹침 보정 Sharpe·t 계산과 PEAD 대비 판정 |
| `src/swing_it/report/__init__.py` | 리포트 계층 선언 |
| `src/swing_it/report/cards.py` | 피처 기여도 · 유사 과거 사례 · 근거 카드 마크다운 |
| `src/swing_it/report/layers_md.py` | 층 비교 결과 마크다운 리포트 |
| `src/swing_it/report/forward_log.py` | 전진 기록 append · 사후 채점 |
| `src/swing_it/cli_layers.py` | `swing-layers` — 층별 비교 판정 |
| `src/swing_it/cli_today.py` | `swing-today` — 오늘 후보 카드 + 전진 기록 |
| `pyproject.toml` | **수정** — `[project.scripts]` 두 줄 추가 |
| `tests/` | 위 각 모듈 대응 테스트 |

**다른 계획의 파일을 건드리는 두 곳:** `news/store.py` 확장은 계획 2 소유자가 명시적으로 지시한 것이고(시그니처까지 지정받았다), `pyproject.toml` 은 엔트리포인트 추가다. 그 외 계획 1·2·3 파일은 수정하지 않는다.

## 실행 선행 조건

계획 4 는 계획 2 가 **구현까지** 끝난 뒤에 착수한다. 계획 3 은 이미 구현·머지됐다 — `features/llm.py`·`score/`·`adapters/news_db.py` 가 레포에 있다.

계획 2 의 미구현 잔여는 Task 1(schema)·2(bigkinds)·4(mapping)·7(store + `swing-news` CLI) 네 개다. 그중 **Task 7 이 `news/store.py` 와 `load_daily_news` 를 만든다** — 계획 4 Task 1 의 선행 조건이다. (계획 2 Task 3 은 `adapters/db.py` 의 `load_stock_master` 이고 이미 커밋돼 있다. Task 1 에서 store 를 Task 3 이라고 읽지 마라.) Task 7 실행에는 `swing-news` 가 한 번 돌아 `data/news.db` 가 존재해야 한다.

`pyproject.toml` 의 `[project.scripts]` 에는 이미 `swing-baseline`·`swing-news`·`swing-score` 세 줄이 있다. Task 7 은 거기에 두 줄을 **더한다**(기존 줄은 그대로 둔다).

## 사전등록 상수 (스펙 §6 — 그리드 돌리기 전에 확정)

```
top_n            = 15
n_folds          = 4
min_train        = 200
n_trials         = 200      (음성대조 널 시행 수)
n_boot           = 2000     (페어드 부트스트랩)
cost_one_way     = 0.0023   (왕복 46bp)
hold_end         = 40
adv_floor        = 5000.0
seed             = 0
tv_multiple      = Task 7 Step 1 의 표본 수 실측으로 확정 (3.0 또는 1.0)
PEAD_SHARPE      = 0.80     (kr-quant 검증값, 스펙 §7)
PEAD_T           = 2.20
COMPARABLE_RATIO = 0.75     (PEAD Sharpe 의 75% 이상이면 '비등')
```

---

### Task 1: 규칙 기반 뉴스 피처와 기사 카드 로더

**Files:**
- Create: `src/swing_it/features/news.py`
- Modify: `src/swing_it/adapters/news_db.py` — `load_article_cards` 함수 추가 (파일 끝에 덧붙인다. 기존 `connect_news`·`load_articles` 는 건드리지 않는다)
- Create: `tests/conftest.py` — `news_db` sqlite fixture (두 테스트 파일이 공유한다)
- Test: `tests/test_features_news.py`
- Test: `tests/test_news_db_cards.py`

**Interfaces:**
- Consumes:
  - `swing_it.news.store.load_daily_news(conn, start, end, *, ops_only=False) -> pd.DataFrame` — 컬럼 `[date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, hotness, rank, in_ops, primary_news_id]` (계획 2)
  - `data/news.db` 의 `articles` · `daily_news` 테이블 (계획 2)
- Produces:
  - `swing_it.features.news.NEWS_FEATURE_COLUMNS: list[str]` — `["article_count", "outlet_count", "article_ratio", "hotness", "hotness_rank"]`
  - `swing_it.features.news.news_features(daily_news: pd.DataFrame) -> pd.DataFrame` — 컬럼 `[code, date, *NEWS_FEATURE_COLUMNS]`, 값은 전부 `float`
  - `swing_it.features.news.load_news_features(path: str = "data/news.db", *, start: str = "1900-01-01", end: str = "2999-12-31") -> pd.DataFrame` — sqlite 를 읽기 전용으로 열어 `load_daily_news` → `news_features` 로 넘긴다
  - `swing_it.adapters.news_db.load_article_cards(conn: sqlite3.Connection, start: str, end: str, *, ops_only: bool = False) -> pd.DataFrame` — 컬럼 `[date, code, hotness_rank, hotness, article_id, press, title, body200, url]`

**컬럼 리네임 (계획 2 `daily_news` → 피처 이름):** `n_articles → article_count`, `n_outlets → outlet_count`, `rank → hotness_rank`. `article_ratio`·`hotness` 는 그대로.

`rank` 를 개명하는 이유는 pandas `DataFrame.rank` 메서드와 SQL 예약어 양쪽에서 충돌하기 때문이고, `n_*` 을 `*_count` 로 바꾸는 이유는 근거 카드에 컬럼명이 그대로 찍히므로 사람이 읽어서 뜻이 통해야 하기 때문이다.

**`daily_news` 컬럼 중 무엇을 피처로 쓰는지 전부 명시한다.** 이 표에서 `피처` 인 5개만 `news_columns` 에 들어간다. 나머지를 "프레임에 있으니까" 밀어넣으면 안 된다.

| `daily_news` 컬럼 | 처리 | 이유 |
|---|---|---|
| `n_articles` | **피처** → `article_count` | 기사 건수. 순수 뉴스 축 |
| `n_outlets` | **피처** → `outlet_count` | 보도 매체 수. 순수 뉴스 축 |
| `article_ratio` | **피처** | 평소 대비 급증 배수 (계획 2 가 60일 baseline·shift(1) 로 계산 완료) |
| `hotness` | **피처** | 위 셋과 거래대금의 가중 합성 |
| `rank` | **피처** → `hotness_rank` | 그날의 핫함 순위. 스펙 §5 — 핫함 순위 자체가 피처가 된다 |
| `tv_ratio` | **제외** | 계획 1 `FEATURE_COLUMNS` 에 **이미 있다**. 아래 참조 |
| `adv` | **제외** | 계획 1 `log_adv` 가 같은 값에서 나온다 |
| `code`·`date` | 조인 키 | 피처가 아니다 |
| `in_train` | **제외** | 표본 선택 플래그 (`load_daily_news` 는 애초에 저장하지 않는다) |
| `in_ops` | **제외** | 표본 선택 플래그. 운영 상위 10 표시용 |
| `primary_news_id` | **제외** | 근거 카드 조인 키 (`load_article_cards` 가 쓴다) |

**`tv_ratio`·`adv` 제외는 취향이 아니라 필수다.** 계획 1 `FEATURE_COLUMNS` 가 `tv_ratio` 를 이미 갖고 있어서, 뉴스 프레임에 남겨두면 `merge(on=["code","date"])` 가 `tv_ratio_x`/`tv_ratio_y` 를 만들고 `FEATURE_COLUMNS` 조회가 `KeyError` 로 죽는다. 죽지 않더라도 ②층에 같은 거래대금 축을 두 번 넣는 셈이라 "규칙 뉴스 피처가 기여했다"가 거짓이 된다. `news_features` 가 `_SOURCE` 에 없는 컬럼을 통째로 버리므로 구조적으로 막혀 있다.

**피처 예산 (스펙 §6, 10~20개):** 계획 1 의 7개 + 뉴스 5개 + LLM 5개 = **17개**. 상한 안에 딱 들어간다. 여기에 `material_type` 원-핫 9개나 LLM `n_articles` 를 더하면 넘친다.

**`load_article_cards` 가 집계 수치를 안 돌려주는 이유:** `article_count`·`outlet_count`·`article_ratio` 는 이미 ②층 **피처**라서 후보 행에 실려 있다. 카드는 거기서 읽는다. 같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다.

**`published_at` 은 없다.** 빅카인즈는 일자 단위(`YYYYMMDD`)만 주고 발행 시각을 주지 않는다. 운영(krx-news-rest-api)에는 시각이 있지만 학습/운영 스키마를 갈라놓으면 계획 2 의 200자 통일 결정과 같은 이유로 깨진다. 카드에 시각을 찍지 않는다.

**롤링을 다시 하지 않는 이유:** 계획 2 `add_hotness` 가 60일 baseline 을 `shift(1)` 로 이미 계산했다. `article_ratio` 와 `hotness` 는 그 결과물이다. 여기서 또 롤링하면 이중 계산이고 point-in-time 도 흐려진다. 그래서 `news_features` 에는 `window` 류 파라미터가 **없다.**

- [ ] **Step 1: 실패하는 뉴스 피처 테스트 작성**

`tests/test_features_news.py`:

```python
import pandas as pd
import pytest

from swing_it.features.news import NEWS_FEATURE_COLUMNS, news_features


def _daily_news(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14"] * n),
            "code": [f"{i:06d}" for i in range(n)],
            "n_articles": [7, 3, 1],
            "n_outlets": [5, 2, 1],
            "article_ratio": [4.2, 1.8, 0.9],
            "adv": [8420.0, 6100.0, 5200.0],
            "tv_ratio": [4.6, 2.1, 1.2],
            "hotness": [0.81, 0.44, 0.20],
            "rank": [1, 2, 3],
            "in_ops": [True, True, False],
            "primary_news_id": ["n1", "n2", "n3"],
        }
    )


def test_registered_feature_columns():
    assert NEWS_FEATURE_COLUMNS == [
        "article_count", "outlet_count", "article_ratio", "hotness", "hotness_rank"
    ]


def test_output_columns_and_order():
    out = news_features(_daily_news())
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]


def test_source_columns_are_renamed():
    out = news_features(_daily_news())
    assert out["article_count"].tolist() == [7.0, 3.0, 1.0]
    assert out["outlet_count"].tolist() == [5.0, 2.0, 1.0]
    assert out["hotness_rank"].tolist() == [1.0, 2.0, 3.0]
    for stale in ("n_articles", "n_outlets", "rank"):
        assert stale not in out.columns


def test_trade_value_columns_are_excluded():
    """tv_ratio·adv 는 계획 1 FEATURE_COLUMNS 에 이미 있다. 중복 투입 금지."""
    out = news_features(_daily_news())
    assert "tv_ratio" not in out.columns
    assert "adv" not in out.columns


def test_bookkeeping_columns_are_excluded():
    out = news_features(_daily_news())
    assert "in_ops" not in out.columns
    assert "primary_news_id" not in out.columns


def test_row_count_is_preserved():
    out = news_features(_daily_news(3))
    assert len(out) == 3


def test_all_feature_values_are_float():
    out = news_features(_daily_news())
    for col in NEWS_FEATURE_COLUMNS:
        assert out[col].dtype == "float64"


def test_date_is_datetime():
    out = news_features(_daily_news())
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_missing_source_column_fails_loudly():
    bad = _daily_news().drop(columns=["hotness"])
    with pytest.raises(ValueError, match="hotness"):
        news_features(bad)


def test_empty_input_returns_empty_frame_with_columns():
    empty = _daily_news().iloc[0:0]
    out = news_features(empty)
    assert out.empty
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]


def test_load_news_features_reads_sqlite(news_db):
    from swing_it.features.news import load_news_features

    out = load_news_features(news_db, start="2026-08-14", end="2026-08-14")
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]
    assert set(out["code"]) == {"034020", "005930", "000660"}
    assert out[out["code"] == "034020"].iloc[0]["article_count"] == 7.0


def test_load_news_features_fails_when_the_db_is_missing(tmp_path):
    from swing_it.features.news import load_news_features

    with pytest.raises(FileNotFoundError, match="뉴스 DB 가 없다"):
        load_news_features(str(tmp_path / "nope.db"), start="2026-01-01", end="2026-12-31")
```

`news_db` fixture 는 `tests/test_news_db_cards.py` 와 같은 것을 쓴다. 두 파일이 공유하도록 **`tests/conftest.py`** 로 옮긴다:

```python
import sqlite3

import pytest


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_features_news.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.features.news'`

- [ ] **Step 3: 뉴스 피처 구현**

`src/swing_it/features/news.py`:

```python
"""규칙 기반 뉴스 피처 — LLM 없이 기사에서 뽑는 것 (스펙 §3 ②층).

계획 2 의 ``daily_news`` 를 모델이 먹을 수 있는 형태로 고르고 이름만 바꾼다.
**계산은 거의 하지 않는다.** 60일 baseline 롤링과 shift(1) 은 계획 2 의
``add_hotness`` 가 이미 끝냈다. 여기서 또 롤링하면 이중 계산이고
point-in-time 보증도 흐려진다. 그래서 ``window`` 류 파라미터가 없다.

빠뜨린 컬럼이 둘 있고, 둘 다 의도적이다:

- ``tv_ratio``·``adv`` — 계획 1 ``FEATURE_COLUMNS`` 에 이미 들어 있다. ②층에
  다시 넣으면 "뉴스 피처가 기여했다"가 사실은 "거래대금을 두 번 넣었다"일 수
  있고, 층별 기여도 분리(스펙 §3)가 무너진다.
- ``in_ops``·``primary_news_id`` — 운영 표시와 조인 키다. 피처가 아니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_FEATURE_COLUMNS: list[str] = [
    "article_count",
    "outlet_count",
    "article_ratio",
    "hotness",
    "hotness_rank",
]

# daily_news 원본 컬럼 → 피처 이름.
# rank 는 pandas 메서드·SQL 예약어와 충돌하고, n_* 은 카드에 찍혔을 때 뜻이 덜 통한다.
_SOURCE = {
    "article_count": "n_articles",
    "outlet_count": "n_outlets",
    "article_ratio": "article_ratio",
    "hotness": "hotness",
    "hotness_rank": "rank",
}


def news_features(daily_news: pd.DataFrame) -> pd.DataFrame:
    """``load_daily_news`` 결과를 ``[code, date, *NEWS_FEATURE_COLUMNS]`` 로 바꾼다."""
    required = ["code", "date", *_SOURCE.values()]
    missing = [col for col in required if col not in daily_news.columns]
    if missing:
        raise ValueError(
            f"daily_news 에 컬럼이 없다: {missing}. 계획 2 의 load_daily_news 출력을 넘겨라."
        )

    out = pd.DataFrame(
        {
            "code": daily_news["code"].astype(str),
            "date": pd.to_datetime(daily_news["date"]),
        }
    )
    for feature, source in _SOURCE.items():
        out[feature] = pd.to_numeric(daily_news[source], errors="coerce").astype(float)
    return out[["code", "date", *NEWS_FEATURE_COLUMNS]].reset_index(drop=True)


def load_news_features(
    path: str | Path = "data/news.db",
    *,
    start: str = "1900-01-01",
    end: str = "2999-12-31",
) -> pd.DataFrame:
    """계획 2 의 sqlite 를 열어 뉴스 피처를 만든다.

    ``adapters.news_db.connect_news`` 를 재사용한다. ``init_db`` 를 부르면 파일이
    없을 때 빈 스키마를 만들어 조용히 0행을 돌려주고, 뉴스 층이 통째로 빠진 채로
    비교가 돌아가고도 아무도 모른다. ``connect_news`` 는 없으면 즉시 실패한다.
    """
    from ..adapters.news_db import connect_news
    from ..news.store import load_daily_news

    conn = connect_news(str(path))
    try:
        return news_features(load_daily_news(conn, start, end))
    finally:
        conn.close()
```

- [ ] **Step 4: 뉴스 피처 테스트 통과 확인**

Run: `uv run pytest tests/test_features_news.py -v`
Expected: 12 passed

- [ ] **Step 5: 실패하는 기사 카드 로더 테스트 작성**

`tests/test_news_db_cards.py`:

`news_db` fixture 는 위 Step 1 에서 만든 `tests/conftest.py` 의 것을 쓴다.

```python
import pandas as pd
import pytest

from swing_it.adapters.news_db import connect_news, load_article_cards


def test_columns_and_order(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert list(df.columns) == [
        "date", "code", "hotness_rank", "hotness",
        "article_id", "press", "title", "body200", "url",
    ]


def test_column_names_match_the_sibling_loader(news_db):
    """같은 어댑터의 load_articles 와 같은 이름을 쓴다 — 호출부가 헷갈리지 않게."""
    from swing_it.adapters.news_db import NEWS_COLUMNS

    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    for shared in ("article_id", "date", "code", "press", "title", "body200"):
        assert shared in NEWS_COLUMNS
        assert shared in df.columns


def test_joins_the_representative_article(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    row = df[df["code"] == "034020"].iloc[0]
    assert row["title"] == "체코 원전 2기 최종 계약"
    assert row["press"] == "한국경제"
    assert row["url"] == "https://x/1"
    assert "두코바니" in row["body200"]


def test_row_without_a_primary_article_is_kept_with_nulls(news_db):
    """대표 기사가 없어도 핫함 수치는 카드에 쓴다. LEFT JOIN 이어야 한다."""
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    row = df[df["code"] == "000660"].iloc[0]
    assert pd.isna(row["article_id"])
    assert row["hotness"] == 0.20


def test_rank_is_renamed_to_hotness_rank(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert "rank" not in df.columns
    assert df[df["code"] == "034020"].iloc[0]["hotness_rank"] == 1


def test_aggregate_counts_are_not_returned(news_db):
    """집계 수치는 ②층 피처로 후보 행에 이미 실려 있다. 두 경로로 들여오지 않는다."""
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    for stale in ("n_articles", "n_outlets", "article_count", "article_ratio"):
        assert stale not in df.columns


def test_date_range_is_respected(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert df["date"].dt.date.astype(str).unique().tolist() == ["2026-08-14"]
    assert len(df) == 3


def test_ops_only_filters_to_the_operating_universe(news_db):
    df = load_article_cards(
        connect_news(news_db), "2026-08-14", "2026-08-14", ops_only=True
    )
    assert df["code"].tolist() == ["034020"]


def test_date_is_datetime(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-01", "2026-08-31")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 4


def test_connect_news_fails_loudly_on_a_missing_db(tmp_path):
    """별도 실패 경로를 만들지 않는다. connect_news 가 이미 한다."""
    with pytest.raises(FileNotFoundError, match="뉴스 DB 가 없다"):
        connect_news(str(tmp_path / "nope.db"))
```

- [ ] **Step 6: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_db_cards.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_article_cards' from 'swing_it.adapters.news_db'`

- [ ] **Step 7: `load_article_cards` 를 `news_db.py` 끝에 추가**

계획 2 가 만든 기존 함수는 건드리지 않는다. 파일 맨 끝에 다음을 덧붙인다.

```python
CARD_COLUMNS: list[str] = [
    "date", "code", "hotness_rank", "hotness",
    "article_id", "press", "title", "body200", "url",
]

_SELECT_CARDS = """
select d.date, d.code, d."rank", d.hotness,
       a.news_id, a.outlet, a.title, a.body, a.url
from daily_news d
left join articles a on a.news_id = d.primary_news_id
where d.date >= ? and d.date <= ?
"""


def load_article_cards(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    *,
    ops_only: bool = False,
) -> pd.DataFrame:
    """근거 카드용 — 일별 핫함에 그날의 대표 기사를 붙여 읽는다.

    ``daily_news.primary_news_id → articles.news_id`` 를 LEFT JOIN 한다.
    ``articles`` 에는 ``code`` 가 없다 — 기사 1건이 종목 N개에 매핑되므로
    종목은 ``daily_news`` 를 거쳐야 나온다. 근거 카드는 (종목, 날짜)당 대표
    기사 한 건이면 되므로 이 조인이 맞다.

    대표 기사가 없는 행도 남긴다. 핫함 수치만으로도 카드에 쓸 값이 있고,
    조인 실패로 후보가 조용히 사라지는 것이 더 나쁘다.

    집계 수치(``n_articles`` 등)는 돌려주지 않는다. ②층 피처로 후보 행에
    이미 실려 있고, 같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다.

    ``rank`` 는 SQL 예약어이므로 ``"rank"`` 로 인용하고, 반환할 때
    ``hotness_rank`` 로 개명한다 (``features.news`` 와 같은 이름).

    이 모듈의 기존 ``load_articles`` 와 컬럼 계약을 맞춘다 —
    ``news_id``→``article_id``, ``outlet``→``press``, ``body``→``body200``.
    한 어댑터 안에서 두 함수가 다른 이름을 쓰면 호출부가 헷갈린다.
    ``url`` 은 ``NEWS_COLUMNS`` 에 없지만 근거 카드 링크에 필요해 여기서만 더 뽑는다.
    """
    sql = _SELECT_CARDS + (" and d.in_ops = 1" if ops_only else "")
    sql += ' order by d.date, d."rank"'

    rows = conn.execute(sql, (start, end)).fetchall()
    raw = pd.DataFrame(rows, columns=CARD_COLUMNS)
    if raw.empty:
        return pd.DataFrame(columns=CARD_COLUMNS)

    raw["date"] = pd.to_datetime(raw["date"])
    raw["code"] = raw["code"].astype(str)
    for col in ("hotness_rank", "hotness"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw[CARD_COLUMNS].reset_index(drop=True)
```

`news_db.py` 는 이미 `sqlite3`·`pandas`·`Path` 를 임포트하고 있다. 추가 임포트는 필요 없다.

**파일이 없을 때의 실패는 `connect_news` 가 이미 처리한다** — `FileNotFoundError("뉴스 DB 가 없다: ...")`. 별도 경로를 만들지 않고 호출부에서 `connect_news` 를 쓴다.

- [ ] **Step 8: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_db_cards.py tests/test_features_news.py -v`
Expected: 22 passed

- [ ] **Step 9: 계획 2 회귀 확인**

Run: `uv run pytest tests/test_news_store.py tests/test_hotness.py -v`
Expected: 계획 2·3 테스트 전부 통과 (`load_article_cards` 는 추가일 뿐 `connect_news`·`load_articles` 를 안 건드렸음을 확인)

- [ ] **Step 10: 커밋**

```bash
git add src/swing_it/features/news.py \
  src/swing_it/adapters/news_db.py tests/conftest.py \
  tests/test_features_news.py tests/test_news_db_cards.py
git commit -m "feat(features): 규칙 기반 뉴스 피처 + 근거 카드용 기사 조인 로더"
```

---

### Task 2: 층 정의와 층별 단일 데이터셋

**Files:**
- Create: `src/swing_it/layers.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Consumes:
  - `swing_it.dataset.FEATURE_COLUMNS: list[str]`
  - `swing_it.dataset.build_dataset(prices, supply_demand, *, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, fwd_return, excess_return]`
  - `swing_it.features.news.NEWS_FEATURE_COLUMNS`, `news_features(daily_news) -> pd.DataFrame` (Task 1)
  - 계획 3 의 `llm` 프레임 `[code, date, *LLM_FEATURE_COLUMNS]`
- Produces:
  - `LAYER_NAMES: list[str]` — `["supply", "supply_news", "supply_news_llm"]` (고정 순서 = 누적 순서)
  - `LAYER_LABELS: dict[str, str]` — `{"supply": "① 수급 단독", "supply_news": "② + 규칙 뉴스", "supply_news_llm": "③ + LLM"}`
  - `layer_features(news_columns: Sequence[str] = (), llm_columns: Sequence[str] = ()) -> dict[str, list[str]]` — 층 이름 → 그 층이 쓰는 피처 컬럼 목록(누적)
  - `build_layered_dataset(prices, supply_demand, *, news=None, news_columns=(), llm=None, llm_columns=(), require_label=True, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, *news_columns, *llm_columns, fwd_return, excess_return]`

`require_label=False` 는 오늘 후보 추론용이다. 라벨이 아직 존재할 수 없는 최근 행을 남긴다. 피처 결측은 그때도 허용하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_layers.py`:

```python
import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.layers import (
    LAYER_LABELS,
    LAYER_NAMES,
    build_layered_dataset,
    layer_features,
)

NEWS_COLS = ["article_count", "hotness"]
LLM_COLS = ["llm_specificity", "llm_persistence"]


def _prices(codes: list[str], n: int, spike_at: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "open": 100.0 + i,
                    "close": 100.0 + i,
                    "trade_value": 6000.0 * (5.0 if i == spike_at else 1.0),
                }
            )
    return pd.DataFrame(rows)


def _supply(codes: list[str], n: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "individual": -100,
                    "foreign_": 60,
                    "institution": 40,
                    "penfnd_etc": 10,
                    "acc_trde_qty": 1000,
                }
            )
    return pd.DataFrame(rows)


def _side(codes: list[str], n: int, columns: list[str], value: float) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            row = {
                "code": code,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
            row.update({col: value for col in columns})
            rows.append(row)
    return pd.DataFrame(rows)


def _built(codes, *, spike_at=25, news=None, llm=None, **kwargs):
    n = kwargs.pop("n", 80)
    return build_layered_dataset(
        _prices(codes, n, spike_at=spike_at),
        _supply(codes, n),
        news=_side(codes, n, NEWS_COLS, 2.0) if news is None else news,
        news_columns=NEWS_COLS,
        llm=_side(codes, n, LLM_COLS, 0.5) if llm is None else llm,
        llm_columns=LLM_COLS,
        hold_start=10,
        hold_end=20,
        **kwargs,
    )


def test_layers_are_cumulative():
    spec = layer_features(NEWS_COLS, LLM_COLS)
    assert list(spec) == LAYER_NAMES
    assert spec["supply"] == FEATURE_COLUMNS
    assert spec["supply_news"] == [*FEATURE_COLUMNS, *NEWS_COLS]
    assert spec["supply_news_llm"] == [*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]


def test_layer_features_defaults_to_supply_only():
    spec = layer_features()
    assert spec["supply"] == spec["supply_news"] == spec["supply_news_llm"]


def test_layer_labels_cover_every_layer():
    assert set(LAYER_LABELS) == set(LAYER_NAMES)


def test_layered_dataset_column_order():
    ds = _built(["A", "B"])
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS,
        "fwd_return", "excess_return",
    ]


def test_every_layer_sees_identical_rows():
    """세 층이 같은 행 위에서 돌지 않으면 층 비교는 아무 의미가 없다."""
    ds = _built(["A", "B"])
    for columns in layer_features(NEWS_COLS, LLM_COLS).values():
        assert ds[columns].notna().all().all()


def test_missing_llm_row_is_dropped_from_all_layers():
    codes = ["A", "B"]
    llm = _side(codes, 80, LLM_COLS, 0.5)
    spike_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=25)
    mask = (llm["code"] == "B") & (llm["date"] == spike_date)
    llm.loc[mask, LLM_COLS] = np.nan
    ds = _built(codes, llm=llm)
    assert "B" not in set(ds["code"])


def test_event_without_a_news_row_is_dropped():
    """뉴스 결측은 채우지 않는다. 표본 정의는 '세 층 피처가 모두 있는 행' 하나뿐이다."""
    codes = ["A", "B"]
    news = _side(["A"], 80, NEWS_COLS, 2.0)  # B 의 뉴스 행이 아예 없다
    ds = _built(codes, news=news)
    assert set(ds["code"]) == {"A"}


def test_require_label_false_keeps_unlabelled_recent_rows():
    ds = _built(["A", "B"], spike_at=78, require_label=False)
    assert len(ds) == 2
    assert ds["excess_return"].isna().all()
    assert ds[[*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]].notna().all().all()


def test_require_label_true_drops_those_same_rows():
    ds = _built(["A", "B"], spike_at=78, require_label=True)
    assert ds.empty


def test_no_side_frames_reduces_to_baseline_columns():
    codes = ["A", "B"]
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25), _supply(codes, 80), hold_start=10, hold_end=20
    )
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"
    ]


def test_side_frame_missing_a_declared_column_fails_loudly():
    import pytest

    codes = ["A"]
    with pytest.raises(ValueError, match="hotness"):
        build_layered_dataset(
            _prices(codes, 80, spike_at=25),
            _supply(codes, 80),
            news=_side(codes, 80, ["article_count"], 2.0),
            news_columns=NEWS_COLS,
            hold_start=10,
            hold_end=20,
        )


def test_rows_are_sorted_by_date_then_code():
    ds = _built(["B", "A"])
    assert ds[["date", "code"]].equals(
        ds[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True)
    )
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_layers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.layers'`

- [ ] **Step 3: 구현**

`src/swing_it/layers.py`:

```python
"""층 정의와 층별 단일 데이터셋 조립 (스펙 §3).

핵심은 하나다: **세 층이 같은 행 위에서 돌아야 한다.** 층마다 표본이 다르면
"③이 ②를 이겼다"는 문장이 아무것도 말해주지 않는다. 그래서 층을 서로 다른
데이터셋으로 만들지 않고, 세 층 피처를 전부 붙인 하나의 테이블을 만든 뒤
층은 그 위에서 쓰는 **컬럼의 부분집합**으로만 정의한다.

결측 채우기 규칙은 두지 않는다. 규칙은 하나다 — 어느 층 피처든 결측이면
그 행은 세 층 모두에서 빠진다. 채움 규칙을 두는 순간 "언제 채웠고 언제
떨궜나"를 추적해야 하고, 그 복잡도가 값을 하지 않는다.

계획 1 의 ``build_dataset`` 시그니처는 건드리지 않는다. 여기서 감싸 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .dataset import FEATURE_COLUMNS, build_dataset

LAYER_NAMES: list[str] = ["supply", "supply_news", "supply_news_llm"]

LAYER_LABELS: dict[str, str] = {
    "supply": "① 수급 단독",
    "supply_news": "② + 규칙 뉴스",
    "supply_news_llm": "③ + LLM",
}


def layer_features(
    news_columns: Sequence[str] = (),
    llm_columns: Sequence[str] = (),
) -> dict[str, list[str]]:
    """층 이름 → 그 층이 쓰는 피처 컬럼 목록(누적)."""
    supply = list(FEATURE_COLUMNS)
    news = [*supply, *news_columns]
    return {
        "supply": supply,
        "supply_news": news,
        "supply_news_llm": [*news, *llm_columns],
    }


def _merge_side(
    base: pd.DataFrame,
    side: pd.DataFrame | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """(code, date) 로 보조 피처 프레임을 왼쪽 조인한다."""
    if not columns:
        return base
    if side is None:
        side = pd.DataFrame(columns=["code", "date", *columns])
    missing = [c for c in ["code", "date", *columns] if c not in side.columns]
    if missing:
        raise ValueError(f"보조 프레임에 컬럼이 없다: {missing}")
    frame = side[["code", "date", *columns]].copy()
    frame["code"] = frame["code"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    out = base.merge(frame, on=["code", "date"], how="left")
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_layered_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    news: pd.DataFrame | None = None,
    news_columns: Sequence[str] = (),
    llm: pd.DataFrame | None = None,
    llm_columns: Sequence[str] = (),
    require_label: bool = True,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """세 층의 피처를 모두 붙인 단일 학습 테이블.

    피처가 하나라도 결측인 행은 층과 무관하게 전부 떨군다. ③층에서만 쓰는
    LLM 피처가 없다는 이유로 ①층이 그 행을 쓰면 표본이 어긋난다.

    Args:
        require_label: ``True`` 면 라벨 결측 행을 떨군다(학습·검증용).
            ``False`` 면 남긴다 — 오늘 후보 추론용으로, 아직 D+40 이 오지 않아
            라벨이 존재할 수 없는 최근 이벤트를 살린다.
    """
    common = dict(
        adv_floor=adv_floor,
        tv_multiple=tv_multiple,
        entry_offset=entry_offset,
        hold_start=hold_start,
        hold_end=hold_end,
    )
    base = (
        build_dataset(prices, supply_demand, **common)
        if require_label
        else _unlabelled_base(prices, supply_demand, **common)
    )

    out = _merge_side(base, news, list(news_columns))
    out = _merge_side(out, llm, list(llm_columns))

    feature_columns = [*FEATURE_COLUMNS, *news_columns, *llm_columns]
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_columns)
    if require_label:
        out = out.dropna(subset=["excess_return"])

    columns = ["code", "date", *feature_columns, "fwd_return", "excess_return"]
    return out[columns].sort_values(["date", "code"]).reset_index(drop=True)


def _unlabelled_base(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    adv_floor: float,
    tv_multiple: float,
    entry_offset: int,
    hold_start: int,
    hold_end: int,
) -> pd.DataFrame:
    """라벨 없는 이벤트까지 포함한 베이스 테이블.

    ``build_dataset`` 과 같은 피처를 만들되 ``dropna`` 를 라벨에 걸지 않는다.
    계획 1 의 함수를 수정하지 않기 위해 조립 순서만 여기서 다시 쓴다.
    """
    from .events import extract_events
    from .features.supply import supply_features
    from .labels import excess_return, forward_return

    columns = ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]
    events = extract_events(prices, adv_floor=adv_floor, tv_multiple=tv_multiple)
    if events.empty:
        return pd.DataFrame(columns=columns)

    supply = supply_features(supply_demand)
    labels = excess_return(
        forward_return(
            prices,
            entry_offset=entry_offset,
            hold_start=hold_start,
            hold_end=hold_end,
        )
    )
    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    # 진입가가 아직 없는 행(달력 마지막 날 이벤트)은 이벤트일 종가로 대신한다.
    # 스케일 분모로만 쓰이므로 근사로 충분하고, 라벨에는 영향이 없다.
    df = df.merge(prices[["code", "date", "close"]], on=["code", "date"], how="left")
    shares_capacity = (df["adv"] * 1e6) / df["entry_price"].fillna(df["close"])
    for scaled, raw in (
        ("smart_money_net_5d_scaled", "smart_money_net_5d"),
        ("indiv_net_5d_scaled", "indiv_net_5d"),
        ("pension_net_5d_scaled", "pension_net_5d"),
    ):
        df[scaled] = df[raw] / shares_capacity
    df["log_adv"] = np.log10(df["adv"])
    return df[columns]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_layers.py -v`
Expected: 12 passed

- [ ] **Step 5: 계획 1 회귀 확인**

Run: `uv run pytest tests/test_dataset.py tests/test_events.py tests/test_labels.py -v`
Expected: 전부 통과 (계획 1 의 시그니처를 건드리지 않았음을 확인한다)

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/layers.py tests/test_layers.py
git commit -m "feat(layers): 세 층이 동일 표본을 쓰는 단일 데이터셋 조립"
```

---
### Task 3: 피처 목록을 받는 walk-forward 와 공유 폴드

**Files:**
- Create: `src/swing_it/model/layered.py`
- Test: `tests/test_model_layered.py`

**Interfaces:**
- Consumes:
  - `swing_it.dataset.FEATURE_COLUMNS`
  - `swing_it.model.baseline.walk_forward_predict(ds, *, n_folds=4, min_train=200, seed=0) -> pd.DataFrame` — 동등성 검증용
- Produces:
  - `fold_dates(ds: pd.DataFrame, *, n_folds: int = 4, min_train: int = 200) -> list[np.ndarray]` — 폴드별 검증 날짜 배열. **한 번 계산해 모든 층에 넘긴다.**
  - `walk_forward_predict_features(ds: pd.DataFrame, feature_columns: Sequence[str], *, folds: list[np.ndarray] | None = None, n_folds: int = 4, min_train: int = 200, seed: int = 0) -> pd.DataFrame` — 컬럼 `[code, date, excess_return, prediction, fold]`
  - `fit_final_model(ds: pd.DataFrame, feature_columns: Sequence[str], *, seed: int = 0) -> tuple[GradientBoostingRegressor, pd.Series]` — 전 구간 적합 모델과 **학습셋 피처 중앙값**(기여도 계산 기준점)

`fold_dates` 를 분리하는 이유가 이 태스크의 전부다. 층마다 폴드를 따로 계산하면 표본이 같아도 경계가 미세하게 달라질 수 있고, 그러면 "③이 ②를 이겼다"가 폴드 차이일 수 있다.

폴드 규칙은 계획 1 `walk_forward_predict` 를 글자 그대로 옮긴다: `min_train` 번째 **행**의 날짜를 구하고, 그보다 **큰** 날짜들을 `np.array_split` 으로 `n_folds` 등분한다. 각 폴드는 그 폴드 첫 날짜 **이전** 데이터로만 학습한다.

모델은 `GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=seed)` 로 고정한다. 층이 늘어도 여기서 올리지 않는다(스펙 §6).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_model_layered.py`:

```python
import numpy as np
import pandas as pd
import pytest

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.model.baseline import walk_forward_predict
from swing_it.model.layered import (
    fit_final_model,
    fold_dates,
    walk_forward_predict_features,
)

EXTRA = ["article_count", "llm_specificity"]


def _dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in [*FEATURE_COLUMNS, *EXTRA]})
    df["code"] = [f"{i % 40:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    df["excess_return"] = df[FEATURE_COLUMNS[0]] * 0.05 + rng.normal(scale=0.01, size=n)
    return df[["code", "date", *FEATURE_COLUMNS, *EXTRA, "excess_return"]]


def test_matches_plan1_walk_forward_exactly():
    """계획 1 과의 호환 계약. 이게 깨지면 층 비교의 ①층이 베이스라인이 아니다."""
    ds = _dataset(1200)
    expected = walk_forward_predict(ds, n_folds=4, min_train=200, seed=0)
    got = walk_forward_predict_features(
        ds, FEATURE_COLUMNS, n_folds=4, min_train=200, seed=0
    )
    pd.testing.assert_frame_equal(expected, got)


def test_extra_features_are_actually_used():
    ds = _dataset(1200)
    ds["planted"] = ds["excess_return"] * 3.0  # 라벨을 거의 그대로 담은 피처
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, "planted"])
    assert b["prediction"].corr(b["excess_return"]) > a["prediction"].corr(a["excess_return"])


def test_fold_dates_are_contiguous_and_ordered():
    folds = fold_dates(_dataset(1200), n_folds=4, min_train=200)
    assert len(folds) == 4
    for i in range(1, len(folds)):
        assert folds[i][0] > folds[i - 1][-1]


def test_fold_dates_start_after_the_min_train_row():
    ds = _dataset(1200).sort_values(["date", "code"]).reset_index(drop=True)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    assert folds[0][0] > ds.loc[200, "date"]


def test_shared_folds_give_identical_fold_boundaries_across_feature_sets():
    ds = _dataset(1200)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS, folds=folds)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, *EXTRA], folds=folds)
    pd.testing.assert_frame_equal(
        a[["code", "date", "excess_return", "fold"]],
        b[["code", "date", "excess_return", "fold"]],
    )


def test_never_trains_on_the_dates_it_predicts():
    ds = _dataset(1200)
    pred = walk_forward_predict_features(ds, FEATURE_COLUMNS)
    bounds = pred.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    for i in range(1, len(bounds)):
        assert bounds.iloc[i]["min"] > bounds.iloc[i - 1]["max"]


def test_too_small_a_sample_fails_loudly():
    with pytest.raises(ValueError, match="min_train"):
        fold_dates(_dataset(50), n_folds=4, min_train=200)


def test_fit_final_model_returns_model_and_median_baseline():
    ds = _dataset(600)
    model, baseline = fit_final_model(ds, [*FEATURE_COLUMNS, *EXTRA])
    assert list(baseline.index) == [*FEATURE_COLUMNS, *EXTRA]
    assert np.isclose(baseline["article_count"], ds["article_count"].median())
    assert model.predict(ds[[*FEATURE_COLUMNS, *EXTRA]].head(3)).shape == (3,)


def test_model_stays_shallow():
    """스펙 §6 — 층이 늘어도 모델은 얕게 고정한다."""
    model, _ = fit_final_model(_dataset(600), FEATURE_COLUMNS)
    assert model.max_depth == 2
    assert model.n_estimators == 100
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_model_layered.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.model.layered'`

- [ ] **Step 3: 구현**

`src/swing_it/model/layered.py`:

```python
"""피처 목록을 인자로 받는 walk-forward — 층 비교의 실행부.

계획 1 의 ``walk_forward_predict`` 는 ``FEATURE_COLUMNS`` 를 고정으로 쓴다.
그 시그니처를 깨지 않기 위해 여기에 확장판을 따로 둔다. 두 함수가 같은 결과를
내는지는 ``tests/test_model_layered.py::test_matches_plan1_walk_forward_exactly``
가 고정한다.

``fold_dates`` 를 분리한 것이 이 모듈의 요점이다. 폴드 경계를 한 번 계산해
세 층에 그대로 넘겨야 "③이 ②를 이겼다"가 폴드 차이가 아님을 보증할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

MAX_DEPTH = 2
N_ESTIMATORS = 100


def fold_dates(
    ds: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 200,
) -> list[np.ndarray]:
    """검증 구간을 시간순 ``n_folds`` 등분한 날짜 배열 목록.

    계획 1 ``walk_forward_predict`` 와 글자 그대로 같은 규칙이다. 폴드 경계는
    날짜 단위로 끊는다 — 같은 날짜가 학습과 예측에 동시에 들어가면 같은 날
    다른 종목의 결과가 새어 들어온다.
    """
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    if len(df) <= min_train:
        raise ValueError(f"표본이 {len(df)} 행뿐이라 min_train={min_train} 을 못 채운다")

    first_test_date = df.loc[min_train, "date"]
    test_dates = dates[dates > first_test_date]
    if len(test_dates) < n_folds:
        raise ValueError(f"검증 가능한 날짜가 {len(test_dates)} 개뿐이라 {n_folds} 폴드를 못 만든다")
    return list(np.array_split(test_dates, n_folds))


def walk_forward_predict_features(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    folds: list[np.ndarray] | None = None,
    n_folds: int = 4,
    min_train: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """지정한 피처만 써서 시간순 폴드로 학습·예측한다.

    Args:
        folds: ``fold_dates`` 결과. 층 비교에서는 **반드시** 미리 계산해 넘긴다.
            생략하면 이 함수가 직접 계산한다(단일 층 실행용).
    """
    columns = list(feature_columns)
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    if folds is None:
        folds = fold_dates(df, n_folds=n_folds, min_train=min_train)

    results = []
    for fold, chunk in enumerate(folds):
        train = df[df["date"] < chunk[0]]
        test = df[df["date"].isin(chunk)]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(
            max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
        )
        model.fit(train[columns], train["excess_return"])
        block = test[["code", "date", "excess_return"]].copy()
        block["prediction"] = model.predict(test[columns])
        block["fold"] = fold
        results.append(block)

    return pd.concat(results, ignore_index=True)


def fit_final_model(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[GradientBoostingRegressor, pd.Series]:
    """전 구간으로 적합한 모델과 피처 중앙값을 함께 돌려준다.

    중앙값은 근거 카드의 기여도 계산에서 '평범한 이벤트' 기준점으로 쓴다
    (Task 6). 모델과 함께 나와야 기준점이 학습 분포와 어긋나지 않는다.
    """
    columns = list(feature_columns)
    model = GradientBoostingRegressor(
        max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
    )
    model.fit(ds[columns], ds["excess_return"])
    return model, ds[columns].median()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_model_layered.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/model/layered.py tests/test_model_layered.py
git commit -m "feat(model): 피처 지정 walk-forward + 층 간 공유 폴드 경계"
```

---

### Task 4: 층 하나의 종합 판정 — 진단 4종 · 하방 회피 · 선별도

**Files:**
- Create: `src/swing_it/validate/layer_verdict.py`
- Test: `tests/test_layer_verdict.py`

**Interfaces:**
- Consumes:
  - `swing_it.validate.gate.gate_report(pred, *, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict`
  - `swing_it.validate.diagnostics.fold_null_table(pred, *, top_n=15, n_trials=200, seed=0) -> pd.DataFrame`
  - `swing_it.validate.diagnostics.untouched_window(pred, *, fraction=0.25, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict`
  - `swing_it.validate.diagnostics.placebo_shuffle(pred, *, seed=0) -> pd.DataFrame`
- Produces:
  - `CHECK_NAMES: tuple[str, ...]` — `("전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL")`
  - `selectivity(pred: pd.DataFrame, *, top_n: int = 15) -> dict` — 키 `mean_pool_per_date, median_pool_per_date, min_pool_per_date, selection_ratio, low_selectivity`
  - `downside_stats(pred: pd.DataFrame, *, top_n: int = 15, tail_quantile: float = 0.2) -> dict` — 키 `p10, pool_p10, p10_gap, tail_capture_rate, expected_tail_rate, tail_avoidance`
  - `layer_verdict(pred: pd.DataFrame, *, top_n: int = 15, n_trials: int = 200, cost_one_way: float = 0.0023, fraction: float = 0.25, seed: int = 0) -> dict` — 키 `gate, folds, untouched, placebo, downside, selectivity, checks, overall_verdict`

**왜 `gate_report` 하나로 판정하지 않는가.** 계획 1 이 정확히 그렇게 했다가 PASS 를 받았고, 폴드별로 뜯으니 3/4 폴드가 랜덤과 구별 불가였다. 이 모듈은 그 실수를 구조적으로 못 하게 만든다. `overall_verdict` 는 네 검사가 **모두** 통과할 때만 `PASS` 다.

| 검사 | 통과 조건 |
|---|---|
| `전체_널_대비` | `gate_report(pred)["verdict"] == "PASS"` |
| `모든_폴드_널_대비` | `fold_null_table(pred)["beats_null"].all()` |
| `손_안_댄_최종창` | `untouched_window(pred)["net_mean"] > 0` **그리고** `percentile > 95` |
| `플라시보_FAIL` | `gate_report(placebo_shuffle(pred))["verdict"] == "FAIL"` |

플라시보만 부호가 반대다. 라벨을 섞었는데도 PASS 가 나오면 파이프라인에 누수가 있다는 뜻이므로, **FAIL 이 나와야 통과**다.

**하방 회피** (`downside_stats`) — verdict 문서 시사점 2. 계획 1 에서 모델이 값을 한 유일한 구간(폴드 3, 나쁜 레짐)에서 한 일은 상위를 잘 고른 게 아니라 **최악의 꼬리를 피한 것**이었다(전체 −2.99% vs 선택분 −0.93%). 반면 평균 기준 gap 은 나머지 세 폴드에서 0.0001 수준으로 사실상 0이었다. 평균만 보면 뉴스·LLM 층의 진짜 기여를 놓칠 수 있다.

- `p10` — 선택분 초과수익의 **하위 10% 분위**. 전체 선택 행을 모아 한 번에 잰다
- `pool_p10` / `p10_gap` — 풀 전체의 하위 10% 분위와 그 차이. 양수면 선택분의 하방이 풀보다 얕다
- `tail_capture_rate` — 날짜별 풀 하위 20%에 속한 이벤트를 선택분이 담은 비율(날짜 평균). **낮을수록 좋다.** 무작위 선택이면 기대값이 정확히 `tail_quantile`(0.2)
- `expected_tail_rate` / `tail_avoidance` — 기대값과 `expected_tail_rate - tail_capture_rate`. 양수면 꼬리를 피했다는 뜻

**선별도** (`selectivity`) — verdict 문서 시사점 3. `selection_ratio = top_n / mean_pool_per_date`. 계획 1 폴드 3 은 하루 평균 24.1건 중 상위 15 = **0.62** 였다. 풀의 3분의 2를 담고 있으면서 "상위 15 선별"이라 부르면 독자가 오해한다. `selection_ratio > 0.5` 면 `low_selectivity` 가 `True` 가 되고, 리포트가 경고 문장을 출력한다(Task 7).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_layer_verdict.py`:

```python
import numpy as np
import pandas as pd

from swing_it.validate.layer_verdict import (
    CHECK_NAMES,
    downside_stats,
    layer_verdict,
    selectivity,
)


def _pred(n_dates: int, n_per_date: int, *, signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n_per_date):
            excess = rng.normal(scale=0.05)
            rows.append(
                {
                    "code": f"{i:06d}",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                    "excess_return": excess,
                    "prediction": excess if signal else rng.normal(),
                    "fold": d * 4 // n_dates,
                }
            )
    return pd.DataFrame(rows)


def test_check_names_match_the_baseline_verdict_format():
    assert CHECK_NAMES == (
        "전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"
    )


def test_selectivity_reports_pool_size_and_ratio():
    out = selectivity(_pred(20, 40, signal=False), top_n=15)
    assert out["mean_pool_per_date"] == 40.0
    assert out["median_pool_per_date"] == 40.0
    assert out["min_pool_per_date"] == 40
    assert np.isclose(out["selection_ratio"], 15 / 40)
    assert out["low_selectivity"] is False


def test_selection_ratio_is_capped_at_one_when_the_pool_is_smaller():
    """풀이 10개인데 상위 15를 고르면 전부 담는 것이다. 1.0 을 넘을 수 없다."""
    out = selectivity(_pred(20, 10, signal=False), top_n=15)
    assert out["selection_ratio"] == 1.0
    assert out["low_selectivity"] is True


def test_low_selectivity_trips_above_half():
    """계획 1 폴드 3 은 하루 24.1건 중 상위 15 = 0.62 였다."""
    assert selectivity(_pred(20, 24, signal=False), top_n=15)["low_selectivity"] is True
    assert selectivity(_pred(20, 40, signal=False), top_n=15)["low_selectivity"] is False


def test_random_prediction_captures_the_tail_at_the_expected_rate():
    out = downside_stats(_pred(200, 40, signal=False), top_n=15, tail_quantile=0.2)
    assert np.isclose(out["expected_tail_rate"], 0.2)
    assert abs(out["tail_capture_rate"] - 0.2) < 0.03
    assert abs(out["tail_avoidance"]) < 0.03
    assert abs(out["p10_gap"]) < 0.02


def test_perfect_prediction_avoids_the_tail_entirely():
    out = downside_stats(_pred(60, 40, signal=True), top_n=15, tail_quantile=0.2)
    assert out["tail_capture_rate"] == 0.0
    assert np.isclose(out["tail_avoidance"], 0.2)
    assert out["p10"] > out["pool_p10"]  # 선택분의 하방이 풀보다 얕다
    assert out["p10_gap"] > 0


def test_inverted_prediction_concentrates_in_the_tail():
    pred = _pred(60, 40, signal=True)
    pred["prediction"] = -pred["prediction"]
    out = downside_stats(pred, top_n=15, tail_quantile=0.2)
    assert out["tail_capture_rate"] > 0.4
    assert out["tail_avoidance"] < 0
    assert out["p10_gap"] < 0


def test_noise_layer_fails_overall():
    out = layer_verdict(_pred(60, 40, signal=False), top_n=15, n_trials=100)
    assert out["overall_verdict"] == "FAIL"
    assert out["checks"]["전체_널_대비"] is False


def test_perfect_layer_passes_every_check():
    out = layer_verdict(
        _pred(80, 40, signal=True), top_n=15, n_trials=100, cost_one_way=0.0
    )
    assert out["overall_verdict"] == "PASS"
    assert all(out["checks"].values())


def test_placebo_check_passes_only_when_shuffled_labels_fail():
    """라벨을 섞었는데 PASS 가 나오면 누수다. 그래서 FAIL 이 정상이다."""
    out = layer_verdict(
        _pred(80, 40, signal=True), top_n=15, n_trials=100, cost_one_way=0.0
    )
    assert out["placebo"]["verdict"] == "FAIL"
    assert out["checks"]["플라시보_FAIL"] is True


def test_a_layer_that_only_wins_overall_still_fails():
    """전체 널만 이기고 폴드가 갈리면 FAIL. 계획 1 이 걸린 바로 그 함정이다."""
    rng = np.random.default_rng(0)
    frames = []
    for fold in range(4):
        block = _pred(20, 40, signal=(fold == 3), seed=fold)
        block["date"] = block["date"] + pd.Timedelta(days=20 * fold)
        block["fold"] = fold
        if fold != 3:
            block["prediction"] = rng.normal(size=len(block))
        frames.append(block)
    pred = pd.concat(frames, ignore_index=True)
    out = layer_verdict(pred, top_n=15, n_trials=100, cost_one_way=0.0)
    assert out["checks"]["모든_폴드_널_대비"] is False
    assert out["overall_verdict"] == "FAIL"


def test_report_carries_every_section():
    out = layer_verdict(_pred(60, 40, signal=False), top_n=15, n_trials=50)
    for key in ["gate", "folds", "untouched", "placebo", "downside",
                "selectivity", "checks", "overall_verdict"]:
        assert key in out
    assert len(out["folds"]) == 4
    assert set(out["checks"]) == set(CHECK_NAMES)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_layer_verdict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.validate.layer_verdict'`

- [ ] **Step 3: 구현**

`src/swing_it/validate/layer_verdict.py`:

```python
"""층 하나에 대한 종합 판정.

``gate_report`` 하나로 판정하지 않는다. 계획 1 이 정확히 그렇게 했다가 PASS 를
받았고, 폴드별로 뜯으니 3/4 폴드에서 전략이 랜덤 널과 소수점 넷째 자리까지
같았다. 전체 표본 평균은 한 구간의 방어 효과를 전 기간의 예측력으로 오인하게
만든다 (``docs/research/2026-08-17-baseline-verdict.md``).

그래서 네 검사를 모두 통과해야 ``PASS`` 다. 플라시보만 부호가 반대다 —
라벨을 섞었는데 PASS 가 나오면 누수이므로 FAIL 이 정상이다.

여기에 계획 1 이 사후에 발견한 두 축을 더한다:

- **하방 회피** — 폴드 3 이 보여준 것은 "상위를 잘 고르기"보다 "최악의 꼬리를
  피하기"가 이 데이터에서 더 실현 가능하다는 것이었다. 평균만 보면 놓친다.
- **선별도** — 폴드 3 은 하루 24건 중 상위 15를 골라 풀의 62%를 담고 있었다.
  그걸 "선별"이라 부르면 안 된다. 숫자로 보이게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics import fold_null_table, placebo_shuffle, untouched_window
from .gate import gate_report

CHECK_NAMES: tuple[str, ...] = (
    "전체_널_대비",
    "모든_폴드_널_대비",
    "손_안_댄_최종창",
    "플라시보_FAIL",
)


LOW_SELECTIVITY_RATIO = 0.5


def selectivity(pred: pd.DataFrame, *, top_n: int = 15) -> dict:
    """후보 풀 크기와 ``top_n`` 이 그 풀에서 차지하는 비율.

    ``selection_ratio`` 가 0.62 면 "상위 15 선별"이 아니라 "풀의 62%를 담은 것"이다
    (계획 1 폴드 3 의 실측값). 이 숫자를 리포트에 찍지 않으면 선별했다는 말이
    사실을 오도한다.
    """
    sizes = pred.groupby("date").size()
    mean_pool = float(sizes.mean())
    ratio = float(min(1.0, top_n / mean_pool)) if mean_pool > 0 else 1.0
    return {
        "mean_pool_per_date": mean_pool,
        "median_pool_per_date": float(sizes.median()),
        "min_pool_per_date": int(sizes.min()),
        "selection_ratio": ratio,
        "low_selectivity": bool(ratio > LOW_SELECTIVITY_RATIO),
    }


def downside_stats(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    tail_quantile: float = 0.2,
) -> dict:
    """선택분이 하방을 얼마나 피했는가 (verdict 문서 시사점 2).

    두 축을 잰다.

    - ``p10`` — 선택분 초과수익의 하위 10% 분위. 전체 선택 행을 모아 한 번에
      잰다. 풀의 같은 분위(``pool_p10``)와의 차이가 ``p10_gap`` 이고, 양수면
      선택분의 하방이 풀보다 얕다는 뜻이다.
    - ``tail_capture_rate`` — 날짜마다 풀의 ``tail_quantile`` 임계값을 구하고
      선택된 종목 중 몇 %가 그 아래로 떨어졌는지를 센다. **낮을수록 좋다.**
      무작위 선택이면 기대값이 정확히 ``tail_quantile`` 이다.

    계획 1 에서 모델이 값을 한 유일한 구간이 정확히 이 축이었다. 평균 gap 은
    네 폴드 중 셋에서 0.0001 수준이었지만, 나쁜 레짐에서 최악의 꼬리를 피한
    효과는 2%p 였다.
    """
    captures = []
    picked_all = []
    for _, group in pred.groupby("date"):
        pool = group["excess_return"].to_numpy()
        if len(pool) == 0:
            continue
        threshold = float(np.quantile(pool, tail_quantile))
        picked = group.nlargest(top_n, "prediction")["excess_return"].to_numpy()
        captures.append(float((picked <= threshold).mean()))
        picked_all.append(picked)

    picked_pool = np.concatenate(picked_all)
    p10 = float(np.quantile(picked_pool, 0.10))
    pool_p10 = float(np.quantile(pred["excess_return"].to_numpy(), 0.10))
    capture = float(np.mean(captures))
    return {
        "p10": p10,
        "pool_p10": pool_p10,
        "p10_gap": p10 - pool_p10,
        "tail_capture_rate": capture,
        "expected_tail_rate": float(tail_quantile),
        "tail_avoidance": float(tail_quantile - capture),
    }


def layer_verdict(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    cost_one_way: float = 0.0023,
    fraction: float = 0.25,
    seed: int = 0,
) -> dict:
    """네 검사를 모두 통과해야 PASS 인 종합 판정."""
    gate = gate_report(
        pred, top_n=top_n, n_trials=n_trials, cost_one_way=cost_one_way, seed=seed
    )
    folds = fold_null_table(pred, top_n=top_n, n_trials=n_trials, seed=seed)
    untouched = untouched_window(
        pred,
        fraction=fraction,
        top_n=top_n,
        n_trials=n_trials,
        cost_one_way=cost_one_way,
        seed=seed,
    )
    placebo = gate_report(
        placebo_shuffle(pred, seed=seed),
        top_n=top_n,
        n_trials=n_trials,
        cost_one_way=cost_one_way,
        seed=seed,
    )

    checks = {
        "전체_널_대비": gate["verdict"] == "PASS",
        "모든_폴드_널_대비": bool(folds["beats_null"].all()),
        "손_안_댄_최종창": untouched["net_mean"] > 0 and untouched["percentile"] > 95.0,
        # 라벨을 섞었는데 PASS 면 누수다. FAIL 이 나와야 통과.
        "플라시보_FAIL": placebo["verdict"] == "FAIL",
    }
    return {
        "gate": gate,
        "folds": folds.to_dict("records"),
        "untouched": untouched,
        "placebo": placebo,
        "downside": downside_stats(pred, top_n=top_n),
        "selectivity": selectivity(pred, top_n=top_n),
        "checks": checks,
        "overall_verdict": "PASS" if all(checks.values()) else "FAIL",
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_layer_verdict.py -v`
Expected: 12 passed

- [ ] **Step 5: 계획 1 진단 회귀 확인**

Run: `uv run pytest tests/test_validate_diagnostics.py tests/test_validate_gate.py -v`
Expected: 전부 통과 (계획 1 의 진단 모듈을 읽기만 했음을 확인)

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/validate/layer_verdict.py tests/test_layer_verdict.py
git commit -m "feat(validate): 층 종합 판정 — 진단 4종 + 하방 회피 + 선별도"
```

---
### Task 5: 동일조건 층 비교 하네스와 PEAD 벤치마크

**Files:**
- Create: `src/swing_it/validate/pead.py`
- Create: `src/swing_it/validate/compare.py`
- Test: `tests/test_validate_pead.py`
- Test: `tests/test_validate_compare.py`

**Interfaces:**
- Consumes:
  - `swing_it.layers.LAYER_LABELS`
  - `swing_it.model.layered.fold_dates`, `walk_forward_predict_features`
  - `swing_it.model.baseline.portfolio_returns(pred, *, top_n=15) -> pd.DataFrame`
  - `swing_it.validate.layer_verdict.layer_verdict(pred, *, top_n=15, n_trials=200, cost_one_way=0.0023, fraction=0.25, seed=0) -> dict`
- Produces:
  - `pead.PEAD_SHARPE: float` = `0.80`, `pead.PEAD_T: float` = `2.20`, `pead.COMPARABLE_RATIO: float` = `0.75`
  - `pead.overlap_stats(curve: pd.DataFrame, *, hold_days: int = 40) -> dict` — 키 `mean, sd, n_obs, n_effective, sharpe_annual, t_stat`
  - `pead.pead_benchmark(curve: pd.DataFrame, *, hold_days: int = 40) -> dict` — `overlap_stats` 키 + `pead_sharpe, pead_t, verdict` ∈ `{"BEATS_PEAD","COMPARABLE_TO_PEAD","LOSES_TO_PEAD"}`
  - `pead.benchmark_all_layers(curves: dict[str, pd.DataFrame], *, hold_days: int = 40) -> list[dict]` — `[{layer, ...pead_benchmark}]`
  - `compare.paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict` — 키 `mean_diff, ci_low, ci_high, p_greater, n_dates`
  - `compare.llm_verdict(layer_rows: list[dict], deltas: list[dict]) -> dict` — 키 `layer2_mean, layer3_mean, delta, ci_low, ci_high, p_greater, layer3_overall, verdict` ∈ `{"LLM_JUSTIFIED","LLM_NOT_JUSTIFIED"}`
  - `compare.compare_layers(ds, spec, *, top_n=15, n_folds=4, min_train=200, n_trials=200, n_boot=2000, cost_one_way=0.0023, hold_days=40, seed=0) -> dict` — 키 `rows, deltas, llm, pead, sample`

**겹침 보정이 PEAD 쪽의 핵심이다.** 이벤트가 매일 발생하고 보유가 40거래일이므로 날짜별 관측이 독립이 아니다. 보정 없이 t 를 계산하면 크게 부풀려진다.

- 관측 단위: 날짜별 `mean_excess` (40거래일 보유 수익률)
- 연율 Sharpe: `mean / sd * sqrt(252 / hold_days)`
- 유효 표본: `n_effective = max(1, n_obs // hold_days)` — 겹치지 않는 블록 수
- t: `mean / (sd / sqrt(n_effective))`

보수적인 쪽으로 틀리는 선택이다. 스펙 §2 가 남긴 교훈이 정확히 그것이다.

**판정 (사전등록):** `BEATS_PEAD` = `sharpe_annual > 0.80` **그리고** `t_stat > 2.20` / `COMPARABLE_TO_PEAD` = `sharpe_annual >= 0.60` / 그 외 `LOSES_TO_PEAD`.

**LLM 정당화 판정 규칙 (사전등록).** `LLM_JUSTIFIED` 는 세 조건이 **모두** 참일 때만.

1. ③ − ② 평균 차이가 양수
2. 그 차이의 부트스트랩 95% 신뢰구간 하한이 0 초과
3. ③층의 **`overall_verdict` 가 `PASS`** — `gate` 만이 아니라 Task 4 의 네 검사 전부

3번이 `gate` 가 아니라 `overall_verdict` 인 것이 중요하다. 계획 1 은 `gate` 만 보면 PASS 였다. ③이 그 함정에 걸린 채로 ②를 이겨봤자 LLM 비용을 정당화하지 못한다.

**널 동일성.** `random_null` 은 `pred` 의 `excess_return` 과 날짜 그룹만 쓴다. 세 층의 `pred` 는 같은 행 집합이므로(Task 2) 널 통계가 **완전히 같아야 한다.** 테스트로 강제한다. 같지 않다면 표본이 어긋났다는 뜻이고, 그 순간 비교는 무효다.

- [ ] **Step 1: 실패하는 PEAD 테스트 작성**

`tests/test_validate_pead.py`:

```python
import numpy as np
import pandas as pd

from swing_it.validate.pead import (
    COMPARABLE_RATIO,
    PEAD_SHARPE,
    PEAD_T,
    benchmark_all_layers,
    overlap_stats,
    pead_benchmark,
)


def _curve(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(values), freq="B"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_registered_constants_match_the_spec():
    """스펙 §7 — kr-quant PEAD 검증값 Sharpe ~0.8, t ~2.2."""
    assert PEAD_SHARPE == 0.80
    assert PEAD_T == 2.20
    assert COMPARABLE_RATIO == 0.75


def test_sharpe_is_annualised_from_the_holding_period():
    values = [0.02] * 400
    values[0] = 0.01  # sd 가 0 이 되지 않게
    stats = overlap_stats(_curve(values), hold_days=40)
    series = pd.Series(values)
    expected = series.mean() / series.std(ddof=1) * np.sqrt(252 / 40)
    assert np.isclose(stats["sharpe_annual"], expected)


def test_effective_sample_divides_by_the_holding_period():
    stats = overlap_stats(_curve(list(np.random.default_rng(0).normal(size=400))), hold_days=40)
    assert stats["n_obs"] == 400
    assert stats["n_effective"] == 10


def test_effective_sample_never_drops_below_one():
    assert overlap_stats(_curve([0.01, 0.02, 0.03]), hold_days=40)["n_effective"] == 1


def test_overlap_correction_shrinks_the_t_stat():
    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=0.01, scale=0.05, size=400))
    stats = overlap_stats(_curve(values), hold_days=40)
    naive_t = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(400))
    assert stats["t_stat"] < naive_t / 5


def test_beats_pead_needs_both_sharpe_and_t():
    rng = np.random.default_rng(1)
    out = pead_benchmark(_curve(list(rng.normal(loc=0.06, scale=0.05, size=800))))
    assert out["sharpe_annual"] > PEAD_SHARPE
    assert out["t_stat"] > PEAD_T
    assert out["verdict"] == "BEATS_PEAD"


def test_loses_to_pead_when_there_is_no_edge():
    rng = np.random.default_rng(2)
    out = pead_benchmark(_curve(list(rng.normal(loc=0.0, scale=0.05, size=800))))
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_negative_mean_never_reads_as_comparable():
    rng = np.random.default_rng(3)
    out = pead_benchmark(_curve(list(rng.normal(loc=-0.03, scale=0.05, size=800))))
    assert out["sharpe_annual"] < 0
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_report_carries_the_pead_reference_values():
    out = pead_benchmark(_curve([0.01, 0.02, 0.03, 0.04] * 50))
    assert out["pead_sharpe"] == PEAD_SHARPE
    assert out["pead_t"] == PEAD_T


def test_benchmark_all_layers_tags_each_row():
    rng = np.random.default_rng(4)
    curves = {
        "supply": _curve(list(rng.normal(size=200))),
        "supply_news": _curve(list(rng.normal(size=200))),
    }
    rows = benchmark_all_layers(curves)
    assert [r["layer"] for r in rows] == ["supply", "supply_news"]
    assert all("verdict" in r for r in rows)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_validate_pead.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.validate.pead'`

- [ ] **Step 3: PEAD 모듈 구현**

`src/swing_it/validate/pead.py`:

```python
"""PEAD 대비 벤치마크 (스펙 §7).

PEAD 는 이 레포의 중심이 아니라 **넘어야 할 선**이다. 이미 검증된 알파가
있는데 뉴스 모델이 그보다 못하다면, 뉴스 모델을 운영에 올릴 이유가 없다.

기준값은 kr-quant PEAD 검증 결과다: 전 구간 Sharpe ~0.8, t ~2.2
(저회전, point-in-time 유동성 하한 적용본).

겹침 보정: 이벤트는 매일 나고 보유는 40거래일이라 날짜별 관측이 독립이 아니다.
보정 없는 t 는 크게 부풀려진다. 여기서는 겹치지 않는 블록 수
(``n_obs // hold_days``) 를 유효 표본으로 쓴다. 보수적인 쪽으로 틀리는 선택이다.

이 모듈은 ``math`` 만 쓴다. ``numpy`` 를 임포트하지 마라 — ruff 가 F401 로 잡는다.
"""

from __future__ import annotations

import math

import pandas as pd

PEAD_SHARPE: float = 0.80
PEAD_T: float = 2.20
COMPARABLE_RATIO: float = 0.75

TRADING_DAYS = 252


def overlap_stats(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """날짜별 포트폴리오 초과수익 시계열의 겹침 보정 통계.

    Args:
        curve: ``portfolio_returns`` 결과. ``mean_excess`` 는 40거래일 보유 수익률.
        hold_days: 보유 거래일 수. 겹침 정도를 결정한다.
    """
    values = curve["mean_excess"].dropna().to_numpy(dtype=float)
    n_obs = int(len(values))
    if n_obs < 2:
        raise ValueError(f"관측이 {n_obs} 개뿐이라 분산을 못 잰다")

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    n_effective = max(1, n_obs // hold_days)
    sharpe = mean / sd * math.sqrt(TRADING_DAYS / hold_days) if sd > 0 else 0.0
    t_stat = mean / (sd / math.sqrt(n_effective)) if sd > 0 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "n_obs": n_obs,
        "n_effective": int(n_effective),
        "sharpe_annual": float(sharpe),
        "t_stat": float(t_stat),
    }


def pead_benchmark(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """검증된 PEAD 대비 판정. 이기든 지든 명시적으로 남긴다 (스펙 §10-5)."""
    stats = overlap_stats(curve, hold_days=hold_days)
    sharpe = stats["sharpe_annual"]
    if sharpe > PEAD_SHARPE and stats["t_stat"] > PEAD_T:
        verdict = "BEATS_PEAD"
    elif sharpe >= PEAD_SHARPE * COMPARABLE_RATIO:
        verdict = "COMPARABLE_TO_PEAD"
    else:
        verdict = "LOSES_TO_PEAD"
    return {**stats, "pead_sharpe": PEAD_SHARPE, "pead_t": PEAD_T, "verdict": verdict}


def benchmark_all_layers(
    curves: dict[str, pd.DataFrame], *, hold_days: int = 40
) -> list[dict]:
    """층별 PEAD 판정 목록."""
    return [
        {"layer": layer, **pead_benchmark(curve, hold_days=hold_days)}
        for layer, curve in curves.items()
    ]
```

- [ ] **Step 4: PEAD 테스트 통과 확인**

Run: `uv run pytest tests/test_validate_pead.py -v`
Expected: 10 passed

- [ ] **Step 5: 실패하는 비교 하네스 테스트 작성**

`tests/test_validate_compare.py`:

```python
import numpy as np
import pandas as pd
import pytest

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.validate.compare import compare_layers, llm_verdict, paired_bootstrap

NEWS = ["article_count"]
LLM = ["llm_specificity"]


def _ds(n: int, *, llm_is_signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [*FEATURE_COLUMNS, *NEWS, *LLM]
    df = pd.DataFrame({col: rng.normal(size=n) for col in cols})
    df["code"] = [f"{i % 30:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 30, unit="D")
    noise = rng.normal(scale=0.02, size=n)
    df["excess_return"] = noise + (df["llm_specificity"] * 0.08 if llm_is_signal else 0.0)
    return df[["code", "date", *cols, "excess_return"]]


def _spec() -> dict[str, list[str]]:
    return {
        "supply": list(FEATURE_COLUMNS),
        "supply_news": [*FEATURE_COLUMNS, *NEWS],
        "supply_news_llm": [*FEATURE_COLUMNS, *NEWS, *LLM],
    }


def _portfolio(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_paired_bootstrap_detects_a_real_difference():
    base = np.random.default_rng(0).normal(scale=0.01, size=200)
    out = paired_bootstrap(
        _portfolio(list(base + 0.02)), _portfolio(list(base)), n_boot=1000
    )
    assert np.isclose(out["mean_diff"], 0.02, atol=0.005)
    assert out["ci_low"] > 0
    assert out["p_greater"] > 0.95
    assert out["n_dates"] == 200


def test_paired_bootstrap_finds_nothing_when_series_are_identical():
    values = list(np.random.default_rng(1).normal(scale=0.01, size=200))
    out = paired_bootstrap(_portfolio(values), _portfolio(values), n_boot=1000)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_paired_bootstrap_pairs_on_date_not_position():
    a = _portfolio([0.1, 0.2, 0.3])
    b = _portfolio([0.1, 0.2, 0.3]).iloc[::-1].reset_index(drop=True)
    assert paired_bootstrap(a, b, n_boot=200)["mean_diff"] == 0.0


def test_paired_bootstrap_rejects_disjoint_dates():
    a = _portfolio([0.1, 0.2])
    b = _portfolio([0.1, 0.2])
    b["date"] = b["date"] + pd.Timedelta(days=365)
    with pytest.raises(ValueError, match="공통 날짜"):
        paired_bootstrap(a, b, n_boot=100)


def test_compare_layers_returns_one_row_per_layer():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [r["layer"] for r in result["rows"]] == [
        "supply", "supply_news", "supply_news_llm"
    ]
    assert [r["n_features"] for r in result["rows"]] == [7, 8, 9]


def test_all_layers_share_an_identical_null():
    """널이 다르면 세 층이 다른 표본 위에 있다는 뜻이고, 비교는 그 순간 무효다."""
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert len({r["null_mean"] for r in result["rows"]}) == 1
    assert len({r["null_p95"] for r in result["rows"]}) == 1


def test_every_layer_carries_its_composite_verdict():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    for row in result["rows"]:
        assert row["overall_verdict"] in {"PASS", "FAIL"}
        assert set(row["detail"]["checks"]) == {
            "전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"
        }


def test_every_layer_carries_selectivity_and_downside():
    """평균만 보면 뉴스·LLM 층의 진짜 기여를 놓친다 (baseline-verdict 시사점 2·3)."""
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    for row in result["rows"]:
        assert 0.0 < row["selection_ratio"] <= 1.0
        assert row["mean_pool_per_date"] == 30.0
        assert row["low_selectivity"] is False  # 30건 중 15 = 0.5, 초과가 아니다
        assert 0.0 <= row["tail_capture_rate"] <= 1.0
        assert "p10" in row and "p10_gap" in row


def test_a_signal_layer_avoids_the_tail_better_than_a_noise_layer():
    signal = compare_layers(
        _ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200
    )["rows"][-1]
    noise = compare_layers(
        _ds(1500, llm_is_signal=False), _spec(), n_trials=50, n_boot=200
    )["rows"][-1]
    assert signal["tail_capture_rate"] < noise["tail_capture_rate"]


def test_llm_justified_when_llm_feature_carries_the_signal():
    result = compare_layers(
        _ds(3000, llm_is_signal=True), _spec(), n_trials=200, n_boot=1000,
        cost_one_way=0.0,
    )
    assert result["llm"]["verdict"] == "LLM_JUSTIFIED"
    assert result["llm"]["delta"] > 0


def test_llm_not_justified_when_llm_feature_is_noise():
    result = compare_layers(
        _ds(3000, llm_is_signal=False), _spec(), n_trials=200, n_boot=1000
    )
    assert result["llm"]["verdict"] == "LLM_NOT_JUSTIFIED"


def test_llm_not_justified_when_layer3_only_passes_the_naive_gate():
    """계획 1 이 걸린 함정 — gate 만 PASS 고 종합은 FAIL 이면 정당화되지 않는다."""
    rows = [
        {"layer": "supply_news", "strategy_mean": 0.001, "verdict": "PASS",
         "overall_verdict": "FAIL"},
        {"layer": "supply_news_llm", "strategy_mean": 0.004, "verdict": "PASS",
         "overall_verdict": "FAIL"},
    ]
    deltas = [
        {"layer": "supply_news_llm", "baseline": "supply_news", "mean_diff": 0.003,
         "ci_low": 0.001, "ci_high": 0.005, "p_greater": 0.99, "n_dates": 100}
    ]
    assert llm_verdict(rows, deltas)["verdict"] == "LLM_NOT_JUSTIFIED"


def test_deltas_compare_each_layer_to_the_previous_one():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [(d["layer"], d["baseline"]) for d in result["deltas"]] == [
        ("supply_news", "supply"),
        ("supply_news_llm", "supply_news"),
    ]


def test_pead_rows_accompany_every_layer():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [r["layer"] for r in result["pead"]] == [
        "supply", "supply_news", "supply_news_llm"
    ]


def test_sample_metadata_is_reported():
    ds = _ds(1500, llm_is_signal=True)
    result = compare_layers(ds, _spec(), n_trials=50, n_boot=200)
    assert result["sample"]["n_rows"] == len(ds)
    assert result["sample"]["n_codes"] == ds["code"].nunique()
    assert result["sample"]["top_n"] == 15


def test_compare_layers_rejects_a_spec_whose_layers_are_not_nested():
    bad = {"supply": list(FEATURE_COLUMNS), "supply_news": NEWS}
    with pytest.raises(ValueError, match="누적"):
        compare_layers(_ds(1500, llm_is_signal=True), bad, n_trials=50, n_boot=200)
```

- [ ] **Step 6: 테스트 실패 확인**

Run: `uv run pytest tests/test_validate_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.validate.compare'`

- [ ] **Step 7: 비교 하네스 구현**

`src/swing_it/validate/compare.py`:

```python
"""층 비교 하네스 — 세 층을 동일 조건에서 재는 유일한 경로 (스펙 §3).

동일 조건이란 셋을 말한다:

1. **동일 표본** — ``layers.build_layered_dataset`` 이 보장한다. 여기서는
   층 피처가 서로 누적 관계인지만 다시 확인한다.
2. **동일 폴드** — ``fold_dates`` 를 한 번 계산해 세 층에 그대로 넘긴다.
3. **동일 널** — ``random_null`` 은 ``excess_return`` 과 날짜만 쓰므로,
   1·2 가 지켜지면 세 층의 널이 같아진다. 같지 않으면 어딘가 어긋난 것이다.

층 간 차이가 우연인지는 날짜 페어드 부트스트랩으로 잰다. 같은 날짜의
두 층 포트폴리오를 짝지어 차이를 보므로 시장 공통 변동이 상쇄된다.

층별 판정은 ``gate_report`` 가 아니라 ``layer_verdict`` 다. 계획 1 이
``gate_report`` 만 보고 PASS 를 받았다가 폴드별로 뜯려 FAIL 이 된 전례가 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..layers import LAYER_LABELS
from ..model.baseline import portfolio_returns
from ..model.layered import fold_dates, walk_forward_predict_features
from .layer_verdict import layer_verdict
from .pead import benchmark_all_layers


def paired_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """날짜로 짝지은 ``a - b`` 차이의 부트스트랩 분포.

    Args:
        a, b: ``portfolio_returns`` 결과 (``[date, n_positions, mean_excess]``).

    Returns:
        ``mean_diff`` (관측 평균 차이), ``ci_low``/``ci_high`` (95% 신뢰구간),
        ``p_greater`` (부트스트랩 표본 중 차이가 0 초과인 비율), ``n_dates``.
    """
    merged = a.merge(b, on="date", suffixes=("_a", "_b"))
    diff = (merged["mean_excess_a"] - merged["mean_excess_b"]).to_numpy()
    n = len(diff)
    if n == 0:
        raise ValueError("두 층의 공통 날짜가 없다 — 폴드가 어긋났다")

    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_greater": float((means > 0).mean()),
        "n_dates": int(n),
    }


def _check_nested(spec: dict[str, list[str]]) -> None:
    names = list(spec)
    for prev, cur in zip(names, names[1:]):
        if spec[cur][: len(spec[prev])] != spec[prev]:
            raise ValueError(
                f"층 '{cur}' 이 '{prev}' 의 누적 확장이 아니다. "
                "층은 피처를 더하기만 해야 기여도 분리가 성립한다."
            )


def llm_verdict(layer_rows: list[dict], deltas: list[dict]) -> dict:
    """③층이 LLM 을 쓸 값을 하는지 판정한다 (스펙 §3).

    세 조건이 모두 참일 때만 ``LLM_JUSTIFIED``:
    ③−② 차이가 양수 / 그 차이의 95% CI 하한이 0 초과 / ③층의 **종합 판정**이 PASS.

    세 번째가 ``gate`` 가 아니라 ``overall_verdict`` 인 것이 요점이다. 계획 1 은
    ``gate`` 만 보면 PASS 였지만 폴드별로는 랜덤과 구별되지 않았다. ③이 그 함정에
    걸린 채로 ②를 이겨봤자 LLM 비용을 정당화하지 못한다.
    """
    by_layer = {row["layer"]: row for row in layer_rows}
    layer2 = by_layer["supply_news"]
    layer3 = by_layer["supply_news_llm"]
    delta = next(d for d in deltas if d["layer"] == "supply_news_llm")

    passed = (
        delta["mean_diff"] > 0
        and delta["ci_low"] > 0
        and layer3["overall_verdict"] == "PASS"
    )
    return {
        "layer2_mean": float(layer2["strategy_mean"]),
        "layer3_mean": float(layer3["strategy_mean"]),
        "delta": float(delta["mean_diff"]),
        "ci_low": float(delta["ci_low"]),
        "ci_high": float(delta["ci_high"]),
        "p_greater": float(delta["p_greater"]),
        "layer3_overall": layer3["overall_verdict"],
        "verdict": "LLM_JUSTIFIED" if passed else "LLM_NOT_JUSTIFIED",
    }


def compare_layers(
    ds: pd.DataFrame,
    spec: dict[str, list[str]],
    *,
    top_n: int = 15,
    n_folds: int = 4,
    min_train: int = 200,
    n_trials: int = 200,
    n_boot: int = 2000,
    cost_one_way: float = 0.0023,
    hold_days: int = 40,
    seed: int = 0,
) -> dict:
    """세 층을 동일 표본·동일 폴드·동일 널로 비교한다."""
    _check_nested(spec)
    folds = fold_dates(ds, n_folds=n_folds, min_train=min_train)

    rows: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    for layer, columns in spec.items():
        pred = walk_forward_predict_features(ds, columns, folds=folds, seed=seed)
        detail = layer_verdict(
            pred,
            top_n=top_n,
            n_trials=n_trials,
            cost_one_way=cost_one_way,
            seed=seed,
        )
        curves[layer] = portfolio_returns(pred, top_n=top_n)
        rows.append(
            {
                "layer": layer,
                "label": LAYER_LABELS.get(layer, layer),
                "n_features": len(columns),
                **detail["gate"],
                "overall_verdict": detail["overall_verdict"],
                # 선별도·하방은 평균과 나란히 읽혀야 하므로 최상위로 끌어올린다.
                "mean_pool_per_date": detail["selectivity"]["mean_pool_per_date"],
                "selection_ratio": detail["selectivity"]["selection_ratio"],
                "low_selectivity": detail["selectivity"]["low_selectivity"],
                "p10": detail["downside"]["p10"],
                "p10_gap": detail["downside"]["p10_gap"],
                "tail_capture_rate": detail["downside"]["tail_capture_rate"],
                "tail_avoidance": detail["downside"]["tail_avoidance"],
                "detail": detail,
            }
        )

    names = list(spec)
    deltas = [
        {
            "layer": cur,
            "baseline": prev,
            **paired_bootstrap(curves[cur], curves[prev], n_boot=n_boot, seed=seed),
        }
        for prev, cur in zip(names, names[1:])
    ]
    has_llm_layer = {"supply_news", "supply_news_llm"} <= set(spec)

    return {
        "rows": rows,
        "deltas": deltas,
        "llm": llm_verdict(rows, deltas) if has_llm_layer else None,
        "pead": benchmark_all_layers(curves, hold_days=hold_days),
        "sample": {
            "n_rows": int(len(ds)),
            "n_codes": int(ds["code"].nunique()),
            "date_range": [str(ds["date"].min().date()), str(ds["date"].max().date())],
            "n_folds": n_folds,
            "top_n": top_n,
            "n_trials": n_trials,
            "n_boot": n_boot,
            "cost_one_way": cost_one_way,
            "hold_days": hold_days,
            "seed": seed,
        },
    }
```

- [ ] **Step 8: 테스트 통과 확인**

Run: `uv run pytest tests/test_validate_compare.py -v`
Expected: 16 passed

- [ ] **Step 9: 커밋**

```bash
git add src/swing_it/validate/pead.py src/swing_it/validate/compare.py \
  tests/test_validate_pead.py tests/test_validate_compare.py
git commit -m "feat(validate): PEAD 겹침보정 벤치마크 + 동일조건 층 비교 하네스"
```

---
### Task 6: 근거 카드와 전진 기록

**Files:**
- Create: `src/swing_it/report/__init__.py`
- Create: `src/swing_it/report/cards.py`
- Create: `src/swing_it/report/forward_log.py`
- Test: `tests/test_report_cards.py`
- Test: `tests/test_forward_log.py`

**Interfaces:**
- Consumes:
  - `swing_it.model.layered.fit_final_model(ds, feature_columns, *, seed=0) -> tuple[GradientBoostingRegressor, pd.Series]`
  - `swing_it.adapters.news_db.connect_news(path=None)`, `load_article_cards(conn, start, end, *, ops_only=False) -> pd.DataFrame` — 컬럼 `[date, code, hotness_rank, hotness, article_id, press, title, body200, url]` (Task 1)
  - `swing_it.adapters.db.load_stock_master(conn) -> pd.DataFrame` — `[code, name, market, sector, last_trade_date, is_delisted]`
- Produces (`report.cards`):
  - `DISCLAIMER: str`
  - `feature_contributions(model, row: pd.Series, baseline: pd.Series, feature_columns: Sequence[str]) -> pd.Series` — 내림차순
  - `find_analogs(train: pd.DataFrame, row: pd.Series, feature_columns: Sequence[str], *, k: int = 3) -> pd.DataFrame` — 컬럼 `[code, date, excess_return, distance]`
  - `render_card(row, *, rank, contributions, analogs, articles, gate_verdict, layer_label, entry_date, exit_date, names=None, material_types=None, top_features=5) -> str`
  - `render_candidates(candidates, *, model, baseline, train, feature_columns, articles, gate_verdict, layer_label, entry_date, exit_date, names=None, material_types=None) -> str`
- Produces (`report.forward_log`):
  - `LOG_KEY: tuple[str, ...]` — `("logged_at", "layer", "code", "event_date")`
  - `build_records(candidates, *, feature_columns, layer, gate_verdict, logged_at, entry_date, exit_date, seed=0, version=None) -> list[dict]`
  - `append_forward_log(path: str | Path, records: Sequence[dict]) -> int` — append-only. 새로 쓴 건수 반환
  - `read_forward_log(path: str | Path) -> pd.DataFrame`
  - `score_forward_log(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame` — `labels` 는 `[code, date, excess_return]`, 결과에 `realized_excess` 추가

**기여도 정의:** 피처 `j` 를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량, `f(x) - f(x | x_j := median_j)`. 상호작용 때문에 합이 총 예측과 정확히 일치하지 않는다. **이 사실을 카드에 각주로 찍는다.** 정확한 가법 분해가 필요하면 SHAP 이 필요하지만, 하루 10건을 사람이 훑고 "무엇이 이걸 밀어올렸나"를 보는 데는 순위면 충분하고, 의존성을 하나 덜 늘린다.

**기사 표시:** `load_article_cards` 는 `(code, date)` 당 **대표 기사 1건**을 준다(`daily_news.primary_news_id` 조인) — 제목·매체·URL·본문 200자. 발행 시각은 없다(빅카인즈가 일자 단위만 준다).

"기사 N건 · 매체 M곳 · 평소 대비 X배"라는 집계 수치는 `load_article_cards` 가 아니라 **후보 행의 ②층 피처**(`article_count`·`outlet_count`·`article_ratio`)에서 읽는다. 같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다. 그 피처가 없는 층(①층)에서는 그 줄을 생략한다.

**`material_type` 은 표시 전용이다.** 모델 행렬에 들어가지 않으므로 후보 행에 없다. `render_card` 가 `material_types: dict[str, str]` 로 따로 받는다 — 사람이 "이게 수주 건인지 루머인지"를 보는 데 쓴다. 없으면 그 줄을 생략한다.

대표 기사가 없는 행(`news_id` 결측)도 핫함 수치는 살아 있으므로 카드를 만든다.

**FAIL 표시 강제:** 카드 맨 아래에 그 층의 판정을 반드시 찍는다. `PASS` 가 아닌 모델의 카드가 추천처럼 읽히면 안 된다. 테스트로 강제한다.

**전진 기록이 append-only 인 이유:** 전진검증의 가치는 "그때 그렇게 예측했다"가 사후에 바뀌지 않는 데 있다. 같은 키를 다시 쓰려 하면 조용히 건너뛴다. 모델을 고쳐 다시 돌려도 과거 기록은 남는다.

- [ ] **Step 1: 실패하는 카드 테스트 작성**

`tests/test_report_cards.py`:

```python
import numpy as np
import pandas as pd

from swing_it.model.layered import fit_final_model
from swing_it.report.cards import feature_contributions, find_analogs, render_card

COLS = ["tv_ratio", "article_count", "outlet_count", "article_ratio", "llm_specificity"]


def _train(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in COLS})
    df["code"] = [f"{i % 20:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    df["excess_return"] = df["llm_specificity"] * 0.08 + rng.normal(scale=0.01, size=n)
    return df


def _candidate() -> pd.Series:
    return pd.Series(
        {
            "code": "034020",
            "date": pd.Timestamp("2026-08-14"),
            "tv_ratio": 4.6,
            "article_count": 7.0,
            "outlet_count": 5.0,
            "article_ratio": 4.2,
            "llm_specificity": 2.5,
            "prediction": 0.0481,
        }
    )


def _articles() -> pd.DataFrame:
    """load_article_cards 반환 형태 — 집계 수치는 없다. (code, date) 당 대표 기사 1건."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14", "2026-08-14"]),
            "code": ["034020", "005930"],
            "hotness_rank": [1.0, 2.0],
            "hotness": [0.81, 0.44],
            "article_id": ["n1", "n2"],
            "press": ["한국경제", "더벨"],
            "title": ["체코 원전 2기 최종 계약", "무관한 기사"],
            "body200": ["두산에너빌리티가 체코 두코바니 원전 2기 주기기 공급 계약을 체결했다.", "본문"],
            "url": ["https://x/1", "https://x/2"],
        }
    )


def _card(**overrides) -> str:
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    kwargs = dict(
        rank=3,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="FAIL",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
        names={"034020": "두산에너빌리티"},
        material_types={"034020": "수주"},
    )
    kwargs.update(overrides)
    return render_card(row, **kwargs)


def test_contribution_is_zero_for_a_feature_already_at_baseline():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate().copy()
    row["llm_specificity"] = baseline["llm_specificity"]
    contrib = feature_contributions(model, row, baseline, COLS)
    assert np.isclose(contrib["llm_specificity"], 0.0)


def test_the_driving_feature_ranks_first():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert contrib.index[0] == "llm_specificity"
    assert contrib.iloc[0] > 0


def test_contributions_are_sorted_descending():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert list(contrib) == sorted(contrib, reverse=True)


def test_analogs_return_k_nearest_with_realised_returns():
    analogs = find_analogs(_train(), _candidate(), COLS, k=3)
    assert list(analogs.columns) == ["code", "date", "excess_return", "distance"]
    assert len(analogs) == 3
    assert analogs["distance"].is_monotonic_increasing


def test_analogs_are_actually_close_in_feature_space():
    train = _train()
    analogs = find_analogs(train, train.iloc[7].copy(), COLS, k=1)
    assert analogs.iloc[0]["distance"] < 1e-9  # 자기 자신이 최근접


def test_card_contains_the_essentials():
    card = _card()
    assert "034020 두산에너빌리티" in card
    assert "2026-08-17" in card  # 진입일
    assert "2026-10-13" in card  # 목표 청산일
    assert "llm_specificity" in card
    assert "체코 원전 2기 최종 계약" in card
    assert "유사 과거 사례" in card


def test_card_shows_article_counts_and_the_representative_article():
    card = _card()
    assert "기사 7건 · 매체 5곳 · 평소 대비 4.2배" in card
    assert "한국경제" in card
    assert "두코바니" in card  # 본문 200자


def test_counts_come_from_the_candidate_row_not_the_article_frame():
    """집계 수치는 ②층 피처다. load_article_cards 는 그것을 돌려주지 않는다."""
    row = _candidate().copy()
    row["article_count"] = 99.0
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="PASS",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "기사 99건" in card


def test_card_omits_the_counts_line_for_a_layer_without_news_features():
    """①층 후보 행에는 article_count 가 없다. 그 줄만 빠지고 카드는 나온다."""
    row = _candidate().drop(["article_count", "outlet_count", "article_ratio"])
    cols = ["tv_ratio", "llm_specificity"]
    train = _train()
    model, baseline = fit_final_model(train, cols)
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, cols),
        analogs=find_analogs(train, row, cols, k=3),
        articles=_articles(),
        gate_verdict="PASS",
        layer_label="① 수급 단독",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "기사 " not in card
    assert "체코 원전 2기 최종 계약" in card


def test_material_type_is_display_only():
    """모델 행렬에 없다. 카드에 문자열로만 딸려온다 (스펙 §6 피처 상한)."""
    assert "재료 유형 | 수주" in _card()
    assert "재료 유형" not in _card(material_types=None)


def test_card_never_hides_a_failed_verdict():
    """스펙 §10 — FAIL 을 숨기지 않는다. FAIL 모델의 카드는 추천이 아니다."""
    card = _card(gate_verdict="FAIL")
    assert "FAIL" in card
    assert "매매 신호가 아니다" in card
    assert "널을 이기지 못했다" in card


def test_passing_card_omits_the_warning_but_keeps_the_disclaimer():
    card = _card(gate_verdict="PASS")
    assert "널을 이기지 못했다" not in card
    assert "매매 신호가 아니다" in card


def test_card_does_not_leak_another_codes_article():
    assert "무관한 기사" not in _card()


def test_card_works_when_there_is_no_representative_article():
    articles = _articles()
    articles.loc[
        articles["code"] == "034020",
        ["article_id", "press", "title", "body200", "url"],
    ] = None
    card = _card(articles=articles)
    assert "대표 기사 없음" in card
    assert "기사 7건 · 매체 5곳" in card  # 집계 수치는 후보 행에서 오므로 살아 있다


def test_card_works_when_the_code_has_no_news_row_at_all():
    card = _card(articles=_articles().iloc[0:0])
    assert "근거 기사 없음" in card


def test_card_falls_back_to_bare_code_without_a_name_map():
    card = _card(names=None)
    assert "034020" in card
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_report_cards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report'`

- [ ] **Step 3: 카드 구현**

`src/swing_it/report/__init__.py`:

```python
"""리포트 계층 — 사람이 읽는 출력.

스펙 §1: 최종 매매 판단은 사람이 한다. 그래서 이 계층의 목적은 설득이 아니라
**근거 제시**다. 모델이 판정을 통과하지 못했으면 그 사실도 같이 제시한다.

하루 10건 규모를 전제로 한다 (스펙 §5).
"""
```

`src/swing_it/report/cards.py`:

```python
"""근거 카드 — 왜 걸렸는지, 얼마나 들 건지, 과거에 비슷한 게 어땠는지.

기여도는 '피처를 학습셋 중앙값으로 되돌리면 예측이 얼마나 줄어드는가'다.
상호작용 때문에 합이 총 예측과 정확히 맞지는 않는다. 카드에 그렇게 적는다.
정확한 가법 분해가 필요하면 SHAP 이 필요하지만, 하루 10건을 사람이 훑고
'무엇이 이걸 밀어올렸나'를 보는 데는 순위면 충분하다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DISCLAIMER = (
    "이 카드는 근거 제시이지 매매 신호가 아니다. 진입·청산·비중은 사람이 정한다."
)


def feature_contributions(
    model,
    row: pd.Series,
    baseline: pd.Series,
    feature_columns: Sequence[str],
) -> pd.Series:
    """피처를 하나씩 중앙값으로 되돌렸을 때의 예측 감소량. 내림차순."""
    columns = list(feature_columns)
    x = pd.DataFrame([row[columns].astype(float).to_dict()], columns=columns)
    full = float(model.predict(x)[0])

    out = {}
    for col in columns:
        alt = x.copy()
        alt[col] = float(baseline[col])
        out[col] = full - float(model.predict(alt)[0])
    return pd.Series(out).sort_values(ascending=False)


def find_analogs(
    train: pd.DataFrame,
    row: pd.Series,
    feature_columns: Sequence[str],
    *,
    k: int = 3,
) -> pd.DataFrame:
    """표준화 피처 공간에서 가장 가까운 과거 이벤트 ``k`` 건과 그 **실현** 초과수익."""
    columns = list(feature_columns)
    scale = train[columns].std(ddof=0).replace(0.0, 1.0)
    centre = train[columns].mean()
    target = (row[columns].astype(float) - centre) / scale
    scaled = (train[columns] - centre) / scale

    out = train[["code", "date", "excess_return"]].copy()
    out["distance"] = np.sqrt(((scaled - target) ** 2).sum(axis=1)).to_numpy()
    return out.nsmallest(k, "distance").reset_index(drop=True)


def _counts_line(row: pd.Series) -> str:
    """집계 수치는 ②층 **피처**에서 읽는다 — ``load_article_cards`` 가 아니라.

    같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다. ①층처럼 뉴스 피처가
    없는 층에서는 빈 문자열을 돌려주고, 카드는 그 줄만 생략한다.
    """
    parts = []
    if "article_count" in row and pd.notna(row["article_count"]):
        parts.append(f"기사 {int(row['article_count'])}건")
    if "outlet_count" in row and pd.notna(row["outlet_count"]):
        parts.append(f"매체 {int(row['outlet_count'])}곳")
    if "article_ratio" in row and pd.notna(row["article_ratio"]):
        parts.append(f"평소 대비 {row['article_ratio']:.1f}배")
    return " · ".join(parts)


def _format_articles(articles: pd.DataFrame, row: pd.Series) -> str:
    """집계 수치(후보 행) + 대표 기사 1건(``load_article_cards``).

    ``load_article_cards`` 는 (code, date) 당 대표 기사 한 건을 준다
    (``daily_news.primary_news_id`` 조인). 발행 시각은 없다 — 빅카인즈가
    일자 단위만 주기 때문이고, 운영 스키마와 갈라놓지 않기 위해 쓰지 않는다.
    """
    code = str(row["code"])
    date = pd.Timestamp(row["date"])
    counts = _counts_line(row)

    hit = (
        articles[(articles["code"] == code) & (articles["date"] == date)]
        if articles is not None and not articles.empty
        else articles
    )
    if hit is None or hit.empty:
        head = f"**근거 기사 없음**{' — ' + counts if counts else ''}"
        return head + "\n"

    art = hit.iloc[0]
    header = f"**근거 {counts}**" if counts else "**근거 기사**"
    header += f" (핫함 {art['hotness']:.2f} · 그날 {int(art['hotness_rank'])}위)"
    if pd.isna(art.get("article_id")):
        return header + "\n\n- 대표 기사 없음 (클러스터 대표를 못 고름)\n"
    return "\n".join(
        [
            header,
            "",
            f"- [{art['press']}] [{art['title']}]({art['url']})",
            "",
            f"> {art['body200']}",
            "",
        ]
    )


def render_card(
    row: pd.Series,
    *,
    rank: int,
    contributions: pd.Series,
    analogs: pd.DataFrame,
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    names: dict[str, str] | None = None,
    material_types: dict[str, str] | None = None,
    top_features: int = 5,
) -> str:
    """후보 한 건의 근거 카드 마크다운.

    ``material_types`` 는 표시 전용이다. LLM 이 뽑은 재료 유형을 모델 행렬에
    넣으면 원-핫 9개가 스펙 §6 의 피처 상한을 혼자 먹으므로, 사람이 읽는
    문자열로만 딸려 보낸다.
    """
    code = str(row["code"])
    heading = f"{code} {(names or {}).get(code, '')}".strip()
    event_date = pd.Timestamp(row["date"])

    lines = [
        f"### {rank}. {heading} — 예측 초과수익 {row['prediction']:+.2%}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 층 | {layer_label} |",
        f"| 이벤트일 | {event_date.date()} |",
    ]
    material = (material_types or {}).get(code)
    if material:
        lines.append(f"| 재료 유형 | {material} |")
    lines += [
        f"| 진입 | {pd.Timestamp(entry_date).date()} 시가 |",
        f"| 목표 청산 | {pd.Timestamp(exit_date).date()} 종가 (D+40, 중간 청산 없음) |",
        "",
    ]

    top = contributions.head(top_features)
    lines += [
        f"**왜 걸렸나 — 예측 기여도 상위 {len(top)}**",
        "",
        "| 피처 | 값 | 기여 |",
        "|---|---|---|",
    ]
    for feature, value in top.items():
        lines.append(f"| `{feature}` | {float(row[feature]):.3f} | {value:+.4f} |")
    lines += [
        "",
        "> 기여도 = 해당 피처를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량.",
        "> 피처 간 상호작용이 있어 합이 총 예측과 정확히 일치하지는 않는다. 순위로 읽어라.",
        "",
        _format_articles(articles, row),
        f"**유사 과거 사례 (피처 공간 최근접 {len(analogs)}건, 실현 수익)**",
        "",
        "| 종목 | 이벤트일 | 실현 초과수익 | 거리 |",
        "|---|---|---|---|",
    ]
    for _, a in analogs.iterrows():
        lines.append(
            f"| {a['code']} | {pd.Timestamp(a['date']).date()} | "
            f"{a['excess_return']:+.2%} | {a['distance']:.2f} |"
        )

    lines.append("")
    if gate_verdict != "PASS":
        lines += [
            "> **이 층은 음성대조 널을 이기지 못했다.** 아래 후보는 모델이 뽑았다는 사실 "
            "외에 어떤 통계적 근거도 없다.",
            "",
        ]
    lines += [f"**판정: {gate_verdict}** ({layer_label} 층). {DISCLAIMER}", ""]
    return "\n".join(lines)


def render_candidates(
    candidates: pd.DataFrame,
    *,
    model,
    baseline: pd.Series,
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    names: dict[str, str] | None = None,
    material_types: dict[str, str] | None = None,
) -> str:
    """후보 전체를 예측 내림차순으로 렌더한다.

    후보는 모두 같은 이벤트일이므로 진입·청산일이 하나씩만 필요하다.
    """
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    cards = [
        render_card(
            row,
            rank=rank,
            contributions=feature_contributions(model, row, baseline, feature_columns),
            analogs=find_analogs(train, row, feature_columns, k=3),
            articles=articles,
            gate_verdict=gate_verdict,
            layer_label=layer_label,
            entry_date=entry_date,
            exit_date=exit_date,
            names=names,
            material_types=material_types,
        )
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1)
    ]
    return "\n---\n\n".join(cards)
```

- [ ] **Step 4: 카드 테스트 통과 확인**

Run: `uv run pytest tests/test_report_cards.py -v`
Expected: 16 passed

- [ ] **Step 5: 렌더 결과 눈으로 확인**

Run:
```bash
uv run python - <<'PY'
from tests.test_report_cards import _card
print(_card())
PY
```

Expected: 아래와 같은 마크다운이 나온다 (숫자는 시드에 따라 달라진다).

````markdown
### 3. 034020 두산에너빌리티 — 예측 초과수익 +4.81%

| 항목 | 값 |
|---|---|
| 층 | ③ + LLM |
| 이벤트일 | 2026-08-14 |
| 재료 유형 | 수주 |
| 진입 | 2026-08-17 시가 |
| 목표 청산 | 2026-10-13 종가 (D+40, 중간 청산 없음) |

**왜 걸렸나 — 예측 기여도 상위 5**

| 피처 | 값 | 기여 |
|---|---|---|
| `llm_specificity` | 2.500 | +0.1893 |
| `article_count` | 7.000 | +0.0104 |
| `tv_ratio` | 4.600 | +0.0041 |
| `outlet_count` | 5.000 | +0.0009 |
| `article_ratio` | 4.200 | -0.0007 |

> 기여도 = 해당 피처를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량.
> 피처 간 상호작용이 있어 합이 총 예측과 정확히 일치하지는 않는다. 순위로 읽어라.

**근거 기사 7건 · 매체 5곳 · 평소 대비 4.2배** (핫함 0.81 · 그날 1위)

- [한국경제] [체코 원전 2기 최종 계약](https://x/1)

> 두산에너빌리티가 체코 두코바니 원전 2기 주기기 공급 계약을 체결했다.

**유사 과거 사례 (피처 공간 최근접 3건, 실현 수익)**

| 종목 | 이벤트일 | 실현 초과수익 | 거리 |
|---|---|---|---|
| 000013 | 2024-01-14 | +18.42% | 0.61 |
| 000007 | 2024-02-21 | +15.03% | 0.74 |
| 000002 | 2024-03-08 | +21.10% | 0.88 |

> **이 층은 음성대조 널을 이기지 못했다.** 아래 후보는 모델이 뽑았다는 사실 외에 어떤 통계적 근거도 없다.

**판정: FAIL** (③ + LLM 층). 이 카드는 근거 제시이지 매매 신호가 아니다. 진입·청산·비중은 사람이 정한다.
````

- [ ] **Step 6: 실패하는 전진 기록 테스트 작성**

`tests/test_forward_log.py`:

```python
import json

import numpy as np
import pandas as pd

from swing_it.report.forward_log import (
    append_forward_log,
    build_records,
    read_forward_log,
    score_forward_log,
)

COLS = ["tv_ratio", "llm_specificity"]


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["034020", "005930"],
            "date": pd.to_datetime(["2026-08-14", "2026-08-14"]),
            "tv_ratio": [4.6, 3.2],
            "llm_specificity": [2.5, 0.4],
            "prediction": [0.0481, 0.0192],
        }
    )


def _records(logged_at: str = "2026-08-17", **overrides) -> list[dict]:
    kwargs = dict(
        feature_columns=COLS,
        layer="supply_news_llm",
        gate_verdict="FAIL",
        logged_at=logged_at,
        entry_date="2026-08-17",
        exit_date="2026-10-13",
    )
    kwargs.update(overrides)
    return build_records(_candidates(), **kwargs)


def test_record_captures_features_and_context():
    rec = _records()[0]
    assert rec["code"] == "034020"
    assert rec["event_date"] == "2026-08-14"
    assert rec["logged_at"] == "2026-08-17"
    assert rec["layer"] == "supply_news_llm"
    assert rec["gate_verdict"] == "FAIL"
    assert rec["rank"] == 1
    assert rec["features"] == {"tv_ratio": 4.6, "llm_specificity": 2.5}
    assert rec["entry_date_planned"] == "2026-08-17"
    assert rec["exit_date_planned"] == "2026-10-13"
    assert "code_version" in rec


def test_records_are_ranked_by_prediction():
    assert {r["code"]: r["rank"] for r in _records()} == {"034020": 1, "005930": 2}


def test_append_writes_jsonl(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    assert append_forward_log(path, _records()) == 2
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["code"] == "034020"


def test_append_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "forward_log.jsonl"
    assert append_forward_log(path, _records()) == 2
    assert path.exists()


def test_rerunning_the_same_day_writes_nothing(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records()) == 0
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_existing_record_is_never_overwritten(tmp_path):
    """전진검증의 가치는 '그때 그렇게 예측했다'가 사후에 안 바뀌는 데 있다."""
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    tampered = _records()
    tampered[0]["prediction"] = 0.9999
    append_forward_log(path, tampered)
    log = read_forward_log(path)
    assert log[log["code"] == "034020"]["prediction"].tolist() == [0.0481]


def test_a_later_day_appends_a_new_row(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records("2026-08-18")) == 2
    assert len(read_forward_log(path)) == 4


def test_a_different_layer_is_a_different_record(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records(layer="supply_news")) == 2


def test_read_missing_file_returns_empty_frame(tmp_path):
    assert read_forward_log(tmp_path / "nope.jsonl").empty


def test_score_fills_realised_returns_where_available(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    labels = pd.DataFrame(
        {
            "code": ["034020"],
            "date": pd.to_datetime(["2026-08-14"]),
            "excess_return": [0.062],
        }
    )
    scored = score_forward_log(read_forward_log(path), labels).set_index("code")
    assert scored.loc["034020", "realized_excess"] == 0.062
    assert np.isnan(scored.loc["005930", "realized_excess"])


def test_score_on_an_empty_log_is_a_noop():
    assert score_forward_log(pd.DataFrame(), pd.DataFrame()).empty
```

- [ ] **Step 7: 테스트 실패 확인**

Run: `uv run pytest tests/test_forward_log.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report.forward_log'`

- [ ] **Step 8: 전진 기록 구현**

`src/swing_it/report/forward_log.py`:

```python
"""전진 기록 — 오늘의 후보와 그때의 피처를 영구 저장한다 (스펙 §9).

6개월 뒤 실전 검증을 가능하게 하는 유일한 장치다. walk-forward 는 과거
데이터를 시간순으로 잘라 쓴 것이라 '손 안 댄 창'이라고 부르기 어렵다.
오늘 기록해 6개월 뒤 채점하는 것만이 진짜 out-of-sample 이다.

**append-only.** 같은 (logged_at, layer, code, event_date) 는 조용히 건너뛴다.
모델을 고쳐 다시 돌려도 과거 예측은 남는다. 사후에 바뀔 수 있는 기록은
전진검증 자산이 아니다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

LOG_KEY: tuple[str, ...] = ("logged_at", "layer", "code", "event_date")


def build_records(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    layer: str,
    gate_verdict: str,
    logged_at: str,
    entry_date: str,
    exit_date: str,
    seed: int = 0,
    version: str | None = None,
) -> list[dict]:
    """후보 프레임을 전진 기록 레코드 목록으로 바꾼다."""
    from .. import __version__

    columns = list(feature_columns)
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    return [
        {
            "logged_at": logged_at,
            "layer": layer,
            "code": str(row["code"]),
            "event_date": str(pd.Timestamp(row["date"]).date()),
            "rank": rank,
            "prediction": float(row["prediction"]),
            "gate_verdict": gate_verdict,
            "entry_date_planned": entry_date,
            "exit_date_planned": exit_date,
            "features": {col: float(row[col]) for col in columns},
            "model_seed": int(seed),
            "code_version": version or __version__,
        }
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1)
    ]


def _key(record: dict) -> tuple:
    return tuple(record[field] for field in LOG_KEY)


def append_forward_log(path: str | Path, records: Sequence[dict]) -> int:
    """중복을 건너뛰고 JSONL 에 덧붙인다. 새로 쓴 건수를 돌려준다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple] = set()
    if target.exists():
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    seen.add(_key(json.loads(line)))

    written = 0
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            key = _key(record)
            if key in seen:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1
    return written


def read_forward_log(path: str | Path) -> pd.DataFrame:
    """전진 기록을 프레임으로 읽는다. 없으면 빈 프레임."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def score_forward_log(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """기록된 후보에 실현 초과수익을 붙인다. 아직 안 나온 건 ``NaN``.

    Args:
        labels: ``[code, date, excess_return]``. ``date`` 는 이벤트일.
    """
    if log.empty:
        return log
    out = log.copy()
    out["_event_date"] = pd.to_datetime(out["event_date"])
    realized = labels[["code", "date", "excess_return"]].rename(
        columns={"date": "_event_date", "excess_return": "realized_excess"}
    )
    realized["_event_date"] = pd.to_datetime(realized["_event_date"])
    out = out.merge(realized, on=["code", "_event_date"], how="left")
    return out.drop(columns=["_event_date"])
```

- [ ] **Step 9: 전진 기록 테스트 통과 확인**

Run: `uv run pytest tests/test_forward_log.py -v`
Expected: 12 passed

- [ ] **Step 10: `.gitignore` 확인**

Run: `grep -nE '^(data/|results/|\*\.db)' .gitignore`
Expected: 세 패턴이 모두 나온다. 하나라도 없으면 추가하고 함께 커밋한다 — 전진 기록은 종목·예측이 담긴 데이터라 절대 커밋하지 않는다(스펙 §8).

- [ ] **Step 11: 커밋**

```bash
git add src/swing_it/report tests/test_report_cards.py tests/test_forward_log.py
git commit -m "feat(report): 근거 카드 + append-only 전진 기록"
```

---
### Task 7: 층 비교 리포트와 CLI 2종

**Files:**
- Create: `src/swing_it/report/layers_md.py`
- Create: `src/swing_it/cli_layers.py`
- Create: `src/swing_it/cli_today.py`
- Modify: `pyproject.toml` — `[project.scripts]` 에 두 줄 추가
- Test: `tests/test_report_layers_md.py`

**Interfaces:**
- Consumes:
  - `swing_it.validate.compare.compare_layers(ds, spec, *, top_n=15, n_folds=4, min_train=200, n_trials=200, n_boot=2000, cost_one_way=0.0023, hold_days=40, seed=0) -> dict` — 키 `rows, deltas, llm, pead, sample`
  - `swing_it.adapters.db.connect(dsn=None)`, `load_prices(conn, start, end)`, `load_supply_demand(conn, start, end)`, `load_stock_master(conn)`
  - `swing_it.adapters.news_db.connect_news(path=None)`, `load_article_cards(conn, start, end, *, ops_only=False)`
  - `swing_it.features.news.NEWS_FEATURE_COLUMNS`, `load_news_features(path, *, start, end)`
  - `swing_it.features.llm.LLM_FEATURE_COLUMNS`, `load_llm_features(cache_path, *, news_path, model, prompt_version)`, `available_models(conn)` (계획 3 Task 8)
  - `swing_it.layers.build_layered_dataset(...)`, `layer_features(...)`, `LAYER_LABELS`
  - `swing_it.model.layered.fit_final_model(...)`
  - `swing_it.report.cards.render_candidates(...)`
  - `swing_it.report.forward_log.build_records(...)`, `append_forward_log(...)`
- Produces:
  - `render_layer_report(comparison: dict, *, baseline_reference: dict | None = None, preregistration: dict | None = None) -> str`
  - `swing_it.cli_layers.main(argv: list[str] | None = None) -> int`
  - `swing_it.cli_today.main(argv: list[str] | None = None) -> int`

`swing-layers` 는 **판정 결과와 무관하게 종료 코드 0** 을 낸다. FAIL 은 오류가 아니라 결과다(스펙 §10). 종료 코드로 FAIL 을 표현하면 CI 나 래퍼가 "실패했으니 다시 돌리자"는 유혹에 빠진다. 0 이 아닌 코드는 **실행 자체가 불가능할 때만** 낸다(DB 없음, 표본 없음).

- [ ] **Step 1: 실패하는 리포트 테스트 작성**

`tests/test_report_layers_md.py`:

```python
from swing_it.report.layers_md import render_layer_report

CHECKS_PASS = {
    "전체_널_대비": True, "모든_폴드_널_대비": True,
    "손_안_댄_최종창": True, "플라시보_FAIL": True,
}
CHECKS_FOLD_FAIL = {**CHECKS_PASS, "모든_폴드_널_대비": False}


def _row(layer, label, n_features, mean, verdict, overall, checks, ratio):
    return {
        "layer": layer, "label": label, "n_features": n_features,
        "strategy_mean": mean, "null_mean": 0.0002, "null_p95": 0.0044,
        "percentile": 98.5, "net_mean": mean - 0.0046, "verdict": verdict,
        "overall_verdict": overall,
        "mean_pool_per_date": 15 / ratio, "selection_ratio": ratio,
        "low_selectivity": ratio > 0.5,
        "p10": -0.081, "p10_gap": 0.019,
        "tail_capture_rate": 0.16, "tail_avoidance": 0.04,
        "detail": {"checks": checks, "downside": {"pool_p10": -0.100}},
    }


def _comparison(llm_verdict: str) -> dict:
    return {
        "rows": [
            _row("supply", "① 수급 단독", 7, 0.0031, "FAIL", "FAIL",
                 CHECKS_FOLD_FAIL, 0.62),
            _row("supply_news", "② + 규칙 뉴스", 12, 0.0088, "PASS", "PASS",
                 CHECKS_PASS, 0.31),
            _row("supply_news_llm", "③ + LLM", 17, 0.0091, "PASS", "PASS",
                 CHECKS_PASS, 0.31),
        ],
        "deltas": [
            {"layer": "supply_news", "baseline": "supply", "mean_diff": 0.0057,
             "ci_low": 0.0021, "ci_high": 0.0094, "p_greater": 0.998, "n_dates": 412},
            {"layer": "supply_news_llm", "baseline": "supply_news", "mean_diff": 0.0003,
             "ci_low": -0.0019, "ci_high": 0.0026, "p_greater": 0.61, "n_dates": 412},
        ],
        "llm": {
            "layer2_mean": 0.0088, "layer3_mean": 0.0091, "delta": 0.0003,
            "ci_low": -0.0019, "ci_high": 0.0026, "p_greater": 0.61,
            "layer3_overall": "PASS", "verdict": llm_verdict,
        },
        "pead": [
            {"layer": "supply", "sharpe_annual": 0.13, "t_stat": 0.16,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
            {"layer": "supply_news", "sharpe_annual": 0.38, "t_stat": 0.48,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
            {"layer": "supply_news_llm", "sharpe_annual": 0.39, "t_stat": 0.50,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
        ],
        "sample": {
            "n_rows": 8421, "n_codes": 913, "date_range": ["2019-03-04", "2026-06-20"],
            "n_folds": 4, "top_n": 15, "n_trials": 200, "n_boot": 2000,
            "cost_one_way": 0.0023, "hold_days": 40, "seed": 0,
        },
    }


def test_report_states_the_llm_verdict_up_front():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "LLM_NOT_JUSTIFIED" in md[:600]


def test_report_lists_every_layer_with_both_verdicts():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for label in ["① 수급 단독", "② + 규칙 뉴스", "③ + LLM"]:
        assert label in md
    assert "FAIL" in md
    assert "PASS" in md


def test_report_breaks_out_the_four_checks():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for check in ["전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"]:
        assert check in md


def test_report_shows_selection_ratio_so_low_selectivity_cannot_hide():
    """계획 1 폴드 3 은 풀의 62%를 담고도 '상위 15 선별'이라 불렸다."""
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "선별도" in md
    assert "selection_ratio" in md
    assert "0.62" in md
    assert "24.2건" in md  # 15 / 0.62


def test_report_warns_when_selection_ratio_exceeds_half():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "선택성이 낮다" in md
    assert "① 수급 단독" in md.split("선택성이 낮다")[1][:80]


def test_report_omits_the_warning_when_every_layer_is_selective():
    comparison = _comparison("LLM_NOT_JUSTIFIED")
    for row in comparison["rows"]:
        row["selection_ratio"] = 0.31
        row["mean_pool_per_date"] = 15 / 0.31
        row["low_selectivity"] = False
    assert "선택성이 낮다" not in render_layer_report(comparison)


def test_report_shows_downside_next_to_the_mean():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "하방 회피" in md
    assert "tail_capture_rate" in md
    assert "0.160" in md
    assert "p10" in md


def test_report_shows_the_delta_confidence_interval():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "-0.0019" in md


def test_report_includes_pead_comparison():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "PEAD" in md
    assert "LOSES_TO_PEAD" in md
    assert "0.80" in md


def test_report_records_preregistered_parameters():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for token in ["top_n", "15", "n_trials", "200", "0.0023"]:
        assert token in md


def test_report_prints_the_plan1_baseline_with_a_warning():
    md = render_layer_report(
        _comparison("LLM_NOT_JUSTIFIED"),
        baseline_reference={"strategy_mean": 0.00594, "verdict": "FAIL",
                            "n_events": 25559},
    )
    assert "계획 1 베이스라인" in md
    assert "25,559" in md
    assert "표본이 다르므로" in md


def test_report_prints_the_preregistration_decision():
    md = render_layer_report(
        _comparison("LLM_NOT_JUSTIFIED"),
        preregistration={"tv_multiple": 1.0, "mean_pool": 27.4, "n_events": 8421,
                         "decided_at": "2026-08-17"},
    )
    assert "사전등록" in md
    assert "tv_multiple" in md
    assert "27.4" in md


def test_report_does_not_soften_a_justified_verdict():
    md = render_layer_report(_comparison("LLM_JUSTIFIED"))
    assert "LLM_JUSTIFIED" in md
    assert "LLM_NOT_JUSTIFIED" not in md
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_report_layers_md.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report.layers_md'`

- [ ] **Step 3: 리포트 렌더러 구현**

`src/swing_it/report/layers_md.py`:

```python
"""층 비교 결과 마크다운 리포트.

판정을 맨 위에 둔다. FAIL 을 뒤쪽 표 안에 묻어두면 사람은 앞의 숫자만 보고
좋은 결과라고 읽는다. 스펙 §10 — 부정 결과도 성공이고, 성공은 명확히 보여야 한다.

계획 1 의 판정 문서 형식을 그대로 따른다: 네 검사를 개별 컬럼으로 펼치고,
전체 평균 하나로 판정하지 않았음을 리포트 자체가 증명하게 한다.
"""

from __future__ import annotations

_LLM_MESSAGE = {
    "LLM_JUSTIFIED": (
        "③층이 ②층을 유의하게 이겼고 ③층 자체도 종합 판정을 통과했다. "
        "LLM 피처 추출 비용을 계속 지불할 근거가 있다."
    ),
    "LLM_NOT_JUSTIFIED": (
        "③층이 ②층을 유의하게 이기지 못했다(또는 ③층이 종합 판정을 못 넘었다). "
        "**LLM 을 쓸 이유가 없다.** 규칙 기반 뉴스 피처까지가 이 데이터에서 나오는 전부다."
    ),
}

_CHECK_ORDER = ("전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL")


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def render_layer_report(
    comparison: dict,
    *,
    baseline_reference: dict | None = None,
    preregistration: dict | None = None,
) -> str:
    rows = comparison["rows"]
    deltas = comparison["deltas"]
    llm = comparison["llm"]
    sample = comparison["sample"]
    labels = {row["layer"]: row["label"] for row in rows}

    out = [
        "# 층별 기여도 판정",
        "",
        f"## 판정: {llm['verdict']}",
        "",
        _LLM_MESSAGE[llm["verdict"]],
        "",
        f"③ − ② = {llm['delta']:+.4f} "
        f"(95% CI {llm['ci_low']:+.4f} ~ {llm['ci_high']:+.4f}, "
        f"P(차이>0) = {llm['p_greater']:.3f}, ③층 종합 판정 {llm['layer3_overall']})",
        "",
        "## 표본",
        "",
        f"- 이벤트 {sample['n_rows']:,}건 · 종목 {sample['n_codes']:,}개 · "
        f"{sample['date_range'][0]} ~ {sample['date_range'][1]}",
        "- 세 층 모두 **같은 행**을 본다. 한 층에서라도 피처가 결측인 이벤트는 전부 제외됐다.",
        "",
        "### 사전등록 파라미터",
        "",
        f"`top_n={sample['top_n']}` · `n_folds={sample['n_folds']}` · "
        f"`n_trials={sample['n_trials']}` · `n_boot={sample['n_boot']}` · "
        f"`cost_one_way={sample['cost_one_way']}` · `hold_days={sample['hold_days']}` · "
        f"`seed={sample['seed']}`",
        "",
    ]

    if preregistration:
        out += [
            f"이벤트 정의는 성과를 보기 전에 확정했다 ({preregistration['decided_at']}): "
            f"`tv_multiple={preregistration['tv_multiple']}` — "
            f"dry-run 실측 하루 평균 풀 {preregistration['mean_pool']:.1f}건 · "
            f"총 {preregistration['n_events']:,}건 기준.",
            "",
        ]

    out += [
        "## 층별 종합 판정 (동일 폴드 · 동일 널)",
        "",
        "| 층 | 피처 | 전략 평균 | 비용차감 후 | "
        + " | ".join(_CHECK_ORDER)
        + " | 종합 |",
        "|---|---|---|---|" + "---|" * len(_CHECK_ORDER) + "---|",
    ]
    for row in rows:
        checks = row["detail"]["checks"]
        marks = " | ".join(_mark(checks[name]) for name in _CHECK_ORDER)
        out.append(
            f"| {row['label']} | {row['n_features']} | {_pct(row['strategy_mean'])} | "
            f"{_pct(row['net_mean'])} | {marks} | **{row['overall_verdict']}** |"
        )

    out += [
        "",
        "> 네 검사를 **모두** 통과해야 종합 PASS 다. 계획 1 은 전체 널 대비만 보면 PASS 였고 "
        "폴드별로 뜯으니 3/4 폴드가 랜덤과 구별되지 않았다. 플라시보만 부호가 반대다 — "
        "라벨을 섞었는데 PASS 면 누수이므로 FAIL 이 나와야 통과다.",
        "",
        "## 음성대조 널",
        "",
        "| 층 | 널 평균 | 널 p95 | 백분위 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {_pct(row['null_mean'])} | {_pct(row['null_p95'])} | "
            f"{row['percentile']:.1f} |"
        )
    out += [
        "",
        "> 널 평균·널 p95 가 세 층에서 같은 것이 정상이다. 같은 표본 위에서 같은 시드로 "
        "만든 같은 널이기 때문이다. 다르다면 표본이 어긋난 것이고 비교는 무효다.",
        "",
        "## 선별도",
        "",
        "| 층 | 하루 평균 풀 | selection_ratio |",
        "|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {row['mean_pool_per_date']:.1f}건 | "
            f"{row['selection_ratio']:.2f} |"
        )
    out += [
        "",
        "> `selection_ratio` = `top_n / mean_pool_per_date`. 계획 1 폴드 3 은 하루 평균 "
        "24.1건 중 상위 15 = **0.62** 였다. 뉴스 축이 들어오면 이 값이 내려가야 정상이다.",
        "",
    ]
    if any(row["low_selectivity"] for row in rows):
        offenders = ", ".join(
            row["label"] for row in rows if row["low_selectivity"]
        )
        out += [
            f"⚠️ **선택성이 낮다 — {offenders}.** `selection_ratio` 가 0.5 를 넘는다. "
            "후보 풀의 절반 이상을 담고 있으므로 이것을 '상위 선별'이라고 읽으면 안 된다. "
            "성과가 나오더라도 그것은 선별의 성과가 아니라 풀 자체의 성과에 가깝다.",
            "",
        ]

    out += [
        "## 하방 회피",
        "",
        "| 층 | 평균 | 선택분 p10 | 풀 p10 | p10 격차 | tail_capture_rate |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        detail = row["detail"]["downside"]
        out.append(
            f"| {row['label']} | {_pct(row['strategy_mean'])} | {_pct(row['p10'])} | "
            f"{_pct(detail['pool_p10'])} | {_pct(row['p10_gap'])} | "
            f"{row['tail_capture_rate']:.3f} |"
        )
    out += [
        "",
        "> **평균과 나란히 읽어라.** 계획 1 에서 모델이 값을 한 유일한 구간(폴드 3, 나쁜 "
        "레짐)에서 한 일은 상위를 잘 고른 게 아니라 최악의 꼬리를 피한 것이었다 — "
        "전체 −2.99% vs 선택분 −0.93%. 같은 구간의 평균 gap 은 나머지 세 폴드에서 "
        "0.0001 수준이었다.",
        "> **`tail_capture_rate`** = 날짜별 풀 하위 20%에 속한 이벤트를 선택분이 담은 비율. "
        "**낮을수록 좋다.** 무작위 선택이면 0.200 이다.",
        "> **p10 격차** = 선택분 하위 10% 분위 − 풀 하위 10% 분위. 양수면 하방이 얕다.",
        "",
        "## 층 간 증분 (날짜 페어드 부트스트랩)",
        "",
        "| 비교 | 평균 차이 | 95% CI | P(차이>0) | 날짜 수 |",
        "|---|---|---|---|---|",
    ]
    for delta in deltas:
        cur = labels.get(delta["layer"], delta["layer"])
        prev = labels.get(delta["baseline"], delta["baseline"])
        out.append(
            f"| {cur} − {prev} | {delta['mean_diff']:+.4f} | "
            f"{delta['ci_low']:+.4f} ~ {delta['ci_high']:+.4f} | "
            f"{delta['p_greater']:.3f} | {delta['n_dates']} |"
        )

    pead_rows = comparison["pead"]
    out += [
        "",
        "## PEAD 대비 (스펙 §7)",
        "",
        f"기준: kr-quant 검증 PEAD — 연율 Sharpe {pead_rows[0]['pead_sharpe']:.2f}, "
        f"t {pead_rows[0]['pead_t']:.2f}",
        "",
        "| 층 | 연율 Sharpe | t (겹침 보정) | 유효 표본 | 판정 |",
        "|---|---|---|---|---|",
    ]
    for row in pead_rows:
        out.append(
            f"| {labels.get(row['layer'], row['layer'])} | {row['sharpe_annual']:.2f} | "
            f"{row['t_stat']:.2f} | {row['n_effective']} | **{row['verdict']}** |"
        )
    out += [
        "",
        "> t 는 겹침 보정본이다. 보유 40거래일이 겹치므로 날짜별 관측은 독립이 아니고, "
        "보정 없이는 t 가 크게 부풀려진다. 유효 표본 = 겹치지 않는 블록 수.",
        "",
    ]

    if baseline_reference:
        out += [
            "## 계획 1 베이스라인 참고값",
            "",
            f"- 전략 평균 {_pct(baseline_reference['strategy_mean'])} · "
            f"종합 판정 {baseline_reference['verdict']} · "
            f"이벤트 {baseline_reference['n_events']:,}건",
            "",
            "> **표본이 다르므로 위 층별 표와 직접 비교하지 말 것.** 계획 1 은 기사 유무와 "
            "무관한 모든 거래대금 급증 이벤트를 쓰고, 여기 ①층은 세 층 피처가 모두 있는 "
            "이벤트만 쓴다.",
            "",
        ]

    return "\n".join(out)
```

- [ ] **Step 4: 리포트 테스트 통과 확인**

Run: `uv run pytest tests/test_report_layers_md.py -v`
Expected: 14 passed

- [ ] **Step 5: `swing-layers` CLI 구현**

`src/swing_it/cli_layers.py`:

```python
"""`swing-layers` — 세 층을 동일 조건에서 비교하고 판정을 출력한다.

**FAIL 이어도 종료 코드는 0 이다.** FAIL 은 오류가 아니라 결과다(스펙 §10).
종료 코드로 FAIL 을 표현하면 래퍼나 CI 가 "실패했으니 파라미터 바꿔 재시도"
경로를 만들게 되고, 그게 정확히 이 프로젝트가 두 번 당한 실패 방식이다.
0 이 아닌 코드는 실행 자체가 불가능할 때만 낸다.

``--dry-run`` 은 성과를 계산하지 않고 표본 수와 하루 평균 풀 크기만 센다.
``tv_multiple`` 을 그 숫자만 보고 확정하기 위한 사전등록 절차다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from .adapters.db import connect, load_prices, load_supply_demand
from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
from .features.news import NEWS_FEATURE_COLUMNS, load_news_features
from .layers import build_layered_dataset, layer_features
from .report.layers_md import render_layer_report
from .validate.compare import compare_layers


def _load_llm(args) -> pd.DataFrame:
    """계획 3 의 (종목, 일자) 접기 결과를 읽고 계획 4 가 안 쓰는 두 컬럼을 뗀다.

    ``n_articles`` 는 ②층의 ``article_count`` 와 같은 축이라 중복이고,
    ``material_type`` 은 원-핫 9개가 스펙 §6 상한을 혼자 먹으므로 모델에
    넣지 않는다. ``material_type`` 은 ``swing-today`` 가 카드 표시용으로만 쓴다.
    """
    frame = load_llm_features(
        args.llm_cache,
        news_path=args.news_db,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    return frame[["code", "date", *LLM_FEATURE_COLUMNS]]


def _build(args, llm, tv_multiple: float):
    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    news = load_news_features(args.news_db, start=args.start, end=args.end)
    print(
        f"가격 {len(prices):,} · 수급 {len(supply):,} · 뉴스피처 {len(news):,} · "
        f"LLM {len(llm):,}",
        file=sys.stderr,
    )
    return build_layered_dataset(
        prices,
        supply,
        news=news,
        news_columns=NEWS_FEATURE_COLUMNS,
        llm=llm,
        llm_columns=LLM_FEATURE_COLUMNS,
        adv_floor=args.adv_floor,
        tv_multiple=tv_multiple,
        hold_end=args.hold_end,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="층별 기여도 분리 판정")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--news-db", default="data/news.db")
    parser.add_argument("--llm-cache", default="data/llm_features.db")
    parser.add_argument("--model", default=None,
                        help="생략 시 캐시에서 성공 건수가 가장 많은 모델")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--list-models", action="store_true",
                        help="캐시의 (model, prompt_version, n_ok) 을 출력하고 끝낸다")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--cost-one-way", type=float, default=0.0023)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="표본 수만 세고 끝낸다 (사전등록용)")
    parser.add_argument("--baseline-json", default="results/baseline.json")
    parser.add_argument("--prereg-json", default="results/preregistration.json")
    parser.add_argument("--out-json", default="results/layers.json")
    parser.add_argument("--out-md", default="results/layers.md")
    args = parser.parse_args(argv)

    if args.top_n < 15:
        parser.error("top_n 하한은 15 다 (스펙 §2). 집중은 이미 죽은 영역이다.")
    if args.tv_multiple > 3.0:
        parser.error(
            "tv_multiple 을 3.0 초과로 올리지 마라. 풀을 좁히는 것은 뉴스 축의 역할이다 "
            "(docs/research/2026-08-17-baseline-verdict.md 시사점 1)."
        )

    if args.list_models:
        from .features.llm import available_models

        conn = sqlite3.connect(f"file:{Path(args.llm_cache)}?mode=ro", uri=True)
        try:
            for model, prompt_version, n_ok in available_models(conn):
                print(f"{model}\t{prompt_version}\t{n_ok:,}")
        finally:
            conn.close()
        return 0

    llm = _load_llm(args)

    if args.dry_run:
        for candidate in (3.0, 1.0):
            ds = _build(args, llm, candidate)
            pool = ds.groupby("date").size().mean() if not ds.empty else 0.0
            print(
                f"tv_multiple={candidate}: 이벤트 {len(ds):,}건 · "
                f"종목 {ds['code'].nunique() if not ds.empty else 0:,} · "
                f"하루 평균 풀 {pool:.1f}건"
            )
        print(
            "\n하루 평균 풀 ≥ 40 이고 총 이벤트 ≥ 2,000 이면 3.0, 아니면 1.0 을 골라 "
            f"{args.prereg_json} 에 기록하고 --tv-multiple 로 넘겨라.",
            file=sys.stderr,
        )
        return 0

    ds = _build(args, llm, args.tv_multiple)
    if ds.empty:
        print("세 층 피처가 모두 있는 이벤트가 없다. 계획 2·3 산출물을 확인하라.",
              file=sys.stderr)
        return 1
    print(f"비교 표본 {len(ds):,} 건 · 종목 {ds['code'].nunique():,}", file=sys.stderr)

    spec = layer_features(NEWS_FEATURE_COLUMNS, LLM_FEATURE_COLUMNS)
    comparison = compare_layers(
        ds,
        spec,
        top_n=args.top_n,
        n_folds=args.folds,
        n_trials=args.trials,
        n_boot=args.boot,
        cost_one_way=args.cost_one_way,
        hold_days=args.hold_end,
        seed=args.seed,
    )

    baseline_reference = None
    if Path(args.baseline_json).exists():
        raw = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        baseline_reference = {
            "strategy_mean": raw["strategy_mean"],
            "verdict": raw.get("overall_verdict", raw["verdict"]),
            "n_events": raw.get("n_events", 0),
        }
    preregistration = None
    if Path(args.prereg_json).exists():
        preregistration = json.loads(Path(args.prereg_json).read_text(encoding="utf-8"))

    payload = {
        **comparison,
        "baseline_reference": baseline_reference,
        "preregistration": preregistration,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    markdown = render_layer_report(
        comparison,
        baseline_reference=baseline_reference,
        preregistration=preregistration,
    )
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: `swing-today` CLI 구현**

`src/swing_it/cli_today.py`:

```python
"""`swing-today` — 오늘 후보의 근거 카드를 내고 전진 기록에 남긴다 (스펙 §1·§9).

층은 ``--layer`` 로 고른다. 기본은 ③이지만, `swing-layers` 판정이
``LLM_NOT_JUSTIFIED`` 였다면 ②를 쓰는 것이 옳다. 그 선택은 사람이 한다 —
판정에 따라 자동으로 층을 바꾸면 리포트를 안 읽게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .adapters.db import connect, load_prices, load_stock_master, load_supply_demand
from .adapters.news_db import connect_news, load_article_cards
from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
from .features.news import NEWS_FEATURE_COLUMNS, load_news_features
from .layers import LAYER_LABELS, build_layered_dataset, layer_features
from .model.layered import fit_final_model
from .report.cards import render_candidates
from .report.forward_log import append_forward_log, build_records


def _material_types(llm_full: pd.DataFrame, asof: pd.Timestamp) -> dict[str, str]:
    """기준일의 종목별 재료 유형. 카드 표시 전용이다 (모델 행렬에 없다)."""
    same_day = llm_full[llm_full["date"] == asof]
    return dict(zip(same_day["code"].astype(str), same_day["material_type"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="오늘 후보 + 근거 카드 + 전진 기록")
    parser.add_argument("--db", default=None)
    parser.add_argument("--news-db", default="data/news.db")
    parser.add_argument("--llm-cache", default="data/llm_features.db")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--asof", default=None, help="기준일. 생략 시 데이터 마지막 날")
    parser.add_argument("--layer", default="supply_news_llm", choices=list(LAYER_LABELS))
    parser.add_argument("--layers-json", default="results/layers.json")
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="data/forward_log.jsonl")
    parser.add_argument("--out-md", default=None, help="생략 시 results/today/<기준일>.md")
    args = parser.parse_args(argv)

    if args.top_n < 15:
        parser.error("top_n 하한은 15 다 (스펙 §2).")

    conn = connect(args.db)
    end = args.asof or str(pd.Timestamp.today().date())
    prices = load_prices(conn, args.start, end)
    supply = load_supply_demand(conn, args.start, end)
    master = load_stock_master(conn)
    names = dict(zip(master["code"], master["name"]))

    # 근거 카드용 대표 기사. ops_only=False — 후보는 학습 유니버스에서 나오므로
    # 운영 상위 10 밖의 종목도 카드를 받아야 한다.
    news_conn = connect_news(args.news_db)
    try:
        articles = load_article_cards(news_conn, args.start, end, ops_only=False)
    finally:
        news_conn.close()

    # material_type 은 모델에 넣지 않고 카드 표시용으로만 들고 간다 (스펙 §6).
    llm_full = load_llm_features(
        args.llm_cache,
        news_path=args.news_db,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    common = dict(
        news=load_news_features(args.news_db, start=args.start, end=end),
        news_columns=NEWS_FEATURE_COLUMNS,
        llm=llm_full[["code", "date", *LLM_FEATURE_COLUMNS]],
        llm_columns=LLM_FEATURE_COLUMNS,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        hold_end=args.hold_end,
    )
    train = build_layered_dataset(prices, supply, require_label=True, **common)
    pool = build_layered_dataset(prices, supply, require_label=False, **common)
    if train.empty or pool.empty:
        print("학습 표본 또는 후보 풀이 비었다.", file=sys.stderr)
        return 1

    asof = pd.Timestamp(args.asof) if args.asof else pool["date"].max()
    today = pool[pool["date"] == asof].copy()
    if today.empty:
        print(f"{asof.date()} 에 이벤트가 없다.", file=sys.stderr)
        return 1

    columns = layer_features(NEWS_FEATURE_COLUMNS, LLM_FEATURE_COLUMNS)[args.layer]
    model, baseline = fit_final_model(train, columns, seed=args.seed)
    today["prediction"] = model.predict(today[columns])
    candidates = today.nlargest(args.top_n, "prediction").reset_index(drop=True)

    verdict = "UNKNOWN"
    if Path(args.layers_json).exists():
        payload = json.loads(Path(args.layers_json).read_text(encoding="utf-8"))
        for row in payload["rows"]:
            if row["layer"] == args.layer:
                verdict = row["overall_verdict"]
    else:
        print("results/layers.json 이 없다. 먼저 swing-layers 를 돌려라.", file=sys.stderr)

    # 진입은 기준일 다음 거래일 시가, 목표 청산은 그로부터 hold_end 거래일 뒤.
    calendar = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
    idx = int(calendar[calendar == asof].index[0])
    entry = calendar.iloc[min(idx + 1, len(calendar) - 1)]
    exit_idx = idx + 1 + args.hold_end
    exit_date = (
        calendar.iloc[exit_idx]
        if exit_idx < len(calendar)
        # 달력에 아직 없는 미래는 영업일 근사로 표시한다
        else entry + pd.tseries.offsets.BDay(args.hold_end)
    )

    markdown = "\n".join(
        [
            f"# {asof.date()} 후보 — {LAYER_LABELS[args.layer]}",
            "",
            f"종합 판정 **{verdict}** · 상위 {len(candidates)}종목 / 풀 {len(today)}종목 · "
            f"학습 표본 {len(train):,}건",
            "",
            render_candidates(
                candidates,
                model=model,
                baseline=baseline,
                train=train,
                feature_columns=columns,
                articles=articles,
                gate_verdict=verdict,
                layer_label=LAYER_LABELS[args.layer],
                entry_date=entry,
                exit_date=exit_date,
                names=names,
                material_types=_material_types(llm_full, asof),
            ),
        ]
    )

    out_md = Path(args.out_md or f"results/today/{asof.date()}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")

    written = append_forward_log(
        args.log_path,
        build_records(
            candidates,
            feature_columns=columns,
            layer=args.layer,
            gate_verdict=verdict,
            logged_at=str(pd.Timestamp.today().date()),
            entry_date=str(pd.Timestamp(entry).date()),
            exit_date=str(pd.Timestamp(exit_date).date()),
            seed=args.seed,
        ),
    )
    print(f"{out_md} 작성 · 전진 기록 {written} 건 추가", file=sys.stderr)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: `pyproject.toml` 에 엔트리포인트 추가**

`[project.scripts]` 블록에 두 줄을 **더한다**. 기존 세 줄은 그대로 둔다.

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-news = "swing_it.cli_news:main"
swing-score = "swing_it.score.cli:main"
swing-layers = "swing_it.cli_layers:main"
swing-today = "swing_it.cli_today:main"
```

- [ ] **Step 8: 전체 테스트와 린트**

Run:
```bash
uv sync --extra dev
uv run pytest -v
uv run ruff check src tests
```
Expected: 계획 1~3 의 기존 테스트가 전부 통과하고, 계획 4 가 더한 **123개**가 통과한다
(Task 1: 22 · Task 2: 12 · Task 3: 9 · Task 4: 12 · Task 5: 26 · Task 6: 28 · Task 7: 14).
ruff `All checks passed!`

- [ ] **Step 9: 사전등록 — dry-run 으로 이벤트 정의 확정**

Run:
```bash
export $(grep -v '^#' ../kr-quant/.env | xargs)
uv run swing-layers --dry-run
```

출력된 **하루 평균 풀 크기**만 보고 `tv_multiple` 을 고른다. 성과 숫자는 아직 존재하지 않으므로 이 선택은 결과를 보고 하는 것이 아니다.

- 하루 평균 풀 ≥ 40건 **그리고** 총 이벤트 ≥ 2,000건 → `3.0`
- 그 외 → `1.0`

고른 값을 기록한다:

```bash
mkdir -p results
cat > results/preregistration.json <<'JSON'
{
  "tv_multiple": 1.0,
  "mean_pool": 27.4,
  "n_events": 8421,
  "decided_at": "2026-08-17",
  "rule": "하루 평균 풀 >= 40 이고 총 이벤트 >= 2000 이면 3.0, 아니면 1.0"
}
JSON
```

(`tv_multiple`·`mean_pool`·`n_events` 는 dry-run 이 실제로 출력한 값으로 채운다.)

- [ ] **Step 10: 층 비교 본 실행 — 한 번만**

Run:
```bash
uv run swing-layers --tv-multiple "$(python -c 'import json;print(json.load(open("results/preregistration.json"))["tv_multiple"])')"
```

Expected: `results/layers.md` 가 생성되고 맨 위에 `## 판정: LLM_JUSTIFIED` 또는 `## 판정: LLM_NOT_JUSTIFIED` 가 찍힌다. 종료 코드 0.

**`LLM_NOT_JUSTIFIED` 는 버그가 아니다.** 그건 이 계획이 답하려던 질문에 대한 답이다.
세 층 모두 종합 FAIL 이 나오는 것도 마찬가지다 — 계획 1 이 이미 FAIL 이었고, 스펙 §10 은
부정 결과도 성공이라고 못박았다. **파라미터를 바꿔 다시 돌리지 마라.** 사전등록 config 는
Step 9 에서 확정됐다.

- [ ] **Step 11: 오늘 후보 생성**

Run:
```bash
uv run swing-today --layer supply_news_llm
```
Expected: `results/today/<날짜>.md` 생성, `data/forward_log.jsonl` 에 15건 추가.
같은 명령을 한 번 더 돌리면 `전진 기록 0 건 추가` 가 나와야 한다(append-only 확인).

- [ ] **Step 12: 판정 문서 작성**

`docs/research/2026-08-17-layers-verdict.md` 를 계획 1 의
`docs/research/2026-08-17-baseline-verdict.md` 와 같은 형식으로 쓴다. `results/layers.md` 의
표를 옮기고, 다음 넷을 반드시 담는다.

1. **판정** — `LLM_JUSTIFIED` / `LLM_NOT_JUSTIFIED`, 그리고 층별 종합 판정 세 개
2. **선별도** — 커버리지가 계획 1 의 62%에서 얼마나 내려갔는가. 안 내려갔으면 그렇게 쓴다
3. **하방 회피** — 계획 1 폴드 3 이 시사한 축이 뉴스 층에서도 나타나는가
4. **PEAD 대비** — 이기든 지든 명시적으로 (스펙 §10-5)

- [ ] **Step 13: 커밋**

```bash
git add src/swing_it/report/layers_md.py src/swing_it/cli_layers.py \
  src/swing_it/cli_today.py tests/test_report_layers_md.py pyproject.toml \
  docs/research/2026-08-17-layers-verdict.md
git commit -m "feat(report): 층 비교 리포트 + swing-layers·swing-today CLI"
```

---

## Self-Review 체크

**스펙 커버리지 (계획 4 범위):**

| 스펙 요구 | 태스크 |
|---|---|
| §1 후보 리스트 + 근거 카드 | Task 6, Task 7 (`swing-today`) |
| §1 진입·목표 보유기간 표시 | Task 6 — 카드 표의 진입/목표 청산 행 |
| §1 유사 과거 사례 | Task 6 `find_analogs` |
| §2 top_n 하한 15 | Task 5 기본값, Task 7 CLI 가 15 미만 거부 |
| §2 손절 없음 | Task 2 — 라벨·카드 모두 중간 청산 개념이 없다 |
| §2 음성대조 필수 | Task 4 — 층마다 널 대조 4종 |
| §3 ①층 재사용 | Task 2 `layer_features["supply"] == FEATURE_COLUMNS` |
| §3 ②층 규칙 뉴스 | Task 1 `news_features` + Task 2 병합 |
| §3 ③층 LLM | Task 2 `llm_columns` 병합 |
| §3 ③이 ②를 못 이기면 LLM 을 쓸 이유가 없다 | Task 5 `llm_verdict` — 이 계획의 존재 이유 |
| §3 동일 조건 비교 | Task 2 (동일 표본) · Task 3 (동일 폴드) · Task 5 (동일 널, 테스트로 강제) |
| §4 '핫함'을 과거·현재 동일 계산 | Task 1 — 계획 2 의 `hotness` 를 그대로 쓴다. 재계산하지 않는다 |
| §6 얕은 모델 | Task 3 `MAX_DEPTH=2`, 테스트로 고정 |
| §6 사전등록 | 계획 상단 상수 블록 · Task 7 Step 9 dry-run 절차 · 리포트에 파라미터 기록 |
| §7 PEAD 대비 판정 | Task 5 `pead_benchmark` |
| §9 전진 기록 | Task 6 `forward_log` |
| §10-1 층별 기여도 분리 측정 | Task 5 `compare_layers` |
| §10-2 음성대조 승리 | Task 4 `전체_널_대비` + `모든_폴드_널_대비` |
| §10-3 walk-forward 재현 + 손 안 댄 최종창 | Task 4 `untouched_window` |
| §10-4 비용 견딤 | Task 4 `net_mean` (왕복 46bp) |
| §10-5 PEAD 대비 판정 명시 | Task 5 · Task 7 리포트 |
| §10 부정 결과도 성공 / FAIL 숨기지 않음 | Task 6 (카드에 판정 강제) · Task 7 (판정을 리포트 최상단에, 종료 코드 0, 재시도 경로 없음) |
| §8 데이터 커밋 금지 | Task 6 Step 10 — `.gitignore` 확인 |
| §11 수정주가 | 계획 1 `daily_bars_adjusted` 경로를 그대로 탄다 |
| §11 생존편향 | Task 6 — `load_stock_master` 가 폐지 종목을 포함한다 |

**계획 1 판정 문서가 남긴 숙제:**

| 숙제 | 태스크 |
|---|---|
| 전체 평균 하나로 판정하지 말 것 | Task 4 `layer_verdict` — 네 검사 전부 통과해야 PASS |
| 하방 회피 지표를 층 비교에 넣을 것 | Task 4 `downside_stats` (`p10` · `tail_capture_rate`), Task 5 `rows` 최상위, Task 7 리포트에서 평균과 나란히 |
| 후보 풀 대비 `top_n` 비율을 명시할 것 | Task 4 `selectivity` (`mean_pool_per_date` · `selection_ratio`), Task 7 리포트 표 + 0.5 초과 시 경고 문장 |

**이 계획이 재지 않는 것 (명시적으로 범위 밖):**

`portfolio_returns` 는 날짜당 상위 15 × 40일 보유라 실제로 담으면 동시 보유가 수백 종목이 된다. 시그널 품질 지표로는 타당하고 세 층이 같은 왜곡을 공유하므로 층 비교에는 그대로 쓰지만, **포지션 예산 시뮬레이션은 이 계획 범위 밖이며 어떤 층이든 승격을 검토하기 전에 별도 계획이 필요하다.** 문서 상단 「`portfolio_returns` 는 포트폴리오가 아니다」 절 참조.

**계획 4 범위 밖:** 실시간 모니터링, 진입 타이밍, 주문 실행, 조건검색, 뉴스 인제스트의 quant-airflow 이전.

**타입 정합성 확인:**

- `FEATURE_COLUMNS`(계획 1) → `layer_features` → `compare_layers` 의 `spec` → `walk_forward_predict_features` 의 `feature_columns`: 전부 `list[str]`, 이름 일치.
- `NEWS_FEATURE_COLUMNS`(Task 1) 는 `news_features` 출력 컬럼과 정확히 같은 이름·순서. `rank` → `hotness_rank` 개명이 `features/news.py` 와 `news_db.load_article_cards` 양쪽에서 동일.
- `portfolio_returns` 결과 `[date, n_positions, mean_excess]` → `paired_bootstrap`·`overlap_stats` 모두 `mean_excess` 를 읽는다.
- `gate_report` 반환 키 → `layer_verdict["gate"]` → `compare_layers` 의 `rows` 에 전개 → `render_layer_report` 가 같은 키로 읽음.
- `layer_verdict` 반환 `checks` 의 네 키 = `CHECK_NAMES` = `layers_md._CHECK_ORDER`. 세 곳의 문자열이 동일.
- `fit_final_model` 반환 `(model, baseline: pd.Series)` → `feature_contributions(model, row, baseline, feature_columns)` · `render_candidates(model=, baseline=)`: 순서·이름 일치.
- `load_article_cards` 반환 컬럼(`date, code, hotness_rank, hotness, article_id, press, title, body200, url`) = `cards._format_articles` 가 대표 기사 블록에서 읽는 것과 정확히 일치. 집계 수치는 여기서 오지 않고 `cards._counts_line` 이 후보 행의 ②층 피처(`article_count, outlet_count, article_ratio`)에서 읽는다 — 한 값이 한 경로로만 들어온다.
- `layer_verdict` 의 `selectivity`/`downside` 키 → `compare_layers` 가 `rows` 최상위로 끌어올리는 이름(`mean_pool_per_date, selection_ratio, low_selectivity, p10, p10_gap, tail_capture_rate, tail_avoidance`) → `render_layer_report` 가 읽는 이름: 세 곳 동일. `pool_p10` 만 `row["detail"]["downside"]` 에서 읽는다.
- 계획 3 `load_llm_features` 반환 `["code","date","material_type","n_articles",*LLM_FEATURE_COLUMNS]` → `cli_layers._load_llm` 과 `cli_today` 가 `[["code","date",*LLM_FEATURE_COLUMNS]]` 로 좁힌다. `material_type` 은 `cli_today._material_types` 만 쓰고 모델 행렬에 들어가지 않는다. `n_articles` 는 어디서도 쓰이지 않는다(②층 `article_count` 로 대체).
- `build_records` 의 `entry_date`/`exit_date` 는 `str`, `render_card` 의 `entry_date`/`exit_date` 는 `pd.Timestamp`. `cli_today` 가 각각에 맞는 형으로 변환해 넘긴다.
- `compare_layers` 는 `llm` 키에 `None` 을 넣을 수 있으나(2층 spec), `render_layer_report` 는 3층 입력을 전제한다. CLI 는 항상 `layer_features` 의 3층 spec 을 넘기므로 이 경로는 발생하지 않는다.

## 후속

- **`LLM_NOT_JUSTIFIED` 면:** 계획 3 을 운영 경로에서 내리고 ②층으로 `swing-today` 를 돌린다. 스펙 §10 의 성공 기준 1·5 에 대한 답으로 기록한다. LLM 비용 지출을 멈춘다.
- **세 층 모두 종합 FAIL 이면:** "뉴스 재료 강도는 한국 시장 2~8주 지평에서 재현되는 엣지가 아니다"가 결론이다. 기록하고 끝낸다(스펙 §10). 전진 기록은 계속 쌓되 운영에는 올리지 않는다. 계획 1 에 이은 두 번째 부정 결과이며, 그것도 결과다.
- **커버리지가 계획 1 의 62%에서 안 내려갔으면:** 뉴스 축이 후보 풀을 좁히는 데 실패했다는 뜻이다. 층 성과와 별개로 이것 자체가 기록할 발견이다.
- **6개월 뒤(2027-02 이후):** `score_forward_log` 로 전진 기록을 채점한다. 이것이 손 안 댄 유일한 창이다.
