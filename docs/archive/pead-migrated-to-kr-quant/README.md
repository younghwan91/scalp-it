# PEAD·저변동성·결합 북 — kr-quant 로 이관됨 (2026-08-24)

이 폴더의 문서들은 scalp-it 에서 진행했던 **저빈도 팩터 연구**의 기록이다.
scalp-it 은 만쥬 초단타·틱 스캘핑 전용 레포라, 분기·월 단위 저회전 팩터
(PEAD·저변동성·결합 북)는 **`kr-quant`** 로 옮겼다 (그쪽이 검증 프레임워크 레포).

- 이관 대상 코드: `strategies/lowvol·combo·hedge`, `features/volatility` 등 (kr-quant `feat/lowvol-and-combo`)
- 실전 배포 형태: **롱온리 + 인버스 ETF 헤지** (사용자 개별주 공매도 불가)
- 주의: 저변동·결합은 아직 kr-quant 정식 게이트 미통과 — 재심사 대상.
  scalp-it 검증(24번, 롱온리 t≈0.48)과 kr-quant 재현(롱온리 강함)이 신호정의 차이로 상충 → kr-quant 에서 확정.

여기 문서는 그 연구의 원본 사전등록·판정 기록이다. 코드는 git 히스토리 + kr-quant 에 있다.
