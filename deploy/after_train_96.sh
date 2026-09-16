#!/usr/bin/env bash
# 96번 학습이 끝나면: 체크포인트 선택 → 정책 내보내기 → 매매 기준 해설.
# 세션이 끊겨도 산출물이 남게 하는 게 목적이다(94번 때 배운 것).
set -u
P=~/of80/.rlvenv/bin/python
W=~/git/scalp-it/deploy/research_run.sh
LOG=~/of80/after_train_96.log
cd ~/git/scalp-it || exit 1
# 시드 3개가 모두 마지막 체크포인트(40M)를 낼 때까지 기다린다.
until [ "$(ls ~/of80/rl96/seed19*/ckpt_40M.pt 2>/dev/null | wc -l)" -ge 3 ]; do sleep 120; done
sleep 30
$W $P -W ignore scripts/rl_96/dqn.py --pick                     >> "$LOG" 2>&1
$W $P -W ignore scripts/parity/export_policy.py --study rl96 --out ~/of80/policy_rl96 >> "$LOG" 2>&1
$W $P -W ignore scripts/parity/policy_explain.py --study rl96 --out ~/of80/rl96_criteria.md >> "$LOG" 2>&1
echo "after_train_96 done $(date)" >> "$LOG"
