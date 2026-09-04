#!/usr/bin/env bash
# 한눈에 보는 현황: 챌린지 다운로드 + stage1 학습.
#
# 사람이 직접 치는 용도이므로 의존성 없이 ls/grep/tail만 쓴다. 워치독이나
# 다운로더를 건드리지 않는 읽기 전용이라 아무 때나 몇 번이든 돌려도 안전하다.
#
#   bash scripts/status.sh           한 번 출력
#   bash scripts/status.sh -w        5초마다 갱신 (Ctrl+C로 종료)
#   bash scripts/status.sh -w 5m     5분마다
#   bash scripts/status.sh -w 30s    30초마다
#   bash scripts/status.sh -w 2h     2시간마다
#
# 긴 주기로 볼 때는 clear로 화면을 지우면 직전 값이 사라져 변화를 못 본다.
# 그래서 60초를 넘기면 지우지 않고 아래로 쌓아, 스크롤만으로 추이를 읽게 한다.
set -u
P=/home/pm97/workspace/sukim/adcl
D=$P/data/challenge
RLOG=$P/logs/rclone_challenge.log

show() {
  # 서버는 UTC. 사람이 읽는 줄은 KST로 맞춘다 (+9h). 로그 파일 안의
  # 타임스탬프는 rclone/mmdet이 찍은 UTC 그대로라 9시간 차이가 난다.
  echo "===== $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S') KST ====="

  echo
  echo "[다운로드]"
  # 원격 개수는 가이드라인·rclone lsf로 확인된 고정값. 분모를 하드코딩해야
  # 매번 API를 때리지 않고도 진행률을 보여줄 수 있다.
  for pair in "baseline 2" "baseline(docker) 1" "train 376" "test 1125"; do
    d=${pair% *}; tot=${pair##* }
    n=$(ls -1 "$D/$d" 2>/dev/null | grep -vc '\.part$')
    printf "  %-18s %5d / %-5s  %3d%%\n" "$d" "$n" "$tot" "$((n * 100 / tot))"
  done
  printf "  %-18s %s\n" "합계 용량" "$(du -sh "$D" 2>/dev/null | cut -f1)"

  if pgrep -u "$USER" -x rclone >/dev/null 2>&1; then
    echo "  상태: 실행 중"
  else
    echo "  상태: ** 멈춤 ** (아래 재개 명령 참고)"
  fi
  # --stats-one-line 덕분에 마지막 진행률 줄 하나가 곧 요약이다.
  # rclone이 찍은 UTC 타임스탬프와 INFO 접두어는 떼낸다. 헤더에 이미 KST가
  # 있으므로 여기에 UTC를 같이 보여주면 9시간 차이로 혼동만 준다.
  tail -n 40 "$RLOG" 2>/dev/null | grep -F 'ETA' | tail -1 \
    | sed -E 's#^[0-9/]+ [0-9:]+ +INFO *: *##; s/^/  /'

  # 조용한 실패를 놓치지 않기 위해 항상 띄운다. 0이면 0이라고 말해준다.
  # `-ciE ... | tail -1`: 이 시스템의 grep은 ugrep이고, 대체 패턴에 -c를 주면
  # 패턴별로 한 줄씩 뱉어 변수에 여러 줄이 담긴다.
  err=$(grep -ciE 'error|quota|failed' "$RLOG" 2>/dev/null | tail -1)
  echo "  오류/쿼터 누적: $err"

  echo
  echo "[stage1 학습]"
  lg=$(ls -t /tmp/pm97/work_dirs/stage1/*.log 2>/dev/null | head -1)
  if [ -z "$lg" ]; then
    echo "  로그 없음"
  else
    it=$(grep -oP 'Iter \[\d+/43900\]' "$lg" | tail -1)
    eta=$(grep -oP 'eta: [^,]*' "$lg" | tail -1)
    # nan은 발산, inf는 GradScaler가 정상 처리한 fp16 오버플로 -- 구분해서 센다.
    echo "  ${it:-?}   ${eta:-}"
    echo "  nan $(grep -c 'grad_norm: nan' "$lg") / inf $(grep -c 'grad_norm: inf' "$lg")  (inf는 정상: GradScaler가 step 스킵)"
    if pgrep -u "$USER" -f 'tools/train.py' >/dev/null 2>&1; then
      echo "  상태: 실행 중"
    else
      echo "  상태: ** 멈춤 **"
    fi
  fi
  echo
}

if [ "${1:-}" = "-w" ]; then
  arg=${2:-5}
  case "$arg" in
    *h) secs=$(( ${arg%h} * 3600 )) ;;
    *m) secs=$(( ${arg%m} * 60 )) ;;
    *s) secs=${arg%s} ;;
    *[!0-9]*) echo "간격을 못 읽음: $arg  (예: 30s, 5m, 2h, 또는 초 단위 숫자)" >&2
              exit 1 ;;
    *) secs=$arg ;;
  esac
  [ "$secs" -ge 1 ] 2>/dev/null || { echo "간격은 1초 이상" >&2; exit 1; }
  echo "${secs}초마다 갱신 — Ctrl+C로 종료"
  while true; do
    [ "$secs" -le 60 ] && clear
    show
    sleep "$secs"
  done
else
  show
fi
