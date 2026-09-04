#!/usr/bin/env bash
# venv를 두 벌로 분리: venv_ref(원본 보존·수정 금지) / venv_dev(실험용).
#
# 왜 두 벌인가: 실험 중 패키지를 하나 건드리면 "주최측 환경과 같은가?"를 다시 증명할
# 수 없다. ref는 해시 목록과 tar.gz로 박제해 재현성 증빙으로만 쓰고, 실제 실행은 dev에서
# 한다. ref를 읽기 전용으로 만들어 실수로 쓰는 것을 물리적으로 막는다.
set -euo pipefail
P=/home/pm97/workspace/sukim/adcl/env_etri_extracted
cd "$P"

if [ -d venv ] && [ ! -d venv_ref ]; then
  mv venv venv_ref
fi
[ -d venv_dev ] || cp -a venv_ref venv_dev

# 해시 목록: 크기까지 같이 남긴다. 해시만 있으면 어느 파일이 바뀌었는지 알 수 있지만
# 목록 자체가 잘렸는지는 알 수 없다.
if [ ! -s venv_ref.sha256 ]; then
  ( cd venv_ref && find . -type f -print0 | sort -z \
      | xargs -0 -P 8 sha256sum ) > venv_ref.sha256
fi
wc -l venv_ref.sha256

if [ ! -s venv_ref.tar.gz ]; then
  tar -czf venv_ref.tar.gz.part -C "$P" venv_ref && mv venv_ref.tar.gz.part venv_ref.tar.gz
fi
ls -l venv_ref.tar.gz
sha256sum venv_ref.tar.gz > venv_ref.tar.gz.sha256
chmod -R a-w venv_ref 2>/dev/null || true
echo "완료: venv_ref(읽기전용) / venv_dev(실험용) / venv_ref.sha256 / venv_ref.tar.gz"
