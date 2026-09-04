#!/usr/bin/env bash
# 주최측 Docker 환경(회수본)을 빈 경로에 복구한다. 해제 -> 재지정 -> 검증까지 전자동.
#
# 왜 재지정이 필요한가: venv의 pyvenv.gcfg와 bin/python은 컨테이너 안의
# `/opt/uv-python/cpython-3.10.20-linux-x86_64-gnu`를 가리킨다. 이미지 밖에서는 그
# 경로가 없으므로 그대로 두면 시스템 python으로 조용히 흘러가 **다른 버전으로 도는데
# 에러는 안 난다**. 그게 최악이라 복구 절차에 재지정을 못 박아 넣는다.
#
# 인터프리터도 함께 박제해야 한다 (uv-python.tar.gz). venv만 있으면 복구 불가.
#
#   bash scripts/setup_env_etri.sh /복구할/경로
#   bash scripts/setup_env_etri.sh /복구할/경로 --skip-verify
set -euo pipefail
SRC=/NHNHOME/data/sukim/adcl/env_etri_extracted
DEST=${1:?"복구 경로를 인자로 주세요"}
SKIP=${2:-}

echo "== 0. 아카이브 무결성 =="
( cd "$SRC" && sha256sum -c venv_ref.tar.gz.sha256 )
[ -f "$SRC/uv-python.tar.gz" ] && ( cd "$SRC" && sha256sum -c uv-python.tar.gz.sha256 )

mkdir -p "$DEST"
echo "== 1. 해제 -> $DEST =="
tar -xzf "$SRC/venv_ref.tar.gz" -C "$DEST"
if [ -f "$SRC/uv-python.tar.gz" ]; then
  tar -xzf "$SRC/uv-python.tar.gz" -C "$DEST"
else
  cp -a "$SRC/uv-python" "$DEST/uv-python"
fi
mv "$DEST/venv_ref" "$DEST/venv"
chmod -R u+w "$DEST/venv"

PY="$DEST/uv-python/cpython-3.10.20-linux-x86_64-gnu"
echo "== 2. 인터프리터 재지정 =="
sed -i "s#^home = .*#home = $PY/bin#" "$DEST/venv/pyvenv.cfg"
rm -f "$DEST/venv/bin/python"
ln -s "$PY/bin/python3.10" "$DEST/venv/bin/python"
# console script들의 셔뱅도 원래 /opt/venv를 가리킨다. 새 경로로 고친다.
grep -rl '^#!/opt/venv/bin' "$DEST/venv/bin" 2>/dev/null \
  | xargs -r sed -i "1c\\#!$DEST/venv/bin/python" 2>/dev/null || true

echo "== 3. 확인 =="
"$DEST/venv/bin/python" -c "import sys; print(sys.version.split()[0], sys.prefix)"

if [ "$SKIP" != "--skip-verify" ]; then
  echo "== 4. 검증 =="
  "$DEST/venv/bin/python" \
    /NHNHOME/data/sukim/adcl/scripts/verify_etri_env.py
fi
echo "복구 완료: $DEST/venv/bin/python"
