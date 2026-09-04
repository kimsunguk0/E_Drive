#!/usr/bin/env bash
# Docker 이미지(etri-vad:cu128)를 rootfs로 재구성한다. docker/podman 없이.
#
# 이미지는 OCI 레이아웃(blobs/sha256/... + manifest.json)이다. 레이어를
# **manifest 순서대로** 겹쳐 풀어야 한다 -- 순서를 바꾸면 나중 레이어가 덮어써야 할
# 파일이 이전 값으로 남는다.
#
# whiteout 처리: 상위 레이어가 파일을 지우면 `.wh.<name>` 이라는 빈 파일로 표시된다.
# 그냥 풀면 삭제된 파일이 되살아나고 `.wh.*` 쓰레기가 남으므로, 레이어마다 풀고 나서
# 즉시 해석한다. `.wh..wh..opq`는 그 디렉터리의 하위 전체를 비우라는 뜻이다.
#
#   bash scripts/etri_docker_unpack.sh [이미지디렉터리] [출력rootfs]
set -euo pipefail
SRC=${1:-/tmp/pm97/docker_etri}
OUT=${2:-/tmp/pm97/docker_etri/rootfs}
mkdir -p "$OUT"

mapfile -t LAYERS < <(/tmp/pm97/mamba/envs/drive/bin/python -c "
import json,sys
m=json.load(open('$SRC/manifest.json'))
print('\n'.join(m[0]['Layers']))
")
echo "레이어 ${#LAYERS[@]}개를 $OUT 에 순서대로 적용"

i=0
for L in "${LAYERS[@]}"; do
  i=$((i+1))
  printf "  [%2d/%2d] %s\n" "$i" "${#LAYERS[@]}" "${L##*/}"
  tar -xf "$SRC/$L" -C "$OUT" --no-same-owner --no-same-permissions 2>/dev/null || true

  # opaque whiteout: 이 디렉터리의 기존 내용을 전부 지우라는 표시
  find "$OUT" -name '.wh..wh..opq' -print0 2>/dev/null | while IFS= read -r -d '' m; do
    d=$(dirname "$m"); rm -f "$m"
    find "$d" -mindepth 1 -maxdepth 1 ! -newer "$d" -exec rm -rf {} + 2>/dev/null || true
  done
  # 일반 whiteout: .wh.foo -> foo 삭제
  find "$OUT" -name '.wh.*' -print0 2>/dev/null | while IFS= read -r -d '' m; do
    b=$(basename "$m"); rm -rf "$(dirname "$m")/${b#.wh.}"; rm -f "$m"
  done
done

echo "완료: $(du -sh "$OUT" | cut -f1)"
ls -1 "$OUT" | head -20
