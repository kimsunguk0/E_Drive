#!/usr/bin/env python3
"""Raw ETRI train scenarios -> the training inputs the loaders read.

Input  RAW/<scenario>.tar  (as distributed) or RAW/<scenario>/ (already extracted).
Output OUT/meta_train/<scenario>/{annotation,calibration,meta}/*.parquet   (copied as-is)
       OUT/etri_768/<scenario>/<camera>/<frame:08d>.jpg                     (768x432)
       OUT/etri_768/cache_meta.json

The image transform is the one that built the training cache: undistort -> crop
1920x1080 -> scale 0.4 -> JPEG q95. It is imported unchanged from
src/data_prep/etri_build_cache.py (new_intrinsic, build_maps, crop_of, CROP, QUALITY).
Existing outputs are kept, so an interrupted run can be restarted.
"""
import argparse, json, os, sys, tarfile, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/data_prep'))
from etri_build_cache import CROP, QUALITY, build_maps, crop_of, new_intrinsic  # noqa: E402

SCALE = 0.4
PARQUET_DIRS = ('annotation', 'calibration', 'meta')


def members(src):
    """Yield (relative path, bytes reader) for every file of one scenario."""
    if src.suffix == '.tar':
        with tarfile.open(src, 'r|') as tf:
            for m in tf:
                if m.isfile():
                    yield m.name.split('/', 1)[1], (lambda m=m, tf=tf: tf.extractfile(m).read())
    else:
        for p in sorted(src.rglob('*')):
            if p.is_file():
                yield str(p.relative_to(src)), p.read_bytes


def scenario(job):
    src, out = job
    scen = src.name[:-4] if src.suffix == '.tar' else src.name
    cv2.setNumThreads(1)
    meta_dir, img_dir = out / 'meta_train' / scen, out / 'etri_768' / scen
    maps, info, n_img, pending = {}, {}, 0, []

    def calibrate():
        cal = pq.read_table(meta_dir / 'calibration/calibration.parquet').to_pydict()
        for i, name in enumerate(cal['camera_name']):
            K = np.asarray(cal['K'][i], np.float64).reshape(3, 3)
            dist = np.asarray(cal['distortion'][i], np.float64)
            fe = bool(cal['is_fisheye'][i])
            size = (int(cal['image_width'][i]), int(cal['image_height'][i]))
            nk = new_intrinsic(K, dist, size, fe)
            maps[name] = build_maps(K, dist, nk, size, fe)
            ox, oy = crop_of(name, *size)
            k_cache = np.diag([SCALE, SCALE, 1.]) @ np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1.]]) @ nk
            info[name] = dict(crop=[ox, oy, *CROP], scale=SCALE,
                              out_size=[int(CROP[0] * SCALE), int(CROP[1] * SCALE)],
                              is_fisheye=fe, new_K=nk.tolist(), K_cache=k_cache.tolist())
            (img_dir / name).mkdir(parents=True, exist_ok=True)

    def image(rel, data):
        cam, fn = rel.split('/')[-2:]
        dst = img_dir / cam / fn
        if not dst.exists():
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            und = cv2.remap(img, *maps[cam], cv2.INTER_LINEAR)
            ox, oy = crop_of(cam, img.shape[1], img.shape[0])
            small = cv2.resize(und[oy:oy + CROP[1], ox:ox + CROP[0]],
                               (int(CROP[0] * SCALE), int(CROP[1] * SCALE)), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            assert ok, dst
            tmp = dst.with_suffix('.part'); tmp.write_bytes(enc.tobytes()); os.replace(tmp, dst)
        return 1

    for rel, read in members(src):
        top = rel.split('/')[0]
        if top in PARQUET_DIRS:
            dst = meta_dir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(read())
        elif rel.endswith('.jpg') and top.startswith('camera_'):
            # A streamed tar may list images before calibration.parquet: hold them until it arrives.
            if maps:
                n_img += image(rel, read())
            else:
                pending.append((rel, read()))
        if pending and not maps and (meta_dir / 'calibration/calibration.parquet').exists():
            calibrate()
            n_img += sum(image(r, d) for r, d in pending); pending = []
    if not maps:
        calibrate()
    n_img += sum(image(r, d) for r, d in pending)
    return scen, n_img, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='first N scenarios only (for a quick check)')
    a = ap.parse_args()
    srcs = sorted(p for p in a.raw.iterdir() if p.suffix == '.tar' or (p.is_dir() and (p / 'calibration').is_dir()))
    if a.limit:
        srcs = srcs[:a.limit]
    print(f'{len(srcs)} scenarios -> {a.out}', flush=True)
    (a.out / 'etri_768').mkdir(parents=True, exist_ok=True)
    meta_path = a.out / 'etri_768/cache_meta.json'
    all_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t0, total = time.time(), 0
    with Pool(a.jobs) as pool:
        for i, (scen, n, info) in enumerate(pool.imap_unordered(scenario, [(s, a.out) for s in srcs]), 1):
            all_meta[scen] = info; total += n
            print(f'  {i}/{len(srcs)} {scen}: {n} images  ({time.time() - t0:.0f} s)', flush=True)
            meta_path.write_text(json.dumps(all_meta))
    print(f'done: {total} images, {len(all_meta)} scenarios', flush=True)


if __name__ == '__main__':
    main()
