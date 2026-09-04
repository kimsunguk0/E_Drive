#!/usr/bin/env python
"""99,000 프레임 metadata 전수 preflight (CPU 전용, GPU 학습과 병행 가능).

왜: 0-GT 관련 크래시를 8-scene에서 못 잡고 full330 200 iter에서 터뜨렸다
(gt_names 완전 빈 프레임 623개, np.array([])가 float64). 더 희귀한 형태가 남아
25시간 런을 중간에 죽일 수 있다. 전수 스캔으로 미리 잡는다.
"""
import collections, pickle, sys
import numpy as np

PKL = "/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl"
PC = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
CLS = ("Car", "Pedestrian", "Cyclist")
QUEUE, SI = 7, 5

d = pickle.load(open(PKL, "rb"))
infos, lanes = d["infos"], d["metadata"].get("map_lanes", {})
print(f"{PKL}\n  infos {len(infos):,}  map_lanes {len(lanes)}")
c = collections.Counter()
bad = collections.defaultdict(list)


def note(k, j, lim=5):
    c[k] += 1
    if len(bad[k]) < lim:
        bad[k].append(j)


byscen = collections.defaultdict(list)
for j, i in enumerate(infos):
    byscen[i["scene_token"]].append(int(i["frame_idx"]))
    nm = np.asarray(i["gt_names"])
    vf = np.asarray(i["valid_flag"])
    bx = np.asarray(i["gt_boxes"], dtype=np.float64) if len(nm) else np.zeros((0, 9))
    if len(nm) == 0:
        note("gt_names 완전 빔", j)
    if len(vf) and not vf.any():
        note("valid_flag 전부 False", j)
    if len(bx) and not np.isfinite(bx).all():
        note("gt_boxes non-finite", j)
    if len(nm) and len(nm) != len(vf):
        note("gt_names/valid_flag 길이 불일치", j)
    ninr = sum((n in CLS) and PC[0] <= b[0] <= PC[3] and PC[1] <= b[1] <= PC[4]
               for n, b in zip(nm[vf.astype(bool)] if len(vf) else nm,
                               bx[vf.astype(bool)] if len(vf) else bx))
    if ninr == 0:
        note("range 안 박스 0개", j)
    for key, shape in (("gt_ego_fut_trajs", 12), ("gt_ego_fut_masks", 6),
                       ("gt_ego_fut_cmd", 3), ("gt_ego_fut_goal", 2)):
        v = np.asarray(i[key], dtype=np.float64).ravel()
        if v.size != shape:
            note(f"{key} shape {v.size}!={shape}", j)
        if not np.isfinite(v).all():
            note(f"{key} non-finite", j)
    if int(np.asarray(i["gt_ego_fut_masks"]).sum()) != 6:
        note("ego_fut_masks 합 != 6", j)
    cmd = np.asarray(i["gt_ego_fut_cmd"]).ravel()
    if abs(cmd.sum() - 1.0) > 1e-6:
        note("cmd one-hot 아님", j)
    for key in ("ego2global_translation", "ego2global_rotation", "can_bus"):
        if not np.isfinite(np.asarray(i[key], dtype=np.float64)).all():
            note(f"{key} non-finite", j)
    for cn, cam in i["cams"].items():
        if not np.isfinite(np.asarray(cam["cam_intrinsic"], dtype=np.float64)).all():
            note("cam_intrinsic non-finite", j); break
    if not lanes.get(i["scene_token"], []):
        note("시나리오 map_lanes 없음", j)
    if int(i["frame_idx"]) < (QUEUE - 1) * SI:
        note(f"history 부족 (frame<{(QUEUE-1)*SI})", j, 3)

print(f"\n시나리오 {len(byscen)}  프레임/시나리오 "
      f"{min(len(v) for v in byscen.values())}~{max(len(v) for v in byscen.values())}")
nc = sum(1 for v in byscen.values()
         if sorted(v) != list(range(min(v), min(v) + len(v))))
print(f"frame_idx 비연속 시나리오 {nc}")
print(f"\n{'항목':<34}{'건수':>9}{'비율':>9}   예시 idx")
for k, n in c.most_common():
    print(f"{k:<34}{n:>9,}{100*n/len(infos):>8.2f}%   {bad[k]}")
fatal = [k for k in c if "non-finite" in k or "불일치" in k or "shape" in k
         or "one-hot" in k or "!= 6" in k]
print(f"\n판정: {'PASS -- 학습을 죽일 항목 없음' if not fatal else 'FAIL ' + str(fatal)}")
print("  (range 안 박스 0개 / gt_names 빔 / history 부족 은 가드 완료 후 정상 통과)")
