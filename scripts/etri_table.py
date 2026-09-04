"""모든 모델 비교 표의 단일 진입점 (4탄 규칙 4: 동일 val · 동일 metric 코드).

열 구성 (부록3 항목 6으로 3–8 m/s 추가)
    가중 / 무가중 / right(vad_cmd) / 이동(|v|>=0.5) / 3–8 m/s

최상단 고정 행은 **라벨 모호성 프록시**다 (부록3 항목 1·2).
    * '노이즈 바닥'이라 부르지 않는다. ego_pose와 hd_ego_pose의 불일치이고,
      채점 원천인 ego_pose 단독 오차는 미지다. 상한성 프록시로만 읽는다.
    * 모델과 동일 조건(같은 val 앵커·같은 test 정합 가중·같은 열 분해)으로 계산한다.
      조건이 다르면 "우리가 프록시에 얼마나 가까운가"를 말할 수 없다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from challenge_metrics import l2_challenge, l2_from_per_step  # noqa: E402

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
PROXY = "/tmp/pm97/data/etri/label_proxy_val.npz"

STOP = 0.5
SPEED_LO, SPEED_HI = 3.0, 8.0
SPEED_HI2 = 15.0          # 부록6 항목1: 15+ m/s 열 추가
VCMD = ["right", "left", "straight"]
COLUMNS = ["가중", "무가중", "right", "이동", "3–8m/s", "15+m/s"]


def wl2(pred, gt, w=None):
    """챌린지 L2. w가 있으면 표본 가중을 시점별 평균에 반영."""
    if pred.shape[0] == 0:
        return None
    if w is None:
        return float(l2_challenge(pred, gt)["L2_avg"])
    if w.sum() <= 0:
        return None
    d = np.sqrt(((pred[..., :2] - gt[..., :2]) ** 2).sum(-1))
    return float(l2_from_per_step((d * (w / w.sum())[:, None]).sum(0))["L2_avg"])


class ValSet:
    """val 앵커 한 벌. 모든 표가 이 객체 하나를 공유한다."""

    def __init__(self):
        d = np.load(CACHE, allow_pickle=True)
        sp = np.load(SPLIT, allow_pickle=True)
        self.d, self.sp = d, sp
        vi = sp["val_idx"]
        self.vi = vi
        self.w = sp["val_weight"].astype(np.float64)
        self.gt = d["fut"][vi].astype(np.float64)
        self.vel = d["vel"][vi].astype(np.float64)
        self.acc = d["acc"][vi].astype(np.float64)
        self.yawrate = d["yawrate"][vi].astype(np.float64)
        self.goal = d["goal"][vi].astype(np.float64)
        self.his = d["his"][vi].astype(np.float64)
        self.speed = d["speed"][vi].astype(np.float64)
        self.vcmd = d["vad_cmd"][vi].astype(int)
        self.meta = d["meta"][vi].astype(int)
        self.scen = np.array([str(d["scenarios"][k]) for k in d["scen_idx"][vi]])
        self.frame = d["frame"][vi].astype(int)
        self.labels = [str(x) for x in d["meta_labels"]]
        self.moving = self.speed >= STOP
        self.mid = (self.speed >= SPEED_LO) & (self.speed < SPEED_HI)
        self.fast = self.speed >= SPEED_HI2

    def masks(self):
        return {"가중": None,
                "무가중": np.ones(len(self.vi), bool),
                "right": self.vcmd == 0,
                "이동": self.moving,
                "3–8m/s": self.mid,
                "15+m/s": self.fast}

    def row(self, pred):
        """열 5개. '가중'만 가중 평균, 나머지는 무가중 부분집합."""
        out = {}
        for name, m in self.masks().items():
            if name == "가중":
                out[name] = wl2(pred, self.gt, self.w)
            else:
                out[name] = wl2(pred[m], self.gt[m], None)
            out["n_" + name] = int(len(self.vi) if m is None else m.sum())
        return out

    def label_proxy_row(self):
        """최상단 고정 행: hd_ego_pose로 만든 대체 GT와 ego_pose GT의 불일치.

        pred 자리에 '대체 라벨'을 넣어 같은 표 코드로 통과시킨다 -- 모델과 완전히
        동일한 경로를 타야 비교가 성립한다.
        """
        if not os.path.exists(PROXY):
            raise FileNotFoundError(
                f"{PROXY} 없음. scripts/etri_label_proxy.py 를 먼저 실행")
        alt = np.load(PROXY)["fut_hd"].astype(np.float64)
        assert alt.shape == self.gt.shape, (alt.shape, self.gt.shape)
        r = self.row(alt)
        r["_name"] = "라벨 모호성 프록시 (hd_ego_pose 대체GT vs ego_pose GT)"
        return r


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def residual(row, proxy):
    """프록시 대비 잔여 = 모델 L2 - 프록시 L2, 부분집합별로.

    4탄 부록4 항목 6: 게이트 판정은 절대값이 아니라 이 잔여 기준이다. 부분집합마다
    라벨 모호성 규모가 다르므로(right 0.0423 vs 이동 0.0346) 절대값끼리 비교하면
    어려운 부분집합에서 개선을 과소평가하게 된다.
    """
    return {c: (None if row[c] is None or proxy[c] is None else row[c] - proxy[c])
            for c in COLUMNS}


def render(rows, title="", note="", with_residual=True):
    """rows: [(이름, row_dict)]. **첫 행은 라벨 프록시여야 한다** (잔여의 기준)."""
    L = []
    if title:
        L.append(f"### {title}\n\n")
    L.append("| | " + " | ".join(COLUMNS) + " |\n")
    L.append("|" + "---|" * (len(COLUMNS) + 1) + "\n")
    for name, r in rows:
        L.append(f"| {name} | " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |\n")
    r0 = rows[0][1]
    L.append("\n표본 — " + ", ".join(f"{c} {r0['n_'+c]:,}" for c in COLUMNS) + "\n")

    if with_residual and len(rows) > 1:
        L.append("\n#### 프록시 대비 잔여 (모델 L2 − 프록시 L2)\n\n")
        L.append("| | " + " | ".join(COLUMNS) + " |\n")
        L.append("|" + "---|" * (len(COLUMNS) + 1) + "\n")
        for name, r in rows[1:]:
            res = residual(r, r0)
            L.append(f"| {name} | " + " | ".join(fmt(res[c]) for c in COLUMNS)
                     + " |\n")
        L.append("\n이후 게이트 판정은 이 잔여 기준이다 (부록4 항목 6).\n")
    if note:
        L.append("\n" + note + "\n")
    return "".join(L)
