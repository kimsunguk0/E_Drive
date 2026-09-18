"""Additional historical scene observations with existing camera-specific jitter."""
import numpy as np
import torch
from torch.utils.data import Dataset
from scene_extensions import SIDE_CAMERA_IDS, SIDE_FRAME_OFFSETS
from motiondrive_v2_data import CAMERA_ORDER

SIDE_KEY = 'side_scene_images'
SIDE_FLIP = (2, 3, 0, 1)


class SideSceneDataset(Dataset):
    def __init__(self, base):
        self.base = base
        if tuple(base.history_offsets) != (1, 2, 5, 10):
            raise ValueError('SIDE pose-index mapping requires CONTROL history')

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_epoch(self, epoch):
        return self.base.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.base[index]
        row = int(self.rows[index])
        frame, scene = int(self.arr['frame'][row]), str(self.scene_names[row])
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + row)
        jitter = rng.uniform(.9, 1.1, (6, 3)) if self.augment else [None] * 6
        item[SIDE_KEY] = torch.stack([
            self.base._image(scene, CAMERA_ORDER[c], frame - offset, (384, 216), jitter[c])
            for c, offset in zip(SIDE_CAMERA_IDS, SIDE_FRAME_OFFSETS)])
        return item


def wrap_side_flip(original):
    def flip(item, width_full, width_hist):
        output = original(item, width_full, width_hist)
        if SIDE_KEY in item:
            output[SIDE_KEY] = item[SIDE_KEY][list(SIDE_FLIP)].flip(-1)
        return output
    return flip
