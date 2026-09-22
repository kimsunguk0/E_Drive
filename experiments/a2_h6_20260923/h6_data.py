"""Six-timepoint motion observation. Scene branch and supervision stay at four.

The matching graph won -0.033 by sampling the same spatial search range four
times more finely. H6 asks the same question on the time axis: keep the existing
reach and add timepoints inside it (NEAR) or extend it (LONG).

Only the MOTION branch sees six frames. The scene branch keeps its four aligned
history images, and history/state supervision keeps its four targets, because
the supervision arrays are (N,4,...) on disk and regenerating them is a
different change. So the head still predicts four offsets -- the ones that have
labels -- while the correlation sees six.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/md_r0_reset_20260914'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matching_resolution as mr

# frame offsets at 10 Hz; the first four are the existing supervised contract
BASE_OFFSETS = (1, 2, 5, 10)
H6 = {
    'H6-NEAR': dict(offsets=(1, 2, 3, 4, 5, 10),
                    seconds=(.1, .2, .3, .4, .5, 1.),
                    extra=(3, 4),
                    question='denser sampling inside the existing reach'),
    'H6-LONG': dict(offsets=(1, 2, 5, 10, 20, 25),
                    seconds=(.1, .2, .5, 1., 2., 2.5),
                    extra=(20, 25),
                    question='extend the reach to -2.0 and -2.5 s'),
}
SUPERVISED = len(BASE_OFFSETS)


class H6CanvasDataset(torch.utils.data.Dataset):
    """MotionCanvasDataset plus the extra motion canvases, in offset order."""

    def __init__(self, canvas, arm):
        spec = H6[arm]
        # The caller already built the four-frame canvas dataset, so it is wrapped
        # rather than rebuilt; rebuilding would re-derive the jitter stream.
        self.inner = canvas
        self.offsets = spec['offsets']
        self.order = [self.offsets.index(o) for o in BASE_OFFSETS]
        self.extra = [o for o in self.offsets if o not in BASE_OFFSETS]
        self.arm = arm

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        if name in {'inner', 'offsets', 'order', 'extra', 'arm'}:
            raise AttributeError(name)
        return getattr(self.inner, name)

    def set_epoch(self, epoch):
        return self.inner.set_epoch(epoch)

    def __getitem__(self, index):
        item = self.inner[index]
        base = self.inner.base
        row = int(base.rows[index])
        scene = str(base.scene_names[row])
        frame = int(base.arr['frame'][row])
        jitter = self.inner._jitter_for(row)
        canvases = {o: item[mr.MOTION_HISTORY_KEY][i]
                    for i, o in enumerate(BASE_OFFSETS)}
        for o in self.extra:
            canvases[o] = self.inner._canvas(scene, 'camera_front', frame - int(o), jitter)
        # ordered by offset so index k always means self.offsets[k]
        item[mr.MOTION_HISTORY_KEY] = torch.stack([canvases[o] for o in self.offsets])
        item['motion_time_offsets'] = torch.tensor(H6[self.arm]['seconds'], dtype=torch.float32)
        return item


def supervised_indices(arm):
    """Positions of the supervised offsets inside this arm's offset list."""
    return [H6[arm]['offsets'].index(o) for o in BASE_OFFSETS]


def patch_motion_encoder(encoder_class, seconds, keep):
    """Give the motion encoder its own timepoints, and keep the readout at four.

    Only this encoder needs the extra times; the scene encoder keeps the batch's
    four-offset time_offsets untouched. The readout is sliced back to the four
    offsets that have targets on disk, so the loss still matches (N,4,4).
    """
    original = encoder_class.forward
    times = torch.tensor(seconds, dtype=torch.float32)
    keep = list(keep)

    def forward(self, current_front_levels, history_front_levels, time_offsets):
        frames = history_front_levels[0].shape[1]
        if frames == len(times):
            time_offsets = times.to(time_offsets.device, time_offsets.dtype)
            time_offsets = time_offsets[None].expand(history_front_levels[0].shape[0], -1)
        out = original(self, current_front_levels, history_front_levels, time_offsets)
        # Select the positions whose offsets have targets on disk. Taking the
        # first four would silently pair offset 3 with the label for offset 5
        # under H6-NEAR, which the history MAE per offset exposes immediately.
        for key in ('history_hat', 'history_logvar'):
            if key in out and out[key].shape[1] > len(keep):
                out[key] = out[key][:, keep]
        return out

    encoder_class.forward = forward
    return original


def flip_extra(original):
    """The extra canvases ride in the same tensor, so the existing wrapper covers
    them; this only asserts that, rather than adding a second mirror."""
    return original
