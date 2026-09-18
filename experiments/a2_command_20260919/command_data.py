"""Read the supplied six-way command. Never derive it from pose or future XY."""
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
import torch
from command_common import CACHE, sha

LABELS = ('LANE_KEEP', 'TURN_LEFT', 'TURN_RIGHT', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'U_TURN')
MIRROR = (0, 2, 1, 4, 3, 5)
KEY = 'provided_command'

def encode_label(label):
    if not isinstance(label, str) or label not in LABELS:
        raise ValueError(f'Unknown provided command: {label!r}')
    return LABELS.index(label)

def commands_for_frames(timestamp_table, command_table, frames):
    """Exact timestamp join to the requested CURRENT frame, no interpolation."""
    frame_id = timestamp_table['frame_id']
    timestamps = timestamp_table['timestamp']
    command_ts = command_table['timestamp']
    labels = command_table['command']
    if (len(frame_id) != len(timestamps) or len(command_ts) != len(labels)
            or len(set(frame_id)) != len(frame_id)
            or len(set(timestamps)) != len(timestamps)
            or len(set(command_ts)) != len(command_ts)
            or not np.isfinite(timestamps).all() or not np.isfinite(command_ts).all()):
        raise ValueError('Nonunique, nonfinite or malformed frame/command timestamps')
    frame_to_time = dict(zip(frame_id, timestamps))
    time_to_label = dict(zip(command_ts, labels))
    try:
        return np.asarray([encode_label(time_to_label[frame_to_time[int(f)]]) for f in frames], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f'No exact provided command for requested frame: {exc}') from exc

def read_scene_commands(meta_dir, frames):
    ts = pq.read_table(meta_dir/'timestamps.parquet', columns=['frame_id', 'timestamp']).to_pydict()
    cm = pq.read_table(meta_dir/'command.parquet', columns=['timestamp', 'command']).to_pydict()
    return commands_for_frames(ts, cm, frames)

def read_clip_command(path):
    """Deployment parser: the official clip has one current command row."""
    values = pq.read_table(path, columns=['command']).column('command').to_pylist()
    if len(values) != 1:
        raise ValueError('Expected exactly one current command per test clip')
    index = encode_label(values[0])
    return np.eye(len(LABELS), dtype=np.float32)[index]

class CommandDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        manifest = json.loads((CACHE/'manifest.json').read_text())
        if manifest['producer_sha256'] != sha(__file__) or manifest['labels'] != list(LABELS):
            raise ValueError('Command input producer changed')
        artifact = CACHE/'provided_commands.npz'
        if sha(artifact) != manifest['artifact']['sha256']:
            raise ValueError('Command artifact SHA mismatch')
        with np.load(artifact, allow_pickle=False) as z:
            rows, command = z['row'], z['command']
        pos = np.searchsorted(rows, base.rows)
        if (pos >= len(rows)).any() or not np.array_equal(rows[pos], base.rows):
            raise ValueError('Dataset rows outside pinned train/tune command inputs')
        actual = hashlib.sha256(np.asarray(base.rows, dtype='<i8').tobytes()).hexdigest()
        if actual != manifest['splits'][base.split]['rows_sha256']:
            raise ValueError('Command dataset split row identity differs')
        self.command_ids = command[pos].copy()
        self.command_ids.setflags(write=False)
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
        command = torch.zeros(len(LABELS), dtype=torch.float32)
        command[int(self.command_ids[index])] = 1.
        item[KEY] = command
        return item

def wrap_command_flip(original):
    def flip(item, *args, **kwargs):
        result = original(item, *args, **kwargs)
        if KEY not in item or item[KEY].shape != (len(LABELS),):
            raise ValueError('Missing six-way command in mirrored sample')
        result[KEY] = item[KEY][list(MIRROR)].clone()
        return result
    return flip
