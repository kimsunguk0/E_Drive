import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_motiondrive_v2_deploy_fixture import (POSE_COLUMNS, POSE_FRAMES, encode_pose,
                                                jpeg_plan, member_bytes, relative_pose_table,
                                                sample_plan)


def raw_pose():
    frames = np.arange(300)
    ts = 100000 + frames * 101
    stamps = pd.DataFrame({"timestamp": ts, "frame_id": frames})
    ego = pd.DataFrame({"timestamp": ts, **{key: frames * (i + .1) for i, key in enumerate(POSE_COLUMNS[1:])},
                        "forbidden_status": frames * 3})
    return stamps, ego


def test_exact_pose_schema_and_no_timestamp_or_future_intermediates():
    timestamps, ego = raw_pose()
    table = relative_pose_table(timestamps.sample(frac=1, random_state=0), ego.iloc[::-1], 180)
    assert tuple(table.column_names) == POSE_COLUMNS
    assert table["frame"].to_pylist() == list(POSE_FRAMES)
    assert not table.schema.metadata
    restored = pq.read_table(io.BytesIO(encode_pose(table)))
    assert restored.equals(table, check_metadata=True)
    assert restored["x"][0].as_py() == ego["x"][150]
    assert restored["x"][-1].as_py() == ego["x"][230]


@pytest.mark.parametrize("issue", ["missing_goal", "duplicate_timestamp", "nonfinite"])
def test_bad_pose_source_fails_closed(issue):
    stamps, ego = raw_pose()
    if issue == "missing_goal":
        stamps = stamps[stamps.frame_id != 230]
    elif issue == "duplicate_timestamp":
        ego.loc[1, "timestamp"] = ego.loc[0, "timestamp"]
    else:
        ego.loc[180, "x"] = np.nan
    with pytest.raises(ValueError):
        relative_pose_table(stamps, ego, 180)


def test_ten_jpegs_use_official_relative_names_only():
    files = jpeg_plan("20260112-105434", 30)
    assert len(files) == 10
    assert len({dst for _, dst in files}) == 10
    assert ("20260112-105434/camera_front/00000029.jpg", "camera_front/frame_-1.jpg") in files
    assert sum(dst.endswith("frame_0.jpg") for _, dst in files) == 6
    assert all("frame_" in dst for _, dst in files)


def test_tar_reads_only_regular_exact_members():
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as tar:
        member = tarfile.TarInfo("scene/calibration/calibration.parquet")
        member.size = 3
        tar.addfile(member, io.BytesIO(b"abc"))
        symlink = tarfile.TarInfo("unsafe")
        symlink.type, symlink.linkname = tarfile.SYMTYPE, "/outside"
        tar.addfile(symlink)
    memory.seek(0)
    with tarfile.open(fileobj=memory, mode="r:") as tar:
        assert member_bytes(tar, "scene/calibration/calibration.parquet") == b"abc"
        with pytest.raises(ValueError):
            member_bytes(tar, "unsafe")


def test_fixed_train_only_scene_selection():
    train = [f"20260112-{n:06d}" for n in range(5)]
    manifest = {"splits": {"train": train[::-1], "tune": ["tune"], "val": ["val"], "historical_val": ["val"]},
                "scene_to_session": {scene: scene for scene in train + ["tune", "val"]}}
    assert sample_plan(manifest) == [(scene, frame) for scene in train[:4] for frame in (30, 180)]
