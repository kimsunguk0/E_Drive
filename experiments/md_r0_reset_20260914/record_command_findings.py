#!/usr/bin/env python3
"""Record what the command fact-find actually established."""
import json
from pathlib import Path

p = Path("/NHNHOME/data/sukim/adcl/reports/md_r0_reset_20260914/command_inventory.json")
d = json.loads(p.read_text())
d["findings"] = {
    "train_and_test_do_not_carry_the_same_command_fields": {
        "test": {"file": "<clip>/command.parquet", "rows_per_clip": 1,
                 "columns": ["command", "vad_cmd"],
                 "meaning": "the command at the current frame, already reduced"},
        "train": {"file": "meta_train/<scene>/meta/command.parquet", "rows_per_scene": 400,
                  "columns": ["timestamp", "command"],
                  "meaning": "a per-frame series; there is no vad_cmd column"},
        "consequence": ("vad_cmd exists only on the test side. Using it as an input would mean "
                        "reproducing the organisers' reduction rule for training, which is not "
                        "verified anywhere. The six-class command string exists on both sides "
                        "and is the only field matchable without inventing that rule."),
        "join_required": "train commands must be matched to each anchor frame by timestamp",
    },
    "class_inventory_from_200_test_clips": {
        "command": {"LANE_KEEP": 177, "TURN_LEFT": 11, "LANE_CHANGE_R": 4,
                    "TURN_RIGHT": 3, "LANE_CHANGE_L": 3, "U_TURN": 2},
        "imbalance": "LANE_KEEP is 88.5 percent; four of the six classes are under 3 percent each",
        "missing_values": 0,
        "caveat": "200 of 1,125 clips; the tail classes are small enough that this is a rough count",
    },
    "flip_mapping_now_determinable": {
        "swap_under_mirroring": [["TURN_LEFT", "TURN_RIGHT"], ["LANE_CHANGE_L", "LANE_CHANGE_R"]],
        "map_to_themselves": ["LANE_KEEP", "U_TURN"],
        "vad_cmd_if_ever_used": "[1,0,0] and [0,1,0] swap; [0,0,1] is fixed",
        "unit_test_required": "a double flip must return the original class",
    },
    "practical_reading": ("a command branch would be trained on a signal that is LANE_KEEP nearly "
                         "nine times out of ten, so the contrast has to be read on the turning and "
                         "lane-change rows as well as on overall D3"),
}
p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
print("findings recorded")
