import pathlib
p = pathlib.Path("experiments/pv_selector_20260910/cache_c_candidates_ablation.py")
s = p.read_text()

old = "    require(plan.receipt['checkpoint_sha256'] == C_CHECKPOINT_SHA, 'Frozen C checkpoint identity mismatch')"
new = ("    # Arm B carries no status anywhere, so the C identity pin is widened to the\n"
       "    # two frozen temporal terminals and the arm is recorded in the manifest.\n"
       "    require(plan.receipt['checkpoint_sha256'] in (C_CHECKPOINT_SHA, B_CHECKPOINT_SHA),\n"
       "            'Frozen temporal checkpoint identity mismatch')")
assert old in s
s = s.replace(old, new)

s = s.replace("C_CHECKPOINT_SHA = 'b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'",
              "C_CHECKPOINT_SHA = 'b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'\n"
              "B_CHECKPOINT_SHA = None  # filled from the B terminal at build time")

import subprocess, hashlib
b = "/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/work_dirs/sparsedrivev2_status_20260910/temporal_b_history_s0_v1/last.pth"
d = hashlib.sha256()
with open(b, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""):
        d.update(c)
s = s.replace("B_CHECKPOINT_SHA = None  # filled from the B terminal at build time",
              "B_CHECKPOINT_SHA = '%s'" % d.hexdigest())

lines = s.split("\n")
for i, line in enumerate(lines):
    if line.startswith("    require(plan.manifest['arguments']['common_status'] is True"):
        j = i
        while not lines[j].rstrip().endswith(")"):
            j += 1
        block = "\n".join(lines[i:j + 1])
        break
else:
    raise SystemExit("common_status guard not found")
s = s.replace(block,
              "    arm_common_status = plan.manifest['arguments']['common_status']\n"
              "    require(isinstance(arm_common_status, bool)\n"
              "            and plan.manifest['arguments']['history_mode'] == 'real',\n"
              "            'Expected a real-history temporal arm')")
s = s.replace("'common_status_route': 'past-only causal4 to common perception only',",
              "'common_status_route': ('past-only causal4 to common perception only'\n"
              "                                        if arm_common_status else 'none anywhere in the network'),")
p.write_text(s)
print("patched, B sha", d.hexdigest()[:16])
