"""Derive the status-enabled selector from the frozen c_refine sources."""
import pathlib, shutil

SRC = pathlib.Path("experiments/c_refine_20260910")
DST = pathlib.Path("experiments/pv_selector_20260910")
DST.mkdir(parents=True, exist_ok=True)

# ---------------- pv_selector.py ----------------
s = (SRC / "c_scene_selector.py").read_text()

s = s.replace('FEATURE_VERSION = "candidate_relative_32_v1_zero_status"',
              'FEATURE_VERSION = "candidate_relative_32_v1_optional_causal_status"')
s = s.replace('SELECTOR_VERSION = "candidate_scene_residual_real_zero_v1"',
              'SELECTOR_VERSION = "candidate_scene_residual_real_zero_status_v1"')

old = '''def build_features32(output, goal_xy=None):
    """Exact old make_candidate_features(output, zeros[B,8], goal) in FP32.

    Features are candidate interval velocities (12), interval accelerations/3
    (10), four zero state slots, goal/50 and endpoint-minus-goal/50 (4), centered
    original score (1), and goal-present (1). Labels and states are not accepted.
    Invalid rows are zero before arithmetic and again in the final output.
    """'''
new = '''def build_features32(output, goal_xy=None, status=None):
    """Exact old make_candidate_features(output, status[B,8], goal) in FP32.

    Features are candidate interval velocities relative to causal vx/vy (12),
    interval accelerations/3 (10), the four causal state slots, goal/50 and
    endpoint-minus-goal/50 (4), centered original score (1), and goal-present
    (1). With ``status=None`` the four state slots stay zero and the velocities
    stay absolute, reproducing the earlier zero-status behaviour bit for bit.
    Only slots 4:8 (causal vx, vy, ax, ay) are read. Labels are never accepted.
    Invalid rows are zero before arithmetic and again in the final output.
    """'''
assert old in s
s = s.replace(old, new)

old = '''        # Keep the explicit subtraction in the original helper, including zeros.
        zeros = torch.zeros((b, 8), device=xy.device, dtype=torch.float32)
        relative_vel = (vel - zeros[:, None, None, 4:6]).reshape(b, k, 12)
        acceleration = (torch.diff(vel, dim=2) * (2. / 3.)).reshape(b, k, 10)
        state = torch.cat((zeros[:, 4:6] / 20., zeros[:, 6:8] / 3.), -1)'''
new = '''        # Keep the explicit subtraction in the original helper, including zeros.
        if status is None:
            status8 = torch.zeros((b, 8), device=xy.device, dtype=torch.float32)
        else:
            status8 = torch.as_tensor(status, device=xy.device, dtype=torch.float32)
            if status8.shape != (b, 8) or not bool(torch.isfinite(status8[:, 4:8]).all()):
                raise ValueError("status must be [B,8] with finite causal slots 4:8")
        relative_vel = (vel - status8[:, None, None, 4:6]).reshape(b, k, 12)
        acceleration = (torch.diff(vel, dim=2) * (2. / 3.)).reshape(b, k, 10)
        state = torch.cat((status8[:, 4:6] / 20., status8[:, 6:8] / 3.), -1)'''
assert old in s
s = s.replace(old, new)

assert '    def forward(self, output, goal_xy=None):' in s
s = s.replace('    def forward(self, output, goal_xy=None):',
              '    def forward(self, output, goal_xy=None, status=None):')
assert '        features32 = build_features32(output, goal_xy)' in s
s = s.replace('        features32 = build_features32(output, goal_xy)',
              '        features32 = build_features32(output, goal_xy, status)')
(DST / "pv_selector.py").write_text(s)

# ---------------- train_pv_selector.py ----------------
t = (SRC / "train_c_scene_selector.py").read_text()

assert 'from c_scene_selector import SceneResidualSelector' in t
t = t.replace('from c_scene_selector import SceneResidualSelector',
              'from pv_selector import SceneResidualSelector\nfrom pv_status import load_status8')

old = '''        if self.arrays['token'].shape != (self.n, self.k, 256):
            raise ValueError('Unexpected scene token shape')'''
new = old + '''
        self.status8 = None
        self.status_provenance = None'''
assert old in t
t = t.replace(old, new)

old = '''    def rows_cpu(self):
        a = self.arrays['rows']
        return a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)'''
new = old + '''

    def attach_status8(self, split):
        """Bind the causal status overlay for final selection, aligned to rows."""
        status8, provenance = load_status8(self.directory, split, self.rows_cpu())
        self.status8 = torch.from_numpy(status8).to(self.device)
        self.status_provenance = provenance

    def status_of(self, index):
        return None if self.status8 is None else self.status8[index]'''
assert old in t
t = t.replace(old, new)

old = '''        inp, goal = cache.inputs(index)
        out = head(inp, goal_xy=goal)'''
new = '''        inp, goal = cache.inputs(index)
        out = head(inp, goal_xy=goal, status=cache.status_of(index))'''
assert old in t
t = t.replace(old, new)

old = '''            inp, goal = train.inputs(index)
            optimizer.zero_grad(set_to_none=True)
            out = head(inp, goal_xy=goal)'''
new = '''            inp, goal = train.inputs(index)
            optimizer.zero_grad(set_to_none=True)
            out = head(inp, goal_xy=goal, status=train.status_of(index))'''
assert old in t
t = t.replace(old, new)

old = "    p.add_argument('--mode', choices=('real', 'zero'), required=True)"
new = old + '''
    # 'real' routes the provided causal status into the 32 selection features
    # only; 'zero' keeps the four state slots at zero as before.
    p.add_argument('--status', choices=('real', 'zero'), required=True)'''
assert old in t
t = t.replace(old, new)

old = '''        head = SceneResidualSelector(mode=a.mode).cuda()'''
new = '''        if a.status == 'real':
            train.attach_status8('train')
            tune.attach_status8('tune')
        head = SceneResidualSelector(mode=a.mode).cuda()'''
assert old in t
t = t.replace(old, new)

old = "        for name in ('train_c_scene_selector.py', 'c_scene_selector.py'):"
new = "        for name in ('train_pv_selector.py', 'pv_selector.py', 'pv_status.py'):"
assert old in t
t = t.replace(old, new)

old = "        manifest = dict(schema='c_scene_selector_frozen_v1', arguments=vars(a),"
new = "        manifest = dict(schema='pv_selector_status_v1', arguments=vars(a),"
assert old in t
t = t.replace(old, new)

old = "                        base_frozen=True, raw_status_input=False, goal='final completed-candidate selection only',"
new = ("                        base_frozen=True, raw_status_input=(a.status == 'real'),\n"
       "                        status_route=('provided causal vx,vy,ax,ay to final completed-candidate '\n"
       "                                      'selection only' if a.status == 'real' else 'constant zero'),\n"
       "                        train_status_provenance=train.status_provenance,\n"
       "                        tune_status_provenance=tune.status_provenance,\n"
       "                        goal='final completed-candidate selection only',")
assert old in t
t = t.replace(old, new)

(DST / "train_pv_selector.py").write_text(t)
print("wrote", DST / "pv_selector.py", DST / "train_pv_selector.py")
