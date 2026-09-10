import pathlib

p = pathlib.Path("experiments/pv_selector_20260910/pv_selector.py")
s = p.read_text()
old = """    def __init__(self, *, mode="real"):
        super().__init__()
        if mode not in ("real", "zero"):
            raise ValueError("mode must be real or zero")
        self.mode = mode
        self.token_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(96, 128), nn.ReLU(),"""
new = """    def __init__(self, *, mode="real", lead_dim=0):
        super().__init__()
        if mode not in ("real", "zero"):
            raise ValueError("mode must be real or zero")
        self.mode = mode
        # Per-row scene context (lead vehicle) broadcast over candidates.
        self.lead_dim = int(lead_dim)
        self.token_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(96 + self.lead_dim, 128), nn.ReLU(),"""
assert old in s
s = s.replace(old, new)

old = "    def forward(self, output, goal_xy=None, status=None):"
new = "    def forward(self, output, goal_xy=None, status=None, lead=None):"
assert old in s
s = s.replace(old, new)

old = """            scene = self.token_projection(safe_tokens)
            residual = self.score_head(torch.cat((features32, scene), -1)).squeeze(-1)"""
new = """            scene = self.token_projection(safe_tokens)
            parts = [features32, scene]
            if self.lead_dim:
                if lead is None or lead.shape != (features32.shape[0], self.lead_dim):
                    raise ValueError("lead must be [B,lead_dim] when lead_dim is set")
                parts.append(lead.float()[:, None].expand(-1, features32.shape[1], -1))
            elif lead is not None:
                raise ValueError("lead given but the head was built without lead_dim")
            residual = self.score_head(torch.cat(parts, -1)).squeeze(-1)"""
assert old in s
s = s.replace(old, new)
p.write_text(s)

t = pathlib.Path("experiments/pv_selector_20260910/train_pv_selector.py")
u = t.read_text()

old = """        self.drop_goal = False"""
new = old + """
        self.lead = None"""
assert old in u
u = u.replace(old, new)

old = """    def status_of(self, index):
        return None if self.status8 is None else self.status8[index]"""
new = old + """

    def attach_lead(self, directory, split):
        \"\"\"Bind ground-truth lead-vehicle context, aligned to the cache rows.\"\"\"
        bundle = np.load(Path(directory) / ('lead_%s.npz' % split), allow_pickle=False)
        if not np.array_equal(bundle['rows'], self.rows_cpu()):
            raise ValueError('Lead-feature rows differ from the cache rows')
        features = bundle['features'].astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite lead features')
        self.lead = torch.from_numpy(features).to(self.device)
        return {'path': str(Path(directory).resolve()), 'names': [str(x) for x in bundle['names']],
                'dim': int(features.shape[1]), 'present_fraction': float(features[:, 0].mean()),
                'source': 'annotation ground truth, oracle probe'}

    def lead_of(self, index):
        return None if self.lead is None else self.lead[index]"""
assert old in u
u = u.replace(old, new)

u = u.replace("        out = head(inp, goal_xy=goal, status=cache.status_of(index))",
              "        out = head(inp, goal_xy=goal, status=cache.status_of(index), lead=cache.lead_of(index))")
u = u.replace("            out = head(inp, goal_xy=goal, status=train.status_of(index))",
              "            out = head(inp, goal_xy=goal, status=train.status_of(index), lead=train.lead_of(index))")

old = "    p.add_argument('--status-vx-mae', type=float, default=0.0)"
new = (old + "\n"
       "    # Oracle probe: ground-truth lead-vehicle context as extra row features.\n"
       "    p.add_argument('--lead-cache', default=None)")
assert old in u
u = u.replace(old, new)

old = "        head = SceneResidualSelector(mode=a.mode).cuda()"
new = """        lead_receipt = None
        if a.lead_cache:
            lead_receipt = train.attach_lead(a.lead_cache, 'train')
            tune.attach_lead(a.lead_cache, 'tune')
        head = SceneResidualSelector(mode=a.mode,
                                     lead_dim=lead_receipt['dim'] if lead_receipt else 0).cuda()"""
assert old in u
u = u.replace(old, new)

old = "                        train_status_provenance=train.status_provenance,"
new = "                        lead_features=lead_receipt,\n" + old
assert old in u
u = u.replace(old, new)
t.write_text(u)
print("patched")
