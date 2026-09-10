import pathlib
p = pathlib.Path("experiments/pv_selector_20260910/pv_status.py")
s = p.read_text()
s += '''

# Measured P7 residual autocorrelation at the 0.5 s row spacing, per channel.
P7_RESIDUAL_RHO = (0.9394, 0.6489, 0.8614, 0.6874)
# Measured P7 b0 residual MAE against the provided causal value, per channel.
P7_RESIDUAL_MAE = (0.983984, 0.036834, 0.280207, 0.084016)


def add_correlated_noise(status8, sessions, frames, vx_mae, seed=0):
    """Return status8 with an AR(1) residual whose vx MAE is ``vx_mae``.

    This is an explicit error MODEL, not a real estimator: an AR(1) process per
    session in frame order, with the autocorrelation P7's residuals actually
    show, and per-channel magnitudes scaled from P7's measured MAE so that the
    vx channel hits the requested level.  It answers how accurate an image
    estimator would have to be, not what any particular estimator would do.
    """
    status8 = np.array(status8, dtype=np.float32, copy=True)
    if vx_mae <= 0:
        return status8, {"vx_mae_target": 0.0, "model": "none"}
    ratio = vx_mae / P7_RESIDUAL_MAE[0]
    # Gaussian MAE = sigma * sqrt(2/pi)
    sigma = np.array(P7_RESIDUAL_MAE, dtype=np.float64) * ratio * np.sqrt(np.pi / 2.0)
    rng = np.random.default_rng(seed)
    noise = np.zeros((len(status8), 4), dtype=np.float64)
    order = np.lexsort((frames, sessions))
    for session in np.unique(sessions):
        index = order[sessions[order] == session]
        for channel in range(4):
            rho = P7_RESIDUAL_RHO[channel]
            innovation = rng.normal(0.0, sigma[channel] * np.sqrt(1.0 - rho * rho), len(index))
            series = np.empty(len(index))
            series[0] = rng.normal(0.0, sigma[channel])
            for i in range(1, len(index)):
                series[i] = rho * series[i - 1] + innovation[i]
            noise[index, channel] = series
    status8[:, 4:8] += noise.astype(np.float32)
    realized = float(np.abs(noise[:, 0]).mean())
    return status8, {"vx_mae_target": float(vx_mae), "vx_mae_realized": realized,
                     "model": "per-session AR(1) with P7 residual autocorrelation",
                     "rho": list(P7_RESIDUAL_RHO), "seed": seed}
'''
p.write_text(s)

t = pathlib.Path("experiments/pv_selector_20260910/train_pv_selector.py")
u = t.read_text()
u = u.replace("from pv_status import load_status8",
              "from pv_status import load_status8, add_correlated_noise")
old = """    def attach_status8(self, split):
        \"\"\"Bind the causal status overlay for final selection, aligned to rows.\"\"\"
        status8, provenance = load_status8(self.directory, split, self.rows_cpu())"""
new = """    def attach_status8(self, split, vx_mae=0.0, seed=0):
        \"\"\"Bind the causal status overlay for final selection, aligned to rows.\"\"\"
        status8, provenance = load_status8(self.directory, split, self.rows_cpu())
        if vx_mae > 0:
            frames = np.load(self.directory / 'frame.npy', allow_pickle=False)
            sessions = np.asarray(self.arrays['session_index'].cpu()
                                  if torch.is_tensor(self.arrays['session_index'])
                                  else self.arrays['session_index'])
            status8, noise = add_correlated_noise(status8, sessions, frames, vx_mae, seed)
            provenance = dict(provenance, degraded=noise)"""
assert old in u
u = u.replace(old, new)
old = """            train.attach_status8('train')
            tune.attach_status8('tune')"""
new = """            train.attach_status8('train', a.status_vx_mae, a.seed)
            tune.attach_status8('tune', a.status_vx_mae, a.seed + 1000)"""
assert old in u
u = u.replace(old, new)
old = "    p.add_argument('--goal', choices=('real', 'zero'), default='real')"
new = (old + "\n"
       "    # Degrade the provided status to the accuracy an image estimator would have.\n"
       "    p.add_argument('--status-vx-mae', type=float, default=0.0)")
assert old in u
u = u.replace(old, new)
t.write_text(u)
print("patched")
