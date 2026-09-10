"""Fixed, label-independent interventions for NAVSIM-style status8.

Contract
--------
Inputs are finite floating arrays [N, 8], with zero command slots 0:4 and
[vx, vy, ax, ay] in slots 4:8 (m/s, m/s, m/s^2, m/s^2). Only vx/ax are
perturbed, except the explicit ``zero_status`` intervention. No clipping,
sample centering, fitted normalization, labels, or model outputs are used.

``nominal_time_s`` must use a FIXED per-session origin and a nominal 10 Hz
grid. Do not subtract the first time in a requested subset/batch. The origin
can be obtained from scene/session metadata before selecting evaluation rows.
The same seed/session/time/channel yields the same error under row reordering,
subsetting, duplicated rows, and different batch boundaries.

Correlated error is sigma * (sqrt(rho)*U_session + sqrt(1-rho)*Z_session(t)),
where U is standard Gaussian and Z is a stationary Gaussian OU process sampled
at 10 Hz. We fix rho=.5 and tau=1 second. Thus marginal E[error]=0 and
E[abs(error)]=target_MAE, using sigma=target_MAE*sqrt(pi/2); correlation at lag
dt is rho+(1-rho)*exp(-abs(dt)/tau). This is a POPULATION MAE contract, not an
exact sample MAE. Retaining realized session bias is deliberate. Vx/ax use
independent streams; all condition strengths reuse the same standardized
draws (common random numbers). These synthetic errors do not claim to reproduce
an image estimator's state-dependent failures or physical cross-axis coupling.

The stationary process starts at tick zero with a N(0,1) value; later values
follow the exact OU transition. Stable SHA256-derived PCG64 seeds avoid Python
hash randomization. Random generation precedes row selection, so requested rows
cannot influence earlier values. Runtime receipts should record the NumPy
version and module hash, as they do for other frozen computation sources.

API: build_conditions(); perturb_status(status8, sessions, nominal_time_s,
condition, seed=20260910). ``status_errors`` exposes the additive synthetic
error before float32 status addition. It intentionally rejects zero_status,
which removes the observed signal rather than adding an independent error.
CLI: python status_perturbations.py --list (prints the protocol; no inference).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np

PROTOCOL_VERSION = "status8_session_ou_v1"
DEFAULT_SEED = 20260910
DT_SECONDS = 0.1
TAU_SECONDS = 1.0
SESSION_VARIANCE_FRACTION = 0.5
MAX_TIME_SECONDS = 36000.0  # Reject accidental Unix timestamps / wrong origins.
STATUS_FIELDS = ("command0", "command1", "command2", "command3", "vx", "vy", "ax", "ay")


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    vx_bias: float = 0.0
    ax_bias: float = 0.0
    vx_mae: float = 0.0
    ax_mae: float = 0.0

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("Condition name must be a plain lowercase identifier")
        if self.kind not in ("baseline", "zero_status", "bias", "session_ou"):
            raise ValueError("Unknown status intervention kind")
        values = (self.vx_bias, self.ax_bias, self.vx_mae, self.ax_mae)
        if not all(math.isfinite(x) for x in values) or min(self.vx_mae, self.ax_mae) < 0:
            raise ValueError("Finite biases and nonnegative finite population MAEs required")
        if self.kind in ("baseline", "zero_status") and any(values):
            raise ValueError("Control interventions cannot also contain error parameters")
        if self.kind == "bias" and (self.vx_mae or self.ax_mae):
            raise ValueError("Bias intervention cannot contain random-error MAEs")
        if self.kind == "session_ou" and (self.vx_bias or self.ax_bias):
            raise ValueError("Correlated error must be mean-zero in expectation")

    def to_dict(self):
        return asdict(self)


def _number(value):
    return format(value, ".12g").replace(".", "p")


def build_conditions():
    """Return 24 predeclared conditions: two controls, 16 biases, six OU pairs."""
    result = [Condition("baseline", "baseline"), Condition("zero_status", "zero_status")]
    for field, levels in (("vx", (.05, .1, .2, .5, 1.0)), ("ax", (.1, .3, .6))):
        for level in levels:
            for sign, marker in ((-1, "m"), (1, "p")):
                result.append(Condition(f"bias_{field}_{marker}{_number(level)}", "bias",
                                        **{field + "_bias": sign * level}))
    for vx in (.1, .3, 1.0):
        for ax in (.1, .3):
            result.append(Condition(f"ou_vx{_number(vx)}_ax{_number(ax)}", "session_ou",
                                    vx_mae=vx, ax_mae=ax))
    return tuple(result)


def protocol():
    return dict(version=PROTOCOL_VERSION, fields=list(STATUS_FIELDS), seed=DEFAULT_SEED,
                dt_seconds=DT_SECONDS, tau_seconds=TAU_SECONDS,
                session_variance_fraction=SESSION_VARIANCE_FRACTION,
                max_relative_time_seconds=MAX_TIME_SECONDS,
                target_mae_definition="Gaussian population E[abs(error)]; no sample normalization",
                session_time_origin="fixed metadata origin; never recomputed per subset",
                independent_noise_channels=["vx", "ax"],
                numpy_version=np.__version__, conditions=[x.to_dict() for x in build_conditions()])


def _metadata(sessions, nominal_time_s):
    raw_sessions = np.asarray(sessions)
    times = np.asarray(nominal_time_s)
    if raw_sessions.ndim != 1 or times.ndim != 1 or len(raw_sessions) != len(times):
        raise ValueError("Sessions and nominal times must be equally sized vectors")
    if any(not isinstance(x, (str, np.str_)) or not str(x) for x in raw_sessions):
        raise ValueError("Each session must be a nonempty stable string identifier")
    if times.dtype.kind not in "fiu" or not np.isfinite(times).all():
        raise ValueError("Nominal times must be finite real seconds")
    times = times.astype(np.float64)
    if ((times < 0) | (times > MAX_TIME_SECONDS)).any():
        raise ValueError("Use relative per-session time in [0, 36000] seconds, not Unix time")
    ticks = np.rint(times / DT_SECONDS).astype(np.int64)
    if (np.abs(times - ticks * DT_SECONDS) > 1e-6).any():
        raise ValueError("Nominal times must lie on the 0.1 second grid")
    return raw_sessions.astype(str), ticks


def _rng(seed, session, channel, component):
    key = json.dumps([PROTOCOL_VERSION, int(seed), session, channel, component],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    number = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(number))


def _standard_process(session, ticks, channel, seed):
    intercept = _rng(seed, session, channel, "session_intercept").standard_normal()
    white = _rng(seed, session, channel, "stationary_ou").standard_normal(int(ticks.max()) + 1)
    alpha = math.exp(-DT_SECONDS / TAU_SECONDS)
    innovation_scale = math.sqrt(1 - alpha * alpha)
    # Drawing a longer white-noise prefix does not change any earlier samples.
    values = np.empty_like(white)
    values[0] = white[0]
    for tick in range(1, len(white)):
        values[tick] = alpha * values[tick - 1] + innovation_scale * white[tick]
    return (math.sqrt(SESSION_VARIANCE_FRACTION) * intercept
            + math.sqrt(1 - SESSION_VARIANCE_FRACTION) * values[ticks])


def status_errors(sessions, nominal_time_s, condition, seed=DEFAULT_SEED):
    """Return label-independent float64 additive errors [N,8]."""
    if not isinstance(condition, Condition):
        raise TypeError("condition must be a validated Condition")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if condition.kind == "zero_status":
        raise ValueError("zero_status is a signal-removal control, not an independent additive error")
    names, ticks = _metadata(sessions, nominal_time_s)
    errors = np.zeros((len(names), 8), dtype=np.float64)
    if condition.kind == "bias":
        errors[:, 4], errors[:, 6] = condition.vx_bias, condition.ax_bias
    elif condition.kind == "session_ou":
        for session in np.unique(names):
            selected = names == session
            for channel, slot, mae in (("vx", 4, condition.vx_mae), ("ax", 6, condition.ax_mae)):
                if mae:
                    standard = _standard_process(str(session), ticks[selected], channel, int(seed))
                    errors[selected, slot] = mae * math.sqrt(math.pi / 2) * standard
    if not np.isfinite(errors).all():
        raise ValueError("Synthetic status error is nonfinite")
    return errors


def perturb_status(status8, sessions, nominal_time_s, condition, seed=DEFAULT_SEED):
    """Return a new finite float32 status8 array; never mutate input arrays."""
    values = np.asarray(status8)
    if values.ndim != 2 or values.shape[1] != 8 or values.dtype.kind != "f":
        raise ValueError("status8 must be a real floating [N,8] array")
    if not np.isfinite(values).all() or (values[:, :4] != 0).any():
        raise ValueError("All eight status fields must be finite and command slots must be zero")
    names, ticks = _metadata(sessions, nominal_time_s)
    if len(values) != len(names):
        raise ValueError("Status and metadata row counts differ")
    if not isinstance(condition, Condition):
        raise TypeError("condition must be a validated Condition")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if condition.kind == "zero_status":
        result = np.zeros(values.shape, dtype=np.float32)
    else:
        errors = status_errors(names, ticks * DT_SECONDS, condition, seed)
        with np.errstate(over="ignore", invalid="ignore"):
            result = values.astype(np.float32, copy=True)
            for slot in (4, 6):
                active = errors[:, slot] != 0
                result[active, slot] = (values[active, slot].astype(np.float64)
                                       + errors[active, slot]).astype(np.float32)
    if not np.isfinite(result).all() or (result[:, :4] != 0).any():
        raise ValueError("Perturbed status8 is nonfinite or changed command slots")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--list", action="store_true", help="Print the fixed intervention protocol")
    parser.parse_args()
    print(json.dumps(protocol(), indent=2, allow_nan=False))
