"""Dependency-free fixed temporal contracts for MotionDrive V2.

Keep this module outside ``models.motiondrive_v2`` so input-contract tooling can
import it without importing Torch through the model package ``__init__``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalContract:
    name: str
    frame_offsets: tuple[int, int, int, int]
    nominal_seconds: tuple[float, float, float, float]


CONTROL = TemporalContract("control", (1, 2, 5, 10), (.1, .2, .5, 1.))
WIDE = TemporalContract("wide", (2, 5, 10, 20), (.2, .5, 1., 2.))
CONTRACTS = {contract.name: contract for contract in (CONTROL, WIDE)}


def temporal_contract(name: str = "control") -> TemporalContract:
    if name not in CONTRACTS:
        raise ValueError("history contract must be control or wide")
    return CONTRACTS[name]


def validate_temporal_contract(name, frame_offsets, nominal_seconds) -> TemporalContract:
    contract = temporal_contract(name)
    try:
        frames = tuple(int(value) for value in frame_offsets)
        seconds = tuple(float(value) for value in nominal_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid temporal contract values") from exc
    if frames != contract.frame_offsets or seconds != contract.nominal_seconds:
        raise ValueError(f"{name} temporal contract values do not match the fixed recipe")
    return contract
