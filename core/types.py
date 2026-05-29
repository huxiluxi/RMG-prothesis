import numpy as np
from dataclasses import dataclass

@dataclass
class Sweep:
    freq: np.ndarray
    fwd: np.ndarray
    rev0: np.ndarray
    rev1: np.ndarray


@dataclass
class SParameters:
    freq: np.ndarray
    s11: np.ndarray
    s21: np.ndarray

@dataclass
class CalData:
    freq: float = 0

    short: complex = 0j
    open: complex = 0j
    load: complex = 0j

    through: complex = 0j
    thrurefl: complex = 0j
    isolation: complex = 0j

    # error terms
    e00: complex = 0j
    e11: complex = 0j
    delta_e: complex = 0j
    e10e01: complex = 0j

    e30: complex = 0j
    e22: complex = 0j
    e10e32: complex = 0j

@dataclass
class FeatureVector:
    x: np.ndarray

@dataclass
class Prediction:
    label: int
    confidence: float
    timestamp: float

@dataclass
class Action:
    name: str

