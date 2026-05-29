import logging
import numpy as np
from core.types import Sweep, SParameters

logger = logging.getLogger(__name__)

class SParameterConverter:
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            logger.info("Initialized SParameterConverter")
    
    @staticmethod
    def from_sweep(sweep: Sweep) -> SParameters:
        eps = 1e-12
        s11 = sweep.rev0 / (sweep.fwd + eps)
        s21 = sweep.rev1 / (sweep.fwd + eps)

        return SParameters(
            freq=sweep.freq,
            s11=s11,
            s21=s21
        )