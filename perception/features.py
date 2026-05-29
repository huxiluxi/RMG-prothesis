import logging
import numpy as np

from core.types import SParameters, FeatureVector

logger = logging.getLogger(__name__)


class MagnitudeFeatureExtractor:
    """
    Converts complex calibrated S-parameters
    into magnitude-only ML feature vectors.
    """

    def __init__(
        self,
        use_s11=True,
        use_s21=True,
        db=False,
        verbose=False
    ):
        self.use_s11 = use_s11
        self.use_s21 = use_s21
        self.db = db
        self.verbose = verbose
        
        if self.verbose:
            logger.debug(f"Initialized feature extractor: S11={use_s11}, S21={use_s21}, dB={db}")

    def _mag(self, x):
        mag = np.abs(x)

        if self.db:
            mag = 20 * np.log10(np.maximum(mag, 1e-12))

        return mag

    def extract(
        self,
        sparams: SParameters
    ) -> FeatureVector:

        features = []

        if self.use_s11:
            features.append(
                self._mag(sparams.s11)
            )

        if self.use_s21:
            features.append(
                self._mag(sparams.s21)
            )

        x = np.concatenate(features)
        
        if self.verbose:
            logger.debug(f"Extracted feature vector: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}")

        return FeatureVector(x=x)