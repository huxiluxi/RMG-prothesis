from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CommitmentFilter:
    def __init__(self, window=5, verbose=False):
        self.verbose = verbose
        self.window = deque(maxlen=window)

        if self.verbose:
            logger.info(f"Initialized CommitmentFilter with window size: {window}")

    def update(self, pred):
        self.window.append(pred)

        values, counts = np.unique(self.window, return_counts=True)

        max_count_idx = np.argmax(counts)
        max_count = counts[max_count_idx]
        majority_value = values[max_count_idx]

        # Only commit if ALL values in window agree
        if max_count >= len(self.window)-1:
            result = majority_value
        else:
            result = -1   # or 6 if you prefer

        if self.verbose:
            logger.info(
                f"CommitmentFilter update: pred={pred}, "
                f"committed={result}, window={list(self.window)}"
            )

        return result