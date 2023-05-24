from active_learning.base_policy import (
    Probs,
    Indices,
    Inputs,
    Labels,
)
from active_learning.stopping_strategy import StoppingStrategy
from scipy.stats import binom
import numpy as np


class QBCB(StoppingStrategy):
    """
    Implements QBCB from David D. Lewis, Eugene Yang, and Ophir Frieder. 2021.
    Certifying One-Phase Technology-Assisted Reviews. In Proceedings of the 30th ACM
    International Conference on Information & Knowledge Management (CIKM '21).
    Association for Computing Machinery, New York, NY, USA, 893–902.
    https://doi.org/10.1145/3459637.3482415
    """

    def __init__(self, *args, positive_sample_size=50, confidence=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_sample_size = positive_sample_size
        self.confidence = confidence
        self.pre_positives = None
        self.pre_sample = None

    def pre_annotation(self, y: Labels) -> int:
        # Generate the random sample `n` from which we take the positive
        # sample size `r` (paper notation).
        # This must be called before using the class as a stopping strategy.
        perm = np.random.default_rng(42).permutation(np.vstack([np.arange(len(y)), y])).T
        if self.positive_sample_size >= y.sum():
            idx = len(y)
        else:
            idx = np.where(perm[:, 0].cumsum() == self.positive_sample_size)[0][0] + 1
        self.pre_sample = perm[:, 1][:idx]
        self.pre_positives = perm[:, 1][np.where(perm[:, 0][:idx] == 1)]

    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        if self.pre_positives is None:
            self.pre_annotation(y)
            if len(self.pre_sample) == len(y):
                return True
        if i < self.min_rounds:
            return False
        coeffs = (
            binom.cdf(
                np.arange(self.positive_sample_size + 1),
                self.positive_sample_size,
                self.target_recall,
            )
            >= self.confidence
        )
        # index of the first True value + 1
        j = np.argmax(coeffs) + 1
        return len(np.intersect1d(train_idxs, self.pre_positives)) >= j

    def __str__(self):
        return "QBCB"
