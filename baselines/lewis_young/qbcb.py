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
    def __init__(self, *args, positive_sample_size=50, confidence=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_sample_size = positive_sample_size
        self.confidence = confidence
        self.pre_positives = None

    def pre_annotation(self, x: Inputs, y: Labels) -> int:
        # Generate the random sample `n` from which we take the positive
        # sample size `r` (paper notation).
        # This must be called before using the class as a stopping strategy.
        perm = np.random.default_rng(42).permutation(
            np.vstack([x, y, np.arange(len(y))])
        )
        idx = np.where(perm[:, 1].cumsum() == self.positive_sample_size)[0][0] + 1
        self.pre_positives = perm[:, 2][np.where(perm[:, 1][:idx] == 1)]

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
            raise ValueError(
                "The positive sample is None. You should call `pre_annotation` before "
                "using this class as a stopping strategy."
            )
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
