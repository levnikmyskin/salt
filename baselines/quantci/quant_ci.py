from active_learning.base_policy import (
    BaseAnnotatingPolicy,
    Probs,
    Indices,
    Inputs,
    Labels,
)
from active_learning.stopping_strategy import StoppingStrategy
import numpy as np


def quant_stop(probs, tr_idxs, val_idxs, te_idxs, nstd):
    tr_probs = probs[tr_idxs]
    val_probs = probs[val_idxs]
    te_probs = probs[te_idxs]
    known_sum = tr_probs.sum() + val_probs.sum()
    est_recall = known_sum / (known_sum + te_probs.sum())
    if nstd == 0:
        return est_recall
    prod = probs * (1 - probs)
    all_var = prod.sum()
    unknown_var = prod[te_idxs].sum()

    est_var = (known_sum**2 / (known_sum + te_probs.sum()) ** 4 * all_var) + (
        1 / (known_sum + te_probs.sum()) ** 2 * (all_var - unknown_var)
    )
    return est_recall - nstd * np.sqrt(est_var)


class QuantStopping(StoppingStrategy):
    def __init__(self, *args, nstd=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.nstd = nstd

    def __str__(self):
        return f"QuantCI {self.nstd} @ {self.target_recall:.2f}"

    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        if i < self.min_rounds:
            return False
        test_idxs = list(set(np.arange(len(y))) - set(train_idxs) - set(val_idxs))

        recall = quant_stop(scores[:, 1], train_idxs, val_idxs, test_idxs, self.nstd)
        self.notify_new_recall(x, y, train_idxs, val_idxs, [recall], i)
        return recall >= self.target_recall
