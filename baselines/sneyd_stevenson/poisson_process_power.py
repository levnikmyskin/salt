from active_learning.base_policy import Inputs, Labels, Indices, Probs
from active_learning.stopping_strategy import StoppingStrategy


class IPP(StoppingStrategy):
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

    def __str__(self):
        return "IPP"
