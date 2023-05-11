from active_learning.stopping_strategy import *
from scipy.stats import hypergeom


class CHMStopping(StoppingStrategyPreviousScores):
    def __init__(self, *args, alpha=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.pos_found = []
        self.len_annotated = []

    def __str__(self):
        return f"CHM @ {self.target_recall:.2f}"

    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        if len(self.prev_training_set) == 0:
            return False

        last_annotated = list(set(train_idxs) - set(self.prev_training_set))
        self.pos_found.append(y[last_annotated].sum())
        self.len_annotated.append(len(last_annotated))
        if i < self.min_rounds:
            return False

        pos_found = np.array(self.pos_found).cumsum()
        annotated_cumsum = np.array(self.len_annotated).cumsum()

        for j in range(1, i):
            if (
                hypergeom.cdf(
                    pos_found[-1] - pos_found[j],
                    len(y) - annotated_cumsum[j],
                    int(pos_found[-1] / self.target_recall - pos_found[j]),
                    annotated_cumsum[-1] - annotated_cumsum[j],
                )
                < self.alpha
            ):
                return True
        return False
