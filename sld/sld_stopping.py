import numpy as np
from baselines.quantci.quant_ci import quant_stop
from sld.sld import adjusted_sld, run_sld
from active_learning.base_policy import Inputs, Labels, Indices, Probs
from active_learning.stopping_strategy import StoppingStrategyPreviousScores
from scipy.spatial.distance import cosine
from utils.metrics import normalized_absolute_error
from functools import cached_property


class SLDQuantStopping(StoppingStrategyPreviousScores):
    def __init__(self, *args, nstd=0, alpha=0.3, use_adjusted=True, use_margin=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nstd = nstd
        self.alpha = alpha
        self.use_adjusted = use_adjusted
        self.use_margin = use_margin

    @cached_property
    def margin_target_recall(self):
        return 2 * self.target_recall - self.target_recall ** 2 if self.use_margin else self.target_recall

    def __str__(self):
        # Notice that (following our paper terminology):
        #  - SALtQuantCI has self.nstd = 2;
        #  - SALt has self.nstd = 0, self.use_margin = False;
        #  - SAL^r_t ha self.nstd = 0, self.use_margin = True;
        return f"SLDQuant {self.nstd} & a={self.alpha} @ {self.target_recall:.2f} w\\ m={int(self.use_margin)}"

    def should_stop(self, x: Inputs, y: Labels, train_idxs: Indices, val_idxs: Indices, scores: Probs, i: int) -> bool:
        if i < self.min_rounds:
            return False
        last_annotated = list(set(train_idxs) - set(self.prev_training_set))
        current_test = list(set(np.arange(len(y))) - set(train_idxs))
        if self.use_adjusted:
            if i < 1:
                sc = np.copy(scores[:, 1])
            else:
                sld_post, _ = adjusted_sld(self.prev_scores[self.prev_test_set], self.prev_scores[self.prev_training_set].mean(0),
                                           1 - cosine(self.prev_scores[last_annotated, 1], scores[last_annotated, 1]))
                sc = np.copy(scores[:, 1])
                sc[self.prev_test_set] = sld_post[:, 1]
                if normalized_absolute_error(y[last_annotated].mean(), sc[last_annotated].mean()) > self.alpha:
                    sc = np.copy(scores[:, 1])
        else:
            if i < 5:
                return False
            sld_post, _ = run_sld(scores[current_test], self.prev_scores[current_test].mean(0))
            sc = np.copy(scores[:, 1])
            sc[current_test] = sld_post[:, 1]

        recall = quant_stop(sc, train_idxs, val_idxs, current_test, self.nstd)
        self.notify_new_recall(x, y, train_idxs, val_idxs, [recall], i)
        return recall >= self.margin_target_recall
