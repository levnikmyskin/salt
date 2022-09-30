from active_learning.base_policy import BaseAnnotatingPolicy, Probs, Indices, Inputs, Labels
from baselines.kanoulas_autostop.sampler import HTAPPriorSampler
from scipy.stats import entropy
from sld.sld_policy import adjusted_sld
import numpy as np


class AutoStopAnnotatingPolicy(BaseAnnotatingPolicy):
    """
    Code was adapted from
    https://github.com/dli1/auto-stop-tar/blob/a7c96c90c5f524ec9469c12cb1c1779a0899cd9a/autostop/tar_model/auto_stop.py
    """
    def __init__(self, *args, **kwargs):
        self.stopping_condition = kwargs.pop('stopping_condition')
        super().__init__(*args, **kwargs)
        self.sampler = HTAPPriorSampler()
        self.current_iter = 0

    def get_next_batch(self, x: Inputs, y: Labels, tr_idxs: Indices, _: Indices, size: int) -> Indices:
        sorted_idxs = self.rank_documents(x, y, tr_idxs, _)
        self.current_iter += 1
        return np.unique(self.sampler.sample(None, sorted_idxs, size, self.stopping_condition))

    def sort_probas(self, probas: Probs) -> Indices:
        return (-probas).argsort()

    def should_stop(self, x: Inputs, y: Labels, train_idxs: Indices, _: Indices) -> bool:
        total_esti_rels, var_1, var_2 = self.sampler.estimate(self.current_iter, self.stopping_condition, set(train_idxs))
        current_rels = y[train_idxs].sum()
        match self.stopping_condition:
            case "loose":
                return current_rels >= self.target_recall * total_esti_rels
            case "strict1":
                return current_rels >= self.target_recall * (total_esti_rels + np.sqrt(var_1))
            case "strict2":
                return current_rels >= self.target_recall * (total_esti_rels + np.sqrt(var_2))
        return False


class SLDAutoStopAnnotatingPolicy(AutoStopAnnotatingPolicy):
    def rank_documents(self, x: Inputs, y: Labels, train_idxs: Indices, val_idxs: Indices) -> Indices:
        self.clf = self.classifier(*self.clf_args, **self.clf_kwargs)

        self.clf.fit(x[train_idxs], y[train_idxs])
        preds = self.clf.predict_proba(x)
        test_idxs = list(set(np.arange(len(y))) - set(train_idxs) - set(val_idxs))
        self.sld_posteriors, _ = self._get_sld_posteriors(x, y, train_idxs, test_idxs, val_idxs, preds)
        preds[test_idxs, 1] = self.sld_posteriors
        return (-preds[:, 1]).argsort()

    def _get_sld_posteriors(self, x, y, tr_idxs, te_idxs, val_idxs, preds):
        clf = self.classifier(*self.clf_args, **self.clf_kwargs)
        all_idxs = np.concatenate((tr_idxs, val_idxs))
        clf.fit(x[all_idxs], y[all_idxs])
        v_true_preds = clf.predict_proba(x[val_idxs])
        sld_posteriors, _ = adjusted_sld(preds[te_idxs], preds[tr_idxs].mean(0),
                                         np.clip(1 - entropy(preds[val_idxs, 1], qk=v_true_preds[:, 1]), 0, 1))
        return sld_posteriors[:, 1], v_true_preds[:, 1]
