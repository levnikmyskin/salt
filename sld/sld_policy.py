from typing import Optional
from active_learning.base_policy import BaseAnnotatingPolicy, Probs, Indices, Inputs, Labels
from sld.sld import run_sld, adjusted_sld
from utils.metrics import normalized_absolute_error
from numpy.typing import NDArray
from scipy.stats import entropy
import numpy as np


class SLDAnnotatingPolicy(BaseAnnotatingPolicy):
    def __init__(self, *args, **kwargs):
        self.use_adjusted = kwargs.pop('use_adjusted', False)
        self.use_sld_for_ranking = kwargs.pop('use_sld_ranking', False)
        self.sld_posteriors: Optional[NDArray[float]] = None
        self.clf_preds: Optional[NDArray[float]] = None
        self.mask: Optional[NDArray[bool]] = None
        super().__init__(*args, **kwargs)

    def rank_documents(self, x: Inputs, y: Labels, tr_idxs: Indices, val_idxs: Indices) -> Indices:
        if not self.use_sld_for_ranking:
            return super().rank_documents(x, y, tr_idxs, val_idxs)
        self.clf = self.classifier(*self.clf_args, **self.clf_kwargs)
        self.clf.fit(x[tr_idxs], y[tr_idxs])
        self.sld_posteriors, self.clf_preds, self.mask = self._get_sld_posteriors(x, y, tr_idxs, val_idxs, self.clf.predict_proba(x))
        self.clf_preds[self.mask] = self.sld_posteriors
        return (-self.clf_preds).argsort()

    def should_stop(self, x: Inputs, y: Labels, train_idxs: Indices, val_idxs: Indices) -> bool:
        if len(train_idxs) + len(val_idxs) == len(y):
            return True

        clf = self.classifier(*self.clf_args, **self.clf_kwargs)
        clf.fit(x[train_idxs], y[train_idxs])
        self.sld_posteriors, self.clf_preds, self.mask = self._get_sld_posteriors(x, y, train_idxs, val_idxs, clf.predict_proba(x))

        true_rels = y[train_idxs].sum() + y[val_idxs].sum()
        est_rels = self.sld_posteriors.sum()
        est_recall = (true_rels / (true_rels + est_rels))
        self.notify_new_recall(x, y, train_idxs, val_idxs, [est_recall, true_rels / (true_rels + self.clf_preds[self.mask].sum())])
        return est_recall >= self.target_recall

    def sort_probas(self, probas: Probs) -> Indices:
        return (-probas).argsort()

    def _get_sld_posteriors(self, x, y, tr_idxs, val_idxs, preds):
        mask = np.ones(len(y), dtype=bool)
        mask[np.concatenate((tr_idxs, val_idxs)).astype(int)] = False
        train_priors = np.array([1 - y[tr_idxs].mean(), y[tr_idxs].mean()])
        if self.use_adjusted:
            clf = self.classifier(*self.clf_args, **self.clf_kwargs)
            all_idxs = np.concatenate((tr_idxs, val_idxs))
            clf.fit(x[all_idxs], y[all_idxs])
            v_true_preds = clf.predict_proba(x[val_idxs])
            sld_posteriors, _ = adjusted_sld(preds[mask], preds[tr_idxs].mean(0), np.clip(1 - entropy(preds[val_idxs, 1], qk=v_true_preds[:, 1]), 0, 1))
            # clf_trust, k = self._get_adjusted_hyperparams(val_preds, train_priors,
            #                                               np.array([1 - y[val_idxs].mean(), y[val_idxs].mean()]), y[tr_idxs].sum(), y[val_idxs].sum())
            # sld_posteriors, _ = adjusted_sld(preds[mask], train_priors, clf_trust**k)
        else:
            sld_posteriors, _ = run_sld(preds[mask], train_priors)
        return sld_posteriors[:, 1], preds[:, 1], mask

    def _get_adjusted_hyperparams(self, val_preds: Probs, tr_priors: Inputs, val_true_priors: Inputs, tr_rel, val_rel):
        # clf_trust = normalized_absolute_error(val_preds[:, 1].mean(), val_true_priors[1])
        est_rels = val_preds[:, 1].mean() * len(val_preds)
        clf_trust = normalized_absolute_error(tr_rel / (tr_rel + val_rel), tr_rel / (tr_rel + est_rels))
        values = np.arange(0, 1.1, 0.1)
        tuning = [adjusted_sld(np.copy(val_preds), np.copy(tr_priors),
                               clf_trust ** v)[0] for v in values]
        # tuning = [adjusted_sld(np.copy(val_preds), tr_priors,
        #                        clf_trust ** v)[1] for v in values]
        # best = np.argmin(list(map(lambda i: normalized_absolute_error(i, val_true_priors[1]), tuning)))
        best = np.argmin(list(map(lambda i: normalized_absolute_error(tr_rel / (tr_rel + val_rel), tr_rel / (tr_rel + (i[:, 1].mean() * len(i)))), tuning)))
        values = np.arange(values[best] - 0.05, values[best] + 0.05, 0.01)
        tuning = [adjusted_sld(np.copy(val_preds), np.copy(tr_priors),
                               clf_trust ** v)[0] for v in values]
        best = np.argmin(list(map(lambda i: normalized_absolute_error(tr_rel / (tr_rel + val_rel), tr_rel / (tr_rel + (i[:, 1].mean() * len(i)))), tuning)))

        return clf_trust, values[best]
