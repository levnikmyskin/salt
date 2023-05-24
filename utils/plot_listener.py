from active_learning.base_policy import Inputs, Labels, Indices
from active_learning.stopping_strategy import StoppingListener
from typing import List, Set
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from utils.data_utils import random_dataset_with_given_prevalences
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


class RecallPlotListener(StoppingListener):
    def __init__(self, name: str):
        self.recalls = []
        self.est_recalls = []
        self.num_annotateds = []
        self.val_len = -1
        self.name = name

    def on_estimated_recall(
        self,
        x: Inputs,
        y: Labels,
        tr_idxs: Indices,
        val_idxs: Indices,
        est_recall: List[float],
        iteration: int,
    ):
        self.recalls.append((y[tr_idxs].sum() + y[val_idxs].sum()) / y.sum())
        self.est_recalls.append(est_recall)
        self.num_annotateds.append(len(tr_idxs))
        self.val_len = len(val_idxs)

    def on_new_batch(self, x: Inputs, y: Labels, tr_idxs: Indices, val_idxs: Indices, batch: Indices):
        pass

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.num_annotateds, self.recalls, label="R")
        sld_rec = [r[0] for r in self.est_recalls]
        clf_rec = [r[1] for r in self.est_recalls]
        ax.plot(self.num_annotateds, sld_rec, label="$\hat{R} \ SLD$")
        ax.plot(self.num_annotateds, clf_rec, label="$\hat{R} \ CLF$")
        plt.suptitle(f"{self.name}; Val size: {self.val_len}. Rec: {self.recalls[-1]:.3}")
        fig.tight_layout()
        plt.legend()
        plt.show()
        plt.close()

    def flush(self, name):
        return self.__init__(name)


class DistribPlotListener(RecallPlotListener):
    def __init__(self, iter_save: Set[int], save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_save = iter_save
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def on_estimated_recall(
        self,
        x: Inputs,
        y: Labels,
        tr_idxs: Indices,
        val_idxs: Indices,
        est_recall: List[float],
        iteration: int,
    ):
        print(f"iteration {iteration}")
        if iteration not in self.iter_save:
            return
        clf = CalibratedClassifierCV(LinearSVC(), ensemble=False, n_jobs=20)
        clf_rand = CalibratedClassifierCV(LinearSVC(), ensemble=False, n_jobs=20)
        test_idxs = list(set(np.arange(len(y))) - set(tr_idxs))
        x_rand_tr, y_rand_tr, _, _ = random_dataset_with_given_prevalences(
            x, y, y[tr_idxs].mean(), y[test_idxs].mean(), len(tr_idxs), len(test_idxs)
        )
        clf.fit(x[tr_idxs], y[tr_idxs])
        clf_rand.fit(x_rand_tr, y_rand_tr)

        est_probs = clf.predict_proba(x)[test_idxs, 1]
        true_probs = clf_rand.predict_proba(x)[test_idxs, 1]
        sns.histplot(
            {
                r"AL $\mathrm{Pr}(y=1|x)$": est_probs,
                r"$Rand \ \mathrm{Pr}(y=1|x)$": true_probs,
            },
            log_scale=(False, True),
            element="step",
        )
        plt.title(r"$\mathrm{Pr}(y=1|x)$ histogram for AL and $Rand$ classifier. $|\mathcal{L}| = %d$" % (iteration * 100))
        plt.savefig(os.path.join(self.save_path, f"iteration_{iteration}.png"))
        plt.close()
        print(f"iteration {iteration} completed.")
