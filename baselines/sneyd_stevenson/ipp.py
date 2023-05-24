from active_learning.base_policy import (
    Probs,
    Indices,
    Inputs,
    Labels,
)
from active_learning.stopping_strategy import StoppingStrategy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import poisson
from scipy.optimize import curve_fit
import numpy as np


def model_func_power(x, a, k):
    return a * x**k


class IPP(StoppingStrategy):
    """
    Adapted from https://github.com/alisonsneyd/stopping_criteria_counting_processes/blob/main/run_stopping_point_experiments.ipynb
    """

    def __init__(self, *args, n_windows=10, min_pos=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_windows = n_windows
        self.min_pos = min_pos

    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        if i < self.min_rounds or y[train_idxs].sum() < self.min_pos:
            return False
        windows, window_edges, window_size = self.__make_windows(len(train_idxs))
        x_p, y_p = self.__power_curve_points(windows, window_edges, window_size, y[train_idxs])
        try:
            p0 = [0.1, 0.001]
            (a, k), pcov = curve_fit(model_func_power, x_p, y_p, p0)

            y_sum = y[train_idxs].sum()
            est_by_curve_end_samp = self.__distance_between_curves(y[train_idxs], a, k)
            if y_sum >= self.target_recall * est_by_curve_end_samp:
                mu = (a / (k + 1)) * (len(y) ** (k + 1) - 1)
                pred_n_rel = self.__predict_n_rel(0.95, len(train_idxs), mu)
                des_n_rel = self.target_recall * pred_n_rel
                return des_n_rel <= y_sum
        except:
            return False
        return False

    def __distance_between_curves(self, y_tr, a, k):
        y3 = model_func_power(np.arange(1, len(y_tr) + 1), a, k)
        return round(np.sum(y3))

    def __make_windows(self, train_len: int):
        # In the original code, the last group is discarded.
        window_size = train_len // self.n_windows
        indices = np.arange(train_len)
        windows = indices[: window_size * self.n_windows].reshape(-1, self.n_windows)
        window_edges = indices[:, [0, -1]]
        return windows, window_edges, window_size

    def __power_curve_points(self, windows, window_edges, window_size, y_tr):
        return window_edges[:, 0] + 1, y_tr[windows].sum(axis=1) / window_size

    def __predict_n_rel(self, des_prob, n_docs, mu):
        return np.argmin(poisson.cdf(np.arange(n_docs) + 1, mu) < des_prob) + 1

    def __str__(self):
        return "IPP"
