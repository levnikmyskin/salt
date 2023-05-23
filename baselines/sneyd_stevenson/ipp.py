from active_learning.base_policy import (
    Probs,
    Indices,
    Inputs,
    Labels,
)
from active_learning.stopping_strategy import StoppingStrategy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import poisson
import numpy as np


class IPP(StoppingStrategy):
    def __init__(self, *args, n_windows, min_pos, **kwargs):
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
        windows, window_size = self.__make_windows(len(train_idxs))
        x_p, y_p = self.__power_curve_points(windows, window_size, y[train_idxs])

    def __make_windows(self, train_len: int):
        window_size = round(train_len / self.n_windows)
        # train_len - 1 and windows_size + 1 used for correct indexing
        # eg. if train_len == 100 and n_windows == 10, then we create
        # np.linspace(0, 99, 11) -> [0, 9, 19, 29, ... ,99]
        windows = np.linspace(0, train_len - 1, window_size + 1)
        return sliding_window_view(windows, 2), window_size

    def __power_curve_points(self, windows, window_size, y_tr):
        return windows[:, 0] + 1, y_tr[windows].sum(axis=1) / window_size

    def __fit_curve(self, x, a, k):
        return a * x**k

    def __predict_n_rel(self, des_prob, n_docs, mu):
        return np.argmin(poisson.cdf(np.arange(n_docs) + 1, mu) < des_prob) + 1
