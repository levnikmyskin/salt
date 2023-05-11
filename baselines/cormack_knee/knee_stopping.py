from typing import Union, List, Tuple
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from active_learning.base_policy import Inputs, Labels, Indices, Probs
from active_learning.stopping_strategy import StoppingStrategy


class KneeStopping(StoppingStrategy):
    def __init__(self, *args, window=3, flats=10, rho="dynamic", **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.flats = flats
        self.rho: Union[str, float] = rho
        self.knee_data: List[Tuple[int, int]] = []

    def __str__(self):
        return "Knee"

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
        x_tr, y_tr = x[train_idxs], y[train_idxs]
        current_rels = y_tr.sum()
        n_assessed = len(y_tr)
        self.knee_data.append((n_assessed, current_rels))
        knee_indices = self.__detect_knee()
        if knee_indices:
            knee_index = knee_indices[-1]
            rank1, r1 = self.knee_data[knee_index]
            rank2, r2 = self.knee_data[-1]
            try:
                current_rho = float(r1 / rank1) / float((r2 - r1 + 1) / (rank2 - rank1))
            except ZeroDivisionError:
                current_rho = 0

            rho = 156 - min(current_rels, 150) if self.rho == "dynamic" else self.rho
            return self._check_rho(current_rho, rho, len(y), current_rels, n_assessed)
        return False

    def _check_rho(self, current_rho, rho, num_docs, current_rels, n_assessed):
        return current_rho > rho and n_assessed > self.min_rounds

    def __detect_knee(self) -> List[int]:
        """
        Implementation of the Knee method. Copied and adapted from
        https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/knee.py

        Detect the so-called knee in the data.
        The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle).

        Uses:
        self._knee_window: The data is smoothed using Gaussian kernel average smoother, this parameter is the window
            used for averaging (higher values mean more smoothing, try 3 to begin with).
        self._knee_flats: How many "flat" points to require before we consider it a knee.

        Alessio: I don't think in Cormack's paper the window size is explicitly mentioned.
        """

        knee_indices = []
        data_size = len(self.knee_data)
        data = np.array(self.knee_data)

        if data_size == 1:
            return knee_indices

        # smooth
        smoothed_data = []
        for i in range(data_size):
            if 0 < i - self.window:
                start_index = i - self.window
            else:
                start_index = 0
            if i + self.window > data_size - 1:
                end_index = data_size - 1
            else:
                end_index = i + self.window

            sum_x_weight = 0
            sum_y_weight = 0
            sum_index_weight = 0
            for j in range(start_index, end_index):
                index_weight = norm.pdf(abs(j - i) / self.window, 0, 1)
                sum_index_weight += index_weight
                sum_x_weight += index_weight * data[j][0]
                sum_y_weight += index_weight * data[j][1]

            smoothed_x = sum_x_weight / sum_index_weight
            smoothed_y = sum_y_weight / sum_index_weight

            smoothed_data.append((smoothed_x, smoothed_y))

        smoothed_data = np.array(smoothed_data)

        # normalize
        normalized_data = MinMaxScaler().fit_transform(smoothed_data)

        # difference
        differed_data = [(x, y - x) for x, y in normalized_data]

        # find indices for local maximums
        candidate_indices = []
        for i in range(1, data_size - 1):
            if (differed_data[i - 1][1] < differed_data[i][1]) and (
                differed_data[i][1] > differed_data[i + 1][1]
            ):
                candidate_indices.append(i)

        # threshold
        step = self.flats * (normalized_data[-1][0] - data[0][0]) / (data_size - 1)

        # knees
        for i in range(len(candidate_indices)):
            candidate_index = candidate_indices[i]

            if i + 1 < len(candidate_indices):  # not last second
                end_index = candidate_indices[i + 1]
            else:
                end_index = data_size

            threshold = differed_data[candidate_index][1] - step

            for j in range(candidate_index, end_index):
                if differed_data[j][1] < threshold:
                    knee_indices.append(candidate_index)
                    break
        return knee_indices
