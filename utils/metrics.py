import numpy as np
from numpy.typing import NDArray
from active_learning.batch_strategy import BatchStrategy
from typing import List, Union


def normalized_absolute_error(train_pr, test_pr) -> float:
    if type(train_pr) != np.ndarray:
        train_pr = np.array([1 - train_pr, train_pr])
    if type(test_pr) != np.ndarray:
        test_pr = np.array([1 - test_pr, test_pr])
    return np.sum(np.abs(train_pr - test_pr)) / (2 * (1 - np.min(train_pr)))


def mean_square_error(recall_t, recall_e) -> float:
    return abs(recall_t - recall_e) ** 2


def relative_error(recall_t, recall_e) -> float:
    return abs(recall_t - recall_e) / recall_t


class YangIdealizedCost:
    uniform = (1, 1, 1, 1)
    expensive_training = (10, 10, 1, 1)
    minecore_like = (1, 1, 5, 5)

    def __init__(self, cost_structure=uniform, initial_seed=2, name="$Cost_u$"):
        self.a_p, self.a_n, self.b_p, self.b_n = cost_structure
        self.initial_seed = initial_seed
        self.name = name

    @staticmethod
    def with_expensive_training(initial_seed=2):
        return YangIdealizedCost(YangIdealizedCost.expensive_training, initial_seed, name="$Cost_e$")

    @staticmethod
    def with_minecore_like(initial_seed=2):
        return YangIdealizedCost(YangIdealizedCost.minecore_like, initial_seed, name="$Cost_m$")

    def evaluate(
        self,
        b_strat: BatchStrategy,
        annotated_order: Union[List[int], NDArray[int]],
        y: NDArray[int],
        stopped_at: int,
        target_recall: float,
    ) -> float:
        n_annotated = b_strat.advance(stopped_at) + self.initial_seed
        if type(annotated_order) is list:
            annotated_order = np.asarray(annotated_order)
        annotated_at_stop = annotated_order[:n_annotated]
        remaining = annotated_order[n_annotated:]
        recall = y[annotated_at_stop].sum() / y.sum()
        recall_csum = y[annotated_order].cumsum() / y.sum()
        optimal_cost = np.where(recall_csum >= target_recall)[0][0] + 1
        annotated_cost = self.costs_of_annotated_docs(y[annotated_at_stop])
        docs_to_target = optimal_cost - n_annotated
        non_ann_cost = (self.b_p * y[remaining][:docs_to_target].sum()) + (self.b_n * (~y[remaining][:docs_to_target].astype(bool)).sum())
        if recall >= target_recall:
            non_ann_cost = 0
        return annotated_cost + non_ann_cost

    def costs_of_annotated_docs(self, y: NDArray[int]) -> float:
        return self.a_p * y.sum() + (self.a_n * (~y.astype(bool)).sum())

    def __str__(self):
        return self.name


"""
Minimum Distance of Pair Assignments (MDPA) [cha2002measuring] for ordinal pdfs `a` and `b`.
The MDPA is a special case of the Earth Mover's Distance [rubner1998metric] that can be
computed efficiently.
[Mirko Bunse's code from Julia adapted]
"""


def mdpa(a, b):
    assert len(a) == len(b), "histograms have to have the same length"
    assert np.isclose(sum(a), sum(b)), "histograms have to have the same mass (difference is $(sum(a)-sum(b))"

    # algorithm 1 in [cha2002measuring]
    prefixsum = 0.0
    distance = 0.0
    for i in range(len(a)):
        prefixsum += a[i] - b[i]
        distance += abs(prefixsum)

    return distance / sum(a)  # the normalization is a fix to the original MDPA
