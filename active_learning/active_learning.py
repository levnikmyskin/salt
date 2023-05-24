from active_learning.base_policy import BaseAnnotatingPolicy, Inputs, Labels
from active_learning.stopping_strategy import StoppingStrategy
from utils.data_utils import take
from dataclasses import dataclass
from active_learning.batch_strategy import BatchStrategy
from typing import Set, Tuple, List, Iterable, Dict, Any
import numpy as np


@dataclass
class ActiveLearningConfig:
    estimator: BaseAnnotatingPolicy
    stopping_strats: List[StoppingStrategy]
    batch_strat: BatchStrategy
    x: Inputs
    y: Labels
    initial_seed: Iterable[int]
    validation_size: int = 0
    validation_step: int = 100
    stop_when_no_pos: bool = True


@dataclass
class Stop:
    it: int
    recall: float
    target: float


class ActiveLearning:
    def __init__(self, config: ActiveLearningConfig):
        self.estimator = config.estimator
        self.stopping_strats = config.stopping_strats
        self.batch_strat = config.batch_strat
        self.x = config.x
        self.y = config.y
        self.initial_seed = config.initial_seed
        self.rng = np.random.default_rng(seed=42)
        self.validation_size = config.validation_size
        self.validation_step = config.validation_step
        self.stop_when_no_pos = config.stop_when_no_pos
        self.stops: Dict[str, Stop] = {}

    def run(self, budget: int, pool_size=100_000) -> Tuple[List[int], List[int], List[int]]:
        train_idx_set = set(self.initial_seed)
        if self.validation_size > 0:
            val_idx_set = set(self.generate_validation_set(set(np.arange(pool_size)) - train_idx_set))
        else:
            val_idx_set = set()

        train_idx_list = list(train_idx_set)
        val_idx_list = list(val_idx_set)
        i = 0
        while self.__current_generated_len([train_idx_list, val_idx_set]) < budget:
            batch = self.estimator.get_next_batch(
                self.x,
                self.y,
                train_idx_list,
                val_idx_list,
                self.batch_strat.next_batch_size(),
            )

            for strat in filter(lambda s: str(s) not in self.stops, self.stopping_strats):
                if strat.should_stop(
                    self.x,
                    self.y,
                    train_idx_list,
                    val_idx_list,
                    np.copy(self.estimator.probas),
                    i,
                ):
                    self.stops[str(strat)] = Stop(
                        i,
                        (self.y[train_idx_list].sum() + self.y[val_idx_list].sum()) / self.y.sum(),
                        strat.target_recall,
                    )

            train_idx_list.extend(
                take(
                    self.batch_strat.current_batch_size(),
                    filter(lambda i: i not in train_idx_set and i not in val_idx_set, batch),
                )
            )
            train_idx_set.update(train_idx_list)

            if (((self.y[train_idx_list].sum() + self.y[val_idx_list].sum()) == self.y.sum()) and self.stop_when_no_pos) or len(
                self.stops
            ) == self.stopping_strats:
                break

            i += 1

        for strat in filter(lambda s: str(s) not in self.stops, self.stopping_strats):
            self.stops[str(strat)] = Stop(i, 1.0, strat.target_recall)
        return (
            train_idx_list,
            val_idx_list,
            list(set(np.arange(len(self.y))) - train_idx_set - val_idx_set),
        )

    def generate_validation_set(self, available_idxs: Set[int], epsilon=1e-3) -> List[int]:
        if not self.validation_size:
            return []
        means = []
        val_idxs = []
        vidxs = self.rng.choice(list(available_idxs), size=self.validation_step, replace=False)
        val_idxs.extend(vidxs)
        means.append(self.y[val_idxs].mean())

        i = 0
        check_every = int((self.validation_size / self.validation_step) // 5)
        # while abs(prev_mean - current_mean) > epsilon and self.validation_size > len(val_idxs):
        while self.validation_size > len(val_idxs):
            vidxs = self.rng.choice(list(available_idxs), size=self.validation_step, replace=False)
            val_idxs.extend(vidxs)
            if (i + 1) % (check_every * 2) == 0:
                m = np.array(means)
                m = np.median(m.reshape((int(m.shape[0] // check_every), check_every)), 1)
                m[np.where(m == 0)[0]] = epsilon
                if np.subtract.outer(m, m).mean() < epsilon:
                    break
            means.append(self.y[val_idxs].mean())
            i += 1
        return val_idxs

    def get_stops_as_dict(self) -> Dict[str, Dict[str, Any]]:
        return dict((k, v.__dict__) for k, v in self.stops.items())

    def __current_generated_len(self, idx_lists):
        return sum(len(i) for i in idx_lists)

    def __get_idxs_data(self, train_idxs):
        return self.x[train_idxs], self.y[train_idxs]
