from abc import ABC, abstractmethod
from active_learning.base_policy import Inputs, Labels, Indices, Probs
from functools import wraps
from typing import List
import numpy as np


class StoppingListener(ABC):
    @abstractmethod
    def on_estimated_recall(
        self,
        x: Inputs,
        y: Labels,
        tr_idxs: Indices,
        val_idxs: Indices,
        est_recalls: List[float],
        iteration: int,
    ):
        raise NotImplementedError()

    @abstractmethod
    def on_new_batch(self, x: Inputs, y: Labels, tr_idxs: Indices, val_idxs: Indices, batch: Indices):
        raise NotImplementedError


class StoppingStrategy(ABC):
    def __init__(self, min_rounds=2, target_recall=0.9):
        self.min_rounds = min_rounds
        self.target_recall = target_recall
        self.listeners: List[StoppingListener] = []

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        raise NotImplementedError()

    def attach_listener(self, listener: StoppingListener):
        self.listeners.append(listener)

    def detach_listener(self, listener: StoppingListener):
        self.listeners.remove(listener)

    def notify_new_recall(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        recall: List[float],
        iteration: int,
    ):
        for listener in self.listeners:
            listener.on_estimated_recall(x, y, train_idxs, val_idxs, recall, iteration)


class StoppingStrategyPreviousScores(StoppingStrategy, ABC):
    def __new__(cls, *args, **kwargs):
        cls.should_stop = cls.__save_previous_state(cls.should_stop)
        return super().__new__(cls)

    def __init__(self, dataset_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_scores = np.empty((dataset_length, 2))
        self.prev_training_set = Indices(np.array([]))
        self.prev_test_set = Indices(np.array([]))

    @staticmethod
    def __save_previous_state(func):
        @wraps(func)
        def inner(self, x, y, tr_idxs, val_idxs, scores, i):
            stop = func(self, x, y, tr_idxs, val_idxs, scores, i)
            self.prev_scores[:] = scores
            self.prev_training_set = np.copy(tr_idxs)
            self.prev_test_set = list(set(np.arange(len(y))) - set(tr_idxs))
            return stop

        return inner

    def should_stop(
        self,
        x: Inputs,
        y: Labels,
        train_idxs: Indices,
        val_idxs: Indices,
        scores: Probs,
        i: int,
    ) -> bool:
        raise NotImplementedError()
