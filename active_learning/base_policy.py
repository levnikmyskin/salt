from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Protocol, Type, Iterable, Any, Mapping, NewType, Union, List
import numpy as np

Inputs = NewType("Inputs", NDArray[float])
Labels = NewType("Labels", NDArray[int])
Indices = NewType("Indices", NDArray[int])
Probs = NewType("Probs", NDArray[float])


class Classifier(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def fit(self, x_tr: Inputs, y_tr: Labels):
        ...

    def predict(self, x_te: Inputs) -> Labels:
        ...

    def predict_proba(self, x_te: Inputs) -> Probs:
        ...


class BaseAnnotatingPolicy(ABC):
    def __init__(
        self,
        classifier: Type[Classifier],
        clf_args: Iterable[Any],
        clf_kwargs: Mapping[str, Any],
    ):
        self.classifier = classifier
        self.clf_args = clf_args
        self.clf_kwargs = clf_kwargs
        self.clf = None
        self.probas = np.array([])

    def get_next_batch(
        self, x: Inputs, y: Labels, train_idxs: Indices, val_idxs: Indices, size: int
    ) -> Indices:
        return self.rank_documents(x, y, train_idxs, val_idxs)

    def rank_documents(
        self, x: Inputs, y: Labels, train_idxs: Indices, val_idxs: Indices
    ) -> Indices:
        self.clf = self.classifier(*self.clf_args, **self.clf_kwargs)
        self.clf.fit(x[train_idxs], y[train_idxs])
        self.probas = self.clf.predict_proba(x)
        return self.sort_probas()

    @abstractmethod
    def sort_probas(self) -> Indices:
        raise NotImplementedError()


class RelevancePolicy(BaseAnnotatingPolicy):
    def sort_probas(self) -> Indices:
        return (-self.probas[:, 1]).argsort()


class UncertaintyPolicy(BaseAnnotatingPolicy):
    def sort_probas(self) -> Indices:
        return (-(0.5 - self.probas[:, 1])).argsort()
