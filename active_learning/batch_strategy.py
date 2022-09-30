from abc import ABC, abstractmethod
import math


class BatchStrategy(ABC):
    def __init__(self, b=0):
        self._b = b

    @abstractmethod
    def next_batch_size(self) -> int:
        raise NotImplementedError()

    def peek_next_b(self) -> int:
        b = self._b
        next_ = self.next_batch_size()
        self._b = b
        return next_

    def current_batch_size(self) -> int:
        return self._b

    @abstractmethod
    def advance(self, iterations: int):
        raise NotImplementedError()


class CormackBatch(BatchStrategy):
    def next_batch_size(self) -> int:
        if self._b == 0:
            self._b = 1
        else:
            self._b += math.ceil(self._b / 10)
        return self._b

    def advance(self, iterations: int):
        raise NotImplementedError()


class LinearStrategy(BatchStrategy):

    def next_batch_size(self) -> int:
        return self._b

    def advance(self, iterations: int):
        annotated = self._b
        for i in range(iterations):
            annotated += self.next_batch_size()
        return annotated
