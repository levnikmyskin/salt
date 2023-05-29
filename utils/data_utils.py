import enum
import itertools
import re
from typing import Optional, Set, Match, Generator
from tmt import TmtManager
from numpy.typing import NDArray
import numpy as np

SEED = 42


pattern = re.compile(r"(?P<policy>(ALvUS|ALvRS|PL|ALvDS))_(?P<label>.+)_(?P<size>\d+)size_(?P<classifier>.+)_results\.pkl")


class ALPolicy(enum.Enum):
    RELEVANCE_SAMPLING = enum.auto()
    UNCERTAINTY_SAMPLING = enum.auto()
    PASSIVE_LEARNING = enum.auto()
    DIVERSITY_SAMPLING = enum.auto()
    TEST_SAMPLING = enum.auto()

    @staticmethod
    def from_string(string: str) -> "ALPolicy":
        if string == "ALvRS" or string == "RS":
            return ALPolicy.RELEVANCE_SAMPLING
        elif string == "ALvUS" or string == "US":
            return ALPolicy.UNCERTAINTY_SAMPLING
        elif string == "PL":
            return ALPolicy.PASSIVE_LEARNING
        elif string == "ALvDS" or string == "DS":
            return ALPolicy.DIVERSITY_SAMPLING

    def __str__(self):
        if self is self.RELEVANCE_SAMPLING:
            return "ALvRS"
        elif self is self.UNCERTAINTY_SAMPLING:
            return "ALvUS"
        elif self is self.PASSIVE_LEARNING:
            return "PL"
        elif self is self.DIVERSITY_SAMPLING:
            return "ALvDS"


def random_dataset_with_given_prevalences(x, y, tr_prev, te_prev, tr_size, te_size, seed=None, return_idxs=False):
    tr_pos_to_take = round(tr_size * tr_prev)
    tr_neg_to_take = abs(tr_size - tr_pos_to_take)
    te_pos_to_take = round(te_size * te_prev)
    te_neg_to_take = abs(te_pos_to_take - te_size)

    x_tr, y_tr, tr_idxs = __get_xy_with_given_pos_and_negs(x, y, tr_pos_to_take, tr_neg_to_take, seed)

    te_indices = set(np.arange(y.shape[0])) - set(tr_idxs)
    if not return_idxs:
        x_te, y_te = x[list(te_indices)], y[list(te_indices)]
        x_te, y_te, _ = __get_xy_with_given_pos_and_negs(x_te, y_te, te_pos_to_take, te_neg_to_take, seed)
        return x_tr, y_tr, x_te, y_te
    return tr_idxs, np.array(list(te_indices))


def random_sample_from_dataset(x, y, tr_size: int, seed=None):
    indices = np.arange(y.shape[0])
    training_idxs = np.random.default_rng(seed=seed).choice(indices, size=tr_size, replace=False)
    test_idxs = list(set(indices) - set(training_idxs))
    return x[training_idxs], y[training_idxs], x[test_idxs], y[test_idxs]


def __get_xy_with_given_pos_and_negs(x, y, pos_to_take, neg_to_take, seed):
    positives = np.random.default_rng(seed=seed).choice(np.where(y == 1)[0], size=pos_to_take, replace=False)
    negatives = np.random.default_rng(seed=seed).choice(np.where(y == 0)[0], size=neg_to_take, replace=False)
    idxs = np.hstack((positives, negatives))
    np.random.default_rng().shuffle(idxs)
    return x[idxs], y[idxs], idxs


def flatten(list_of_lists):
    # Flatten one level of nesting. See https://docs.python.org/3.6/library/itertools.html#recipes
    return itertools.chain.from_iterable(list_of_lists)


def take(n, iterable):
    # Return first n items of the iterable as a list
    return list(itertools.islice(iterable, n))


def filter_file(
    filename: str,
    policy: Optional[ALPolicy],
    classifier: Optional[str],
    sizes: Optional[Set[int]],
    labels: Optional[Set[str]],
) -> Optional[Match]:
    match = pattern.match(filename)
    if not match:
        return None
    if policy and ALPolicy.from_string(match.group("policy")) is not policy:
        return None
    if classifier and match.group("classifier") != classifier.replace(" ", ""):
        return None
    if sizes and int(match.group("size")) not in sizes:
        return None
    if labels and match.group("label") not in labels:
        return None

    return match


def aggregate_same_sizes(policy, classifier, file_list, sizes, labels):
    aggregated = {}
    for file in file_list:
        match = filter_file(file, policy, classifier, sizes, labels)
        if not match:
            continue
        size_f = aggregated.setdefault(int(match.group("size")), [])
        size_f.append(file)
    return aggregated


class PreviousRunUtils:
    def __init__(self, name: str):
        self.manager = TmtManager()
        self.manager.set_entry_by_name(name)
        self.results = list(self.manager.load_results())[0][1]
        self.class_runs = {}

    def get_idxs_iter(self) -> Generator[list[int], None, None]:
        for cls_, data in self.results.items():
            yield cls_, data.get("idxs", []), data["y_c"]

    def get_idxs(self, cls_: str, run: Optional[int] = None) -> list[int]:
        if run is None:
            run = self.class_runs.setdefault(cls_, -1) + 1
            self.class_runs[cls_] = run
        self.results[f"{cls_}_{run}"]["idxs"][0]

    def get_qbcb_sample(self, recall: str, cls_run: str) -> (NDArray[int], NDArray[int]):
        sample = self.results[f"{cls_run}"][f"QBCB@{recall} presample"]
        pre = self.results[f"{cls_run}"][f"QBCB@{recall} prepositives"]
        return sample, pre
