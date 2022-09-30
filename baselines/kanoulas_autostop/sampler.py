"""
Code copied and (if necessary) adapted from https://github.com/dli1/auto-stop-tar
"""
import math
import numpy as np


class HorvitzThompson:
    """
    With-replacement sample, Horvitz-Thompson estimatation.
    """

    def __init__(self):
        self.N = 0
        self.dtype = np.float128
        self.complete_dids = None
        self.complete_labels = None
        self.cumulated_prod_1st_order = None
        self.cumulated_prod_2nd_order = None

        self.dist = None

    def init(self, complete_dids, complete_labels):
        self.N = len(complete_dids)

        self.complete_dids = complete_dids
        self.complete_labels = np.array(complete_labels).reshape((1, self.N))
        self.cumulated_prod_1st_order = np.zeros((1, self.N), dtype=self.dtype)
        self.cumulated_prod_2nd_order = np.zeros((self.N, self.N), dtype=self.dtype)
        return

    def update_distribution(self, **kwargs):
        """Make sampling distribution"""
        raise NotImplementedError

    def _reorder_dist(self, ranked_dids, dist):
        did_prob_dct = dict(zip(ranked_dids, dist))
        new_dist = []
        for did in self.complete_dids:
            new_dist.append(did_prob_dct[did])
        new_dist = np.array(new_dist, dtype=self.dtype).reshape((1, -1))
        return new_dist

    def _mask_sampled_dids(self, sampled_state):
        first_order_mask = np.array([1 if did in sampled_state else 0 for did in self.complete_dids])[None, :]

        M = np.tile(first_order_mask, (self.N, 1))
        MT = M.T
        second_order_mask = M * MT

        return first_order_mask, second_order_mask

    def _sample(self, ranked_dids, dist, n, replace):
        """Sample n items"""
        assert len(ranked_dids) == len(dist), 'dids len != dist len ({} != {})'.format(len(ranked_dids), len(dist))
        return np.random.choice(a=ranked_dids, size=n, replace=replace,
                                p=dist)  # returned type is the same with ranked_dids

    def _update(self, sampled_dids, ranked_dids, dist, stopping_condition):
        if sampled_dids is None:
            return

        n = len(sampled_dids)

        # update first order and second order inclusion probability
        dist = self._reorder_dist(ranked_dids, dist)
        self.cumulated_prod_1st_order += n * np.log(1.0 - dist)
        if stopping_condition == 'strict1' or stopping_condition == 'strict2':
            M = np.tile(dist, (self.N, 1))
            MT = M.T
            temp = 1.0 - M - MT
            np.fill_diagonal(temp, 1)  # set diagonal values to 1 to make sure log calculation is valid
            self.cumulated_prod_2nd_order += n * np.log(temp)

        return

    def sample(self, _, ranked_dids, n, stopping_condition):
        dist = self.dist
        sampled_dids = self._sample(ranked_dids, dist, n, replace=True)
        self._update(sampled_dids, ranked_dids, dist, stopping_condition)
        return sampled_dids  # note: it may contain duplicated dids

    def estimate(self, t, stopping_condition, sampled_state):
        # calculate inclusion probabilities
        first_order_inclusion_probabilities = 1.0 - np.exp(self.cumulated_prod_1st_order)
        assert np.min(first_order_inclusion_probabilities) > 0
        assert np.max(first_order_inclusion_probabilities) <= 1.0

        # total
        first_order_mask, second_order_mask = self._mask_sampled_dids(sampled_state)
        total = np.sum(first_order_mask * (self.complete_labels / first_order_inclusion_probabilities))

        if t > 1:
            variance1, variance2 = -1, -1  # no need to calculate variance if stopping condition is loose

            if stopping_condition == 'strict1' or stopping_condition == 'strict2':
                M = np.tile(first_order_inclusion_probabilities, (self.N, 1))
                MT = M.T
                second_order_inclusion_probabilities = M + MT - (1.0 - np.exp(self.cumulated_prod_2nd_order))
                assert np.min(second_order_inclusion_probabilities) > 0.0

                if stopping_condition == 'strict1':
                    # variance 1

                    # 1/pi_i**2 - 1/pi_i
                    part1 = 1.0 / first_order_inclusion_probabilities ** 2 - 1.0 / first_order_inclusion_probabilities
                    # y_i**2
                    yi_2 = self.complete_labels ** 2

                    # 1/(pi_i*pi_j) - 1/pi_ij
                    M = np.tile(first_order_inclusion_probabilities, (self.N, 1))
                    MT = M.T
                    part2 = 1.0 / (M * MT) - 1.0 / second_order_inclusion_probabilities
                    np.fill_diagonal(part2,
                                     0.0)  # set diagonal values to zero, because summing part2 do not include diagonal values

                    #  y_i * y_j
                    M = np.tile(self.complete_labels, (self.N, 1))
                    MT = M.T
                    yi_yj = M * MT

                    variance1 = np.sum(first_order_mask * part1 * yi_2) + np.sum(second_order_mask * part2 * yi_yj)

                if stopping_condition == 'strict2':
                    v = len(sampled_state)
                    if v == 1:
                        variance2 = 0
                    else:
                        # (v * y_i / pi_i - total)**2
                        variance2 = (v * self.complete_labels / first_order_inclusion_probabilities - total) ** 2
                        variance2 = (self.N - v) / self.N / v / (v - 1) * np.sum(first_order_mask * variance2)

        else:  # t=1 second order pi matrix is zero. no need to calcualate variance.
            variance1, variance2 = 0, 0

        return total, variance1, variance2


class HTAPPriorSampler(HorvitzThompson):
    def __init__(self):
        super(HTAPPriorSampler, self).__init__()

    def update_distribution(self):
        dist = [math.log((self.N+1) / (i + 1)) for i in np.arange(self.N)]  # i+1 from 1 to N
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return
