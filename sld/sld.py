import numpy as np
from collections import namedtuple
from scipy.stats import entropy

History = namedtuple("History", ("iteration", "priors", "posteriors", "stop_criterium", "accuracy", "precision", "recall"))


def run_sld(posteriors_zero, priors_zero, epsilon=1e-6):
    """
    Implements the prior correction method based on EM presented in:
    "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    Saerens, Latinne and Decaestecker, 2002
    http://www.isys.ucl.ac.be/staff/marco/Publications/Saerens2002a.pdf
    :param posteriors_zero: posterior probabilities on test items, as returned by a classifier. A 2D-array with shape
    Ø(items, classes).
    :param priors_zero: prior probabilities measured on training set.
    :param epsilon: stopping threshold.
    :return: posteriors_s, priors_s, history: final adjusted posteriors, final adjusted priors, a list of length s
    where each element is a tuple with the step counter, the current priors (as list), the stopping criterium value,
    accuracy, precision and recall.
    """
    s = 0
    priors_s = posteriors_zero.mean(0)
    posteriors_s = np.copy(posteriors_zero)
    val = 2 * epsilon
    while not val < epsilon and s < 1000:
        # E step
        ratios = (priors_s + 1e-10) / (priors_zero + 1e-10)
        denominators = 0
        for c in range(priors_zero.shape[0]):
            denominators += ratios[c] * posteriors_zero[:, c]
        for c in range(priors_zero.shape[0]):
            posteriors_s[:, c] = ratios[c] * posteriors_zero[:, c] / denominators

        priors_s_minus_one = priors_s.copy()
        # M step
        priors_s = posteriors_s.mean(0)

        # check for stop
        val = 0
        for i in range(len(priors_s_minus_one)):
            val += abs(priors_s_minus_one[i] - priors_s[i])
        s += 1

    return posteriors_s, priors_s


def adjusted_sld(posteriors_zero, priors_zero, clf_trustability, epsilon=1e-6, tau=1):
    s = 0
    priors_s = posteriors_zero.mean(0)
    posteriors_s = np.copy(posteriors_zero)
    val = 2 * epsilon
    while not val < epsilon and s < 1000:
        # E step
        ratios = (priors_s + 1e-10) / (priors_zero + 1e-10)
        ratios = -(clf_trustability * (-ratios + 1)) ** tau + 1

        denominators = 0
        for c in range(priors_zero.shape[0]):
            denominators += ratios[c] * posteriors_zero[:, c]
        for c in range(priors_zero.shape[0]):
            posteriors_s[:, c] = ratios[c] * posteriors_zero[:, c] / denominators

        priors_s_minus_one = priors_s.copy()

        # M step
        priors_s = posteriors_s.mean(0)

        # check for stop
        val = 0
        for i in range(len(priors_s_minus_one)):
            val += abs(priors_s_minus_one[i] - priors_s[i])
        s += 1

    return posteriors_s, priors_s


def adjusted_sld_from_val(x, y, tr_idxs, te_idxs, val_idxs, probs, clf):
    all_idxs = np.concatenate((tr_idxs, val_idxs))
    clf.fit(x[all_idxs], y[all_idxs])
    v_true_preds = clf.predict_proba(x[val_idxs])
    sld_posteriors, _ = adjusted_sld(probs[te_idxs], probs[tr_idxs].mean(0),
                                     np.clip(1 - entropy(probs[val_idxs, 1], qk=v_true_preds[:, 1]), 0, 1))
    return sld_posteriors[:, 1], v_true_preds[:, 1]
