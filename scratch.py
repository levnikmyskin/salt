import numpy as np

from run_experiments import *
from sld.sld import adjusted_sld, run_sld
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import make_classification
from utils.data_utils import take, random_dataset_with_given_prevalences
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import concurrent.futures
import copy


# dataset = fetch_rcv1()


def plot_cluster_with_pos(title, x_svd, cluster_preds, y, idx_to_plot, idx_annotated):
    if idx_to_plot:
        y_idxs = y[idx_to_plot] == 1
        plt.scatter(
            x_svd[idx_to_plot, 0],
            x_svd[idx_to_plot, 1],
            c=cluster_preds[idx_to_plot],
            alpha=0.2,
        )
        plt.scatter(x_svd[idx_to_plot, 0][y_idxs], x_svd[idx_to_plot, 1][y_idxs], c="red")
    else:
        plt.scatter(x_svd[:, 0], x_svd[:, 1], c=cluster_preds, alpha=0.2)
        plt.scatter(x_svd[y == 1, 0], x_svd[y == 1, 1], c="red")
    if idx_annotated is not None:
        plt.scatter(x_svd[idx_annotated, 0], x_svd[idx_annotated, 1], c="brown", alpha=0.8)
        plt.scatter(
            x_svd[idx_annotated, 0][y[idx_annotated] == 1],
            x_svd[idx_annotated, 1][y[idx_annotated] == 1],
            c="blue",
        )
    plt.title(title)
    plt.show()


def run_on_fake_data(sample_size=10_000):
    x, y = make_classification(
        n_samples=sample_size,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1,
        weights=[0.1, 0.4, 0.1, 0.4],
    )
    clusters = y
    y_c = np.copy(y)
    y_c[np.logical_or(y == 0, y == 2)] = 1
    y_c[~np.logical_or(y == 0, y == 2)] = 0
    random_pos = np.random.choice(np.where(clusters == 0)[0], size=2)
    random = np.random.choice(np.where(y_c == 0)[0], replace=False, size=10)
    init_s = np.concatenate((random_pos, random))
    train_idxs = list(init_s)
    for _ in tqdm(range(5)):
        clf = calibrated_svm(n_jobs=20, ensemble=False, cv=min(y_c[train_idxs].sum(), 10))
        # clf = LogisticRegression(n_jobs=20)
        clf.fit(x[train_idxs], y_c[train_idxs])
        sort_preds = (-clf.predict_proba(x)[:, 1]).argsort()

        new_train = take(100, iter(t for t in sort_preds if t not in train_idxs))
        train_idxs.extend(new_train)
        test_idxs = list(set(np.arange(len(y_c))) - set(train_idxs))
        if y_c[test_idxs].sum() == 0:
            break

    clf = calibrated_svm(n_jobs=20, ensemble=False, cv=min(y_c[train_idxs].sum(), 10))
    # clf = LogisticRegression(n_jobs=20)
    clf.fit(x[train_idxs], y_c[train_idxs])
    al_probs = clf.predict_proba(x)
    test_idxs = list(set(np.arange(len(y_c))) - set(train_idxs))
    rand_tr, rand_te = random_dataset_with_given_prevalences(
        x,
        y_c,
        y_c[train_idxs].mean(),
        y_c[test_idxs].mean(),
        len(train_idxs),
        len(test_idxs),
        return_idxs=True,
    )
    clf = calibrated_svm(n_jobs=20, ensemble=False, cv=min(y_c[rand_tr].sum(), 10))
    # clf = LogisticRegression(n_jobs=20)
    clf.fit(x[rand_tr], y_c[rand_tr])
    rand_probs = clf.predict_proba(x)

    colors = ["#4EACC5", "#fc79fc", "#FF9C34", "#af62f7"]
    tr_colors = ["#4E6FC4", "m", "#FF7D32", "#6f00d8"]
    tr_alphas = [0.6, 1.0, 0.6, 1.0]
    min_xs = sorted(((x[clusters == i, 0].mean(), i) for i in set(clusters)), key=lambda k: k[0])
    min_ys = sorted(((x[clusters == i, 1].mean(), i) for i in set(clusters)), key=lambda k: k[0])
    fig1 = plt.figure(1, figsize=(12, 10))
    fig2 = plt.figure(2, figsize=(12, 10))
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    for c in set(clusters):
        cl_ids = clusters[test_idxs] == c
        rand_cl = clusters[rand_tr] == c
        ax1.scatter(x[test_idxs, 0][cl_ids], x[test_idxs, 1][cl_ids], c=colors[c], alpha=0.2)
        ax1.scatter(
            x[train_idxs, 0][clusters[train_idxs] == c],
            x[train_idxs, 1][clusters[train_idxs] == c],
            marker="x",
            c=tr_colors[c],
            alpha=tr_alphas[c],
        )
        ax2.scatter(x[test_idxs, 0][cl_ids], x[test_idxs, 1][cl_ids], c=colors[c], alpha=0.2)
        ax2.scatter(
            x[rand_tr, 0][rand_cl],
            x[rand_tr, 1][rand_cl],
            marker="x",
            c=tr_colors[c],
            alpha=tr_alphas[c],
        )
        coords = (x[clusters == c, 0].mean(), x[clusters == c, 1].mean())
        if c == min_xs[0][1] or c == min_xs[1][1]:
            x_coord = x[clusters == c, 0].min()
        else:
            x_coord = x[clusters == c, 0].max()

        if c == min_ys[0][1] or c == min_ys[1][1]:
            y_coord = x[clusters == c, 1].min()
        else:
            y_coord = x[clusters == c, 1].max()
        ax1.annotate(
            f"AL {al_probs[test_idxs, 1][cl_ids].mean():.3f}\n"
            f"$Rand$ {rand_probs[test_idxs, 1][cl_ids].mean():.3f}\n"
            f"True {y_c[test_idxs][cl_ids].sum() / y_c[clusters == c].sum():.3f}",
            coords,
            xytext=(x_coord, y_coord),
            bbox={"boxstyle": "round", "fc": colors[c], "alpha": 0.4},
            arrowprops={"arrowstyle": "->"},
        )
    ax1.set_title(
        "$P_{U}(y) = %.3f; \hat{P}_{U}^{\mathrm{AL}}(y) = %.3f; \hat{P}_{U}^{Rand}(y) = %.3f$"
        % (
            y_c[test_idxs].mean(),
            al_probs[test_idxs, 1].mean(),
            rand_probs[test_idxs, 1].mean(),
        )
    )
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    plt.show()


if __name__ == "__main__":
    run_on_fake_data()
    # np.random.seed(None)
    # pool_size = 10_000
    # #c152
    # pool_idxs = np.random.choice(np.arange(dataset.data.shape[0]), replace=False, size=pool_size)
    # # pool_idxs = np.arange(pool_size)
    #
    # x, y = dataset.data[pool_idxs], dataset.target[pool_idxs].toarray()
    # classes = np.where(np.logical_and(y.mean(0) >= 0.01, y.mean(0) <= 0.1))[0]
    #
    # for c in tqdm(np.random.choice(classes, replace=False, size=5)):
    # # c = dataset.target_names.tolist().index('C17')
    #     run_on_class(c, x, y)
    # # with concurrent.futures.ProcessPoolExecutor(5) as p:
    # #     for c in np.random.choice(classes, replace=False, size=5):
    # #         futures.append(p.submit(run_on_class, c, x, y))
    # #
    # #     for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    # #         res[c] = f.result()
