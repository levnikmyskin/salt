from tmt import tmt_recorder, tmt_save
from concurrent.futures import ProcessPoolExecutor, as_completed
from active_learning.active_learning import ActiveLearning, ActiveLearningConfig
from active_learning.base_policy import RelevancePolicy, UncertaintyPolicy
from active_learning.batch_strategy import CormackBatch, LinearStrategy
from baselines import cormack_knee, lewis_yang, callaghan_chm, sneyd_stevenson
from sld import SLDQuantStopping
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_rcv1
from tqdm import tqdm
from utils.plot_listener import RecallPlotListener
import pandas as pd
import argparse
import numpy as np
import copy
import sys


def run_al(name, budget, x, y_c, pool_size, policy, stoppings):
    res = {"y_c": y_c}
    if y_c.sum() < 2:
        return name, {}
    random_pos = np.random.choice(np.where(y_c == 1)[0], size=1)
    random = np.random.choice(np.where(y_c == 0)[0], replace=False, size=1)
    conf = ActiveLearningConfig(
        policy,
        stoppings,
        LinearStrategy(b=100),
        x,
        y_c,
        np.concatenate((random_pos, random)),
        stop_when_no_pos=False,
    )
    al = ActiveLearning(conf)
    res["idxs"] = al.run(budget, pool_size)
    res["stops"] = al.get_stops_as_dict()
    for qbcb in filter(lambda s: type(s) is lewis_yang.QBCB, stoppings):
        res[f"{qbcb}@{qbcb.target_recall} presample"] = qbcb.pre_sample
        res[f"{qbcb}@{qbcb.target_recall} prepositives"] = qbcb.pre_positives

    return name, res


def process_futures(futures, res_name):
    results = {}
    for future in tqdm(as_completed(futures), total=len(futures)):
        name, res = future.result()
        results[name] = res
    tmt_save(results, res_name)


def calibrated_svm(**kwargs):
    return CalibratedClassifierCV(LinearSVC(), **kwargs)


if __name__ == "__main__":
    paper_name = ""
    parser = argparse.ArgumentParser(f"Run experiments for paper {paper_name}")
    parser.add_argument("-j", "--jobs", type=int, help="number of processes to spawn")
    parser.add_argument("-lrj", "--lr-j", type=int, help="number of jobs for the LR")
    # parser.add_argument('-b', '--batch', choices=[CormackBatch], default=CormackBatch,
    #                    help='Batch strategy to use. Only Cormack\'s available atm.')
    parser.add_argument(
        "-t",
        "--target-recall",
        nargs="+",
        type=float,
        help="target recall TAR should stop at",
        required=True,
    )
    parser.add_argument("-d", "--description", help="experiment description for tmt")
    parser.add_argument(
        "-p",
        "--pool-size",
        type=int,
        help="size of the pool to annotate",
        default=10_000,
    )
    parser.add_argument("-n", "--name", help="tmt save name", required=True)
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument(
        "-r", "--runs", type=int, default=20, help="number of random runs"
    )
    parser.add_argument(
        "--policy", choices=["RS", "US"], help="active learning policy", default="RS"
    )
    parser.add_argument(
        "--debugging",
        action="store_true",
        help="if specified, the number of classes will be reduced,"
        "tmt and multiprocessing will not be used. Option"
        "--runs will also be ignored",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    pool_size = args.pool_size

    # clf_kwargs = {'n_jobs': args.lr_j , 'ensemble': False}
    clf_kwargs = {"n_jobs": args.lr_j}
    # baselines
    clf = LogisticRegression
    # clf = calibrated_svm
    if args.policy == "RS":
        policy = RelevancePolicy(clf, clf_args=[], clf_kwargs=clf_kwargs)
    else:
        policy = UncertaintyPolicy(clf, clf_args=[], clf_kwargs=clf_kwargs)

    stoppings = [
        cormack_knee.KneeStopping(target_recall=args.target_recall),
        cormack_knee.BudgetStopping(target_recall=args.target_recall),
    ]
    for t in args.target_recall:
        quant = lewis_yang.QuantStopping(target_recall=t)
        quant_1 = copy.deepcopy(quant)
        quant_1.nstd = 1.0
        quant_2 = copy.deepcopy(quant)
        quant_2.nstd = 2.0

        qbcb = lewis_yang.QBCB(target_recall=t)

        ipp = sneyd_stevenson.IPP(target_recall=t)

        adj_sld = SLDQuantStopping(
            target_recall=t, nstd=0.0, dataset_length=pool_size, use_margin=False
        )
        adj_sld_m = SLDQuantStopping(
            target_recall=t, nstd=0.0, dataset_length=pool_size, use_margin=True
        )
        adj_sld_1 = copy.deepcopy(adj_sld)
        adj_sld_1.nstd = 1
        adj_sld_2 = copy.deepcopy(adj_sld)
        adj_sld_2.nstd = 2

        chm = callaghan_chm.CHMStopping(target_recall=t, dataset_length=pool_size)
        stoppings.extend(
            (
                quant,
                quant_1,
                quant_2,
                qbcb,
                ipp,
                adj_sld,
                adj_sld_m,
                adj_sld_1,
                adj_sld_2,
                chm,
            )
        )

    print("Loading dataset...")
    dataset = fetch_rcv1()
    pool_idxs = np.random.choice(
        np.arange(dataset.data.shape[0]), replace=False, size=pool_size
    )

    x, y = dataset.data[pool_idxs], dataset.target[pool_idxs].toarray()
    classes = np.arange(len(dataset.target_names))
    # classes = np.where(np.logical_and(y.mean(0) >= 0.002, y.mean(0) <= 0.2))[0]
    if args.debugging:
        classes = np.random.choice(classes, replace=False, size=45)
        for cls in classes:
            y_c = y[:, cls]
            run_al(
                dataset.target_names[cls],
                pool_size,
                x,
                y_c,
                pool_size,
                policy,
                stoppings,
            )
        sys.exit(0)

    # For debugging
    # classes = np.random.choice(classes, replace=False, size=45)
    recorder = tmt_recorder(args.name, description=args.description)(process_futures)

    jobs = args.jobs if args.jobs else min(len(classes), 45)

    print(f"Running with {jobs} jobs")
    with ProcessPoolExecutor(max_workers=jobs) as p:
        futures = []
        for r in range(args.runs):
            for cls in classes:
                y_c = y[:, cls]
                topic_name = f"{dataset.target_names[cls]}_{r}"
                futures.append(
                    p.submit(
                        run_al,
                        topic_name,
                        pool_size,
                        copy.deepcopy(x),
                        copy.deepcopy(y_c),
                        pool_size,
                        copy.deepcopy(policy),
                        copy.deepcopy(stoppings),
                    )
                )

        recorder(futures, f"{args.name}_results")
    # for c in tqdm(classes):
    #     name, res = run_al(dataset.target_names[c], pool_size, copy.deepcopy(x), copy.deepcopy(y[:, c]), pool_size,
    #                        copy.deepcopy(policy), copy.deepcopy(stoppings))
    # sld_plot.name += f'; C={name}; CLF=SVM'
    # adj_sld_plot.name += f'; C={name}; CLF=SVM'
    # cknee_plot.name += f'; C={name}; CLF=SVM'
    # sld_plot.plot()
    # adj_sld_plot.plot()
    # cknee_plot.plot()
    #
    # sld_plot.flush('SLD')
    # adj_sld_plot.flush('Adj SLD')
    # cknee_plot.flush('CKnee')
