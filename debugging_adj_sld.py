from run_experiments import *
from utils.plot_listener import DistribPlotListener
import numpy as np


if __name__ == "__main__":
    paper_name = ""
    parser = argparse.ArgumentParser(f"Run experiments for paper {paper_name}")
    parser.add_argument("-j", "--jobs", type=int, help="number of processes to spawn")
    parser.add_argument("-lrj", "--lr-j", type=int, help="number of jobs for the LR")
    # parser.add_argument('-b', '--batch', choices=[CormackBatch], default=CormackBatch,
    #                    help='Batch strategy to use. Only Cormack\'s available atm.')
    parser.add_argument(
        "-p",
        "--pool-size",
        type=int,
        help="size of the pool to annotate",
        default=10_000,
    )
    parser.add_argument("-s", "--seed", type=int, help="random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)
    pool_size = args.pool_size

    # clf_kwargs = {'n_jobs': args.lr_j , 'ensemble': False}
    clf_kwargs = {"n_jobs": args.lr_j}
    # baselines
    clf = LogisticRegression
    # clf = calibrated_svm
    policy = RelevancePolicy(clf, clf_args=[], clf_kwargs=clf_kwargs)
    adj_sld = SLDQuantStopping(target_recall=0.9, nstd=0.0, dataset_length=pool_size, alpha=0)
    listener = DistribPlotListener(iter_save={3, 5, 10, 20}, save_path="plots/", name="test")
    adj_sld.attach_listener(listener)
    stoppings = [adj_sld]

    print("Loading dataset...")
    dataset = fetch_rcv1()
    pool_idxs = np.random.choice(np.arange(dataset.data.shape[0]), replace=False, size=pool_size)

    x, y = dataset.data[pool_idxs], dataset.target[pool_idxs].toarray()
    # classes = np.arange(len(dataset.target_names))
    classes = np.where(np.logical_and(y.mean(0) >= 0.002, y.mean(0) <= 0.2))[0]

    # For debugging
    classes = np.random.choice(classes, replace=False, size=45)

    run_al(
        dataset.target_names[classes[1]],
        pool_size,
        copy.deepcopy(x),
        copy.deepcopy(y[:, classes[0]]),
        pool_size,
        copy.deepcopy(policy),
        copy.deepcopy(stoppings),
    )
