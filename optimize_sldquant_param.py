from run_experiments import *
from dataset_elaboration.jeb_bush import load_dataset


def run_hypersearch(name, pool_size, x, y_c, policy, stoppings):
    res = {"y_c": y_c}
    random_pos = np.random.choice(np.where(y_c == 1)[0], size=1)
    random = np.random.choice(np.where(y_c == 0)[0], replace=False, size=1)
    conf = ActiveLearningConfig(
        policy,
        stoppings,
        LinearStrategy(b=100),
        x,
        y_c,
        np.concatenate((random_pos, random)),
        stop_when_no_pos=True,
    )
    al = ActiveLearning(conf)
    res["idxs"] = al.run(pool_size, pool_size)
    res["stops"] = al.get_stops_as_dict()
    return name, res


if __name__ == "__main__":
    paper_name = ""
    parser = argparse.ArgumentParser(f"Run param. optimization for SLD Quant")
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
    parser.add_argument(
        "-p",
        "--pool-size",
        type=int,
        help="size of the pool to annotate",
        default=10_000,
    )
    parser.add_argument("-n", "--name", help="tmt save name", required=True)
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument("-a", "--alpha", type=float, nargs="+", help="alpha values for SLD to test")

    args = parser.parse_args()
    np.random.seed(args.seed)
    pool_size = args.pool_size
    clf_kwargs = {"n_jobs": args.lr_j}
    # baselines
    clf = LogisticRegression
    # clf = calibrated_svm
    policy = RelevancePolicy(clf, clf_args=[], clf_kwargs=clf_kwargs)
    stoppings = []
    for t in args.target_recall:
        for alpha in args.alpha:
            stoppings.append(SLDQuantStopping(nstd=0.0, target_recall=t, dataset_length=pool_size, alpha=alpha))

    print("Loading dataset...")
    dataset = load_dataset()
    recorder = tmt_recorder(args.name)(process_futures)

    jobs = 30

    print(f"Running with {jobs} jobs")
    with ProcessPoolExecutor(max_workers=jobs) as p:
        futures = []
        for topic, x, y in dataset:
            stoppings = []
            for t in args.target_recall:
                for alpha in args.alpha:
                    stoppings.append(
                        SLDQuantStopping(
                            nstd=0.0,
                            target_recall=t,
                            dataset_length=len(y),
                            alpha=alpha,
                        )
                    )
            futures.append(
                p.submit(
                    run_al,
                    topic,
                    len(y),
                    x,
                    y,
                    len(y),
                    copy.deepcopy(policy),
                    copy.deepcopy(stoppings),
                )
            )

        recorder(futures, args.name)
