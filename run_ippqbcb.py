from run_experiments import *
from utils.data_utils import PreviousRunUtils
from active_learning.base_policy import PreviousRunPolicy


def run_al(name, budget, y_c, pool_size, policy, stoppings):
    res = {}
    if y_c.sum() < 2:
        return name, {}
    conf = ActiveLearningConfig(
        policy,
        stoppings,
        LinearStrategy(b=100),
        np.arange(len(y_c)),
        y_c,
        np.array([]),
        stop_when_no_pos=False,
    )
    al = ActiveLearning(conf)
    al.run(budget, pool_size)
    res["stops"] = al.get_stops_as_dict()
    return name, res


if __name__ == "__main__":
    paper_name = "SALt"
    parser = argparse.ArgumentParser(f"Run experiments for paper {paper_name}")
    parser.add_argument("-j", "--jobs", type=int, help="number of processes to spawn")
    parser.add_argument(
        "-t",
        "--target-recall",
        nargs="+",
        type=float,
        help="target recall TAR should stop at",
        required=True,
    )
    parser.add_argument("-d", "--description", help="experiment description for tmt")
    parser.add_argument("-n", "--name", help="tmt save name", required=True)
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument("--previous", required=True, help="if --policy previous, specify here tmt experiment name to load data from")
    parser.add_argument(
        "--debugging",
        action="store_true",
        help="if specified, the number of classes will be reduced,"
        "tmt and multiprocessing will not be used. Option"
        "--runs will also be ignored",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    stoppings = []
    for t in args.target_recall:
        qbcb = lewis_yang.QBCB(target_recall=t)
        ipp = sneyd_stevenson.IPP(target_recall=t)
        stoppings.extend(
            (
                qbcb,
                ipp,
            )
        )

    print("Loading dataset...")
    dataset = fetch_rcv1()
    classes = np.arange(len(dataset.target_names))
    prev_util = PreviousRunUtils(args.previous)

    if args.debugging:
        for cls_, idxs, y_c in prev_util.get_idxs_iter():
            if not idxs:
                continue
            idxs = idxs[0]
            prev_policy = PreviousRunPolicy(annotated_idxs=idxs)
            sample, possample = prev_util.get_qbcb_sample(cls_)
            for qb in filter(lambda s: type(s) is lewis_yang.QBCB, stoppings):
                qb.pre_sample = sample
                qb.pre_positives = possample
            run_al(cls_, len(y_c), copy.deepcopy(y_c), len(y_c), copy.deepcopy(prev_policy), copy.deepcopy(stoppings))

        sys.exit(0)

    # For debugging
    # classes = np.random.choice(classes, replace=False, size=45)
    recorder = tmt_recorder(args.name, description=args.description)(process_futures)

    jobs = args.jobs if args.jobs else min(len(classes), 45)

    print(f"Running with {jobs} jobs")
    with ProcessPoolExecutor(max_workers=jobs) as p:
        futures = []
        for cls_, idxs, y_c in prev_util.get_idxs_iter():
            if not idxs:
                continue
            idxs = idxs[0]
            prev_policy = PreviousRunPolicy(annotated_idxs=idxs)
            sample, possample = prev_util.get_qbcb_sample(cls_)
            for qb in filter(lambda s: type(s) is lewis_yang.QBCB, stoppings):
                qb.pre_sample = sample
                qb.pre_positives = possample
            futures.append(
                p.submit(run_al, cls_, len(y_c), copy.deepcopy(y_c), len(y_c), copy.deepcopy(prev_policy), copy.deepcopy(stoppings))
            )

        recorder(futures, f"{args.name}_results")
