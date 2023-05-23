from run_experiments import *
import pickle


if __name__ == "__main__":
    import os

    paper_name = ""
    parser = argparse.ArgumentParser(
        f"Run experiments for paper {paper_name}. CLEF 2019 dataset."
    )
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
    parser.add_argument(
        "-r", "--runs", type=int, default=20, help="number of random runs"
    )

    args = parser.parse_args()
    sr_path_tr = ".data/sr/2019_only/training"
    sr_path_te = ".data/sr/2019_only/testing"

    clf = LogisticRegression
    clf_kwargs = {"n_jobs": args.lr_j}
    # clf = calibrated_svm
    policy = RelevancePolicy(clf, clf_args=[], clf_kwargs=clf_kwargs)

    topics = list(set(os.listdir(sr_path_tr) + os.listdir(sr_path_te)))
    jobs = args.jobs if args.jobs else max(len(topics), 45)
    recorder = tmt_recorder(args.name)(process_futures)
    print(f"Running with {jobs} jobs and {args.lr_j} jobs for LR")
    print("Loading dataset...")
    with ProcessPoolExecutor(max_workers=jobs) as p:
        futures = []
        for i in range(args.runs):
            for topic in topics:
                # TODO this topic has more than 70k documents, we process it later with all CPU power
                if topic == "vectorized_tfidf_CD009263.pkl":
                    continue
                if not os.path.exists(path := os.path.join(sr_path_tr, topic)):
                    path = os.path.join(sr_path_te, topic)
                with open(path, "rb") as f:
                    x, y = pickle.load(f)
                stoppings = [
                    cormack_knee.KneeStopping(
                        target_recall=args.target_recall, min_rounds=0
                    ),
                    cormack_knee.BudgetStopping(
                        target_recall=args.target_recall, min_rounds=0
                    ),
                ]
                for t in args.target_recall:
                    quant = lewis_young.QuantStopping(target_recall=t, min_rounds=0)
                    quant_1 = copy.deepcopy(quant)
                    quant_1.nstd = 1.0
                    quant_2 = copy.deepcopy(quant)
                    quant_2.nstd = 2.0

                    adj_sld = SLDQuantStopping(
                        target_recall=t, nstd=0.0, dataset_length=len(y), min_rounds=0
                    )
                    adj_sld_m = SLDQuantStopping(
                        target_recall=t,
                        nstd=0.0,
                        dataset_length=len(y),
                        use_margin=True,
                    )
                    adj_sld_1 = copy.deepcopy(adj_sld)
                    adj_sld_1.nstd = 1
                    adj_sld_2 = copy.deepcopy(adj_sld)
                    adj_sld_2.nstd = 2

                    chm = callaghan_chm.CHMStopping(
                        target_recall=t, dataset_length=len(y), min_rounds=0
                    )
                    stoppings.extend(
                        (
                            quant,
                            quant_1,
                            quant_2,
                            adj_sld,
                            adj_sld_m,
                            adj_sld_1,
                            adj_sld_2,
                            chm,
                        )
                    )
                topic_name = f"{topic.split('_')[-1].split('.')[0]}_{i}"
                futures.append(
                    p.submit(
                        run_al,
                        topic_name,
                        len(y),
                        x,
                        y,
                        len(y),
                        copy.deepcopy(policy),
                        stoppings,
                    )
                )
        recorder(futures, f"{args.name}_results")
