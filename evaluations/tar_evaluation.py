from tmt import TmtManager
from utils.metrics import (
    normalized_absolute_error,
    mean_square_error,
    relative_error,
    YangIdealizedCost,
)
from active_learning.batch_strategy import LinearStrategy
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def latex_table_it_recall(stoppings, target_rec=0.8):
    for k, v in res.items():
        s = v["stops"]
        for ss in stoppings:
            if str(ss) not in s:
                s[str(ss)] = {"it": 100, "recall": 1.0}
            s[str(ss)]["it"] = s[str(ss)]["it"] if type(s[str(ss)]["it"]) is int else 100
            s[str(ss)]["cost"] = cost.evaluate(
                LinearStrategy(b=100),
                v["idxs"][0],
                v["y_c"],
                s[str(ss)]["it"],
                target_rec,
            )
        s["class"] = k
        v["stops"] = s
    df = pd.DataFrame([v["stops"] for v in res.values()])
    buf = []
    order = [
        "Knee",
        f"QuantCI 0 @ {target_rec}",
        f"QuantCI 2.0 @ {target_rec}",
        f"SLDQuant 0.0 @ {target_rec}",
        f"SLDQuant 1 @ {target_rec}",
        f"SLDQuant 2 @ {target_rec}",
    ]
    avgs = {o: {"cost": 0, "recall": 0} for o in order}
    for _, row in df.iterrows():
        for k, v in avgs.items():
            v["cost"] += row[k]["cost"]
            v["recall"] += row[k]["recall"]
        s = " & ".join(f'{int(row[o]["cost"])} & {row[o]["recall"]:.3f}' for o in order)
        buf.append(f"\t{row['class']} & {s}")

    avgs = {k: {"cost": v["cost"] / len(res), "recall": v["recall"] / len(res)} for k, v in avgs.items()}
    with open("evaluations/it_recall_template.tex", "r") as f:
        temp = string.Template(f.read())
    return temp.substitute(
        class_results=" \\\\ \n".join(buf),
        avg_results=" & ".join(
            f'{round(avgs[o][k], 3) if k == "recall" else round(avgs[o][k], 0)}' for o in order for k in ["cost", "recall"]
        ),
    )


def compute_tar_metrics(res, cost, target_recall=0.9, max_it=100):
    d = {}
    classes = {}
    for cls_, vals in res.items():
        y_c = vals["y_c"]
        classes[cls_] = y_c.mean()
        tr_idxs = vals["idxs"][0]
        for method, r in vals["stops"].items():
            if "@" in method and f"@ {target_recall:.2f}" not in method:
                continue
            d.setdefault((method, "MSE"), []).append(mean_square_error(target_recall, r["recall"]))
            d.setdefault((method, "RE"), []).append(relative_error(target_recall, r["recall"]))
            it = r["it"] if r["it"] != np.inf else max_it
            pre_cost = 0
            if "qbcb" in method.lower():
                pre_sample = vals["QBCB@0.8 presample"]  # recall doesn't matter wrt recall
                assert pre_sample is not None, "QBCB presample cannot be none"
                pre_cost = cost.costs_of_annotated_docs(y_c[pre_sample])
            d.setdefault((method, "Cost"), []).append(cost.evaluate(LinearStrategy(b=100), tr_idxs, y_c, it, target_recall) + pre_cost)
    return pd.DataFrame(d).set_index(np.array(list(classes.keys()))), pd.Series(classes)


def compute_different_cost_structures(res, cost_structs, target_recall=0.9):
    d = {}
    classes = {}
    for cls_, vals in res.items():
        y_c = vals["y_c"]
        classes[cls_] = y_c.mean()
        tr_idxs = vals["idxs"][0]
        for method, r in vals["stops"].items():
            if "@" in method and f"@ {target_recall:.2f}" not in method:
                continue
            for cs in cost_structs:
                pre_cost = 0
                if "qbcb" in method.lower():
                    pre_sample = vals["QBCB@0.8 presample"]  # recall doesn't matter wrt recall
                    assert pre_sample is not None, "QBCB presample cannot be none"
                    pre_cost = cs.costs_of_annotated_docs(y_c[pre_sample])
                d.setdefault((method, cs.name), []).append(
                    cs.evaluate(LinearStrategy(b=100), tr_idxs, y_c, r["it"], target_recall) + pre_cost
                )
    return pd.DataFrame(d).set_index(np.array(list(classes.keys()))), pd.Series(classes)


def create_recall_df(res, tr):
    d = {}
    classes = {}
    for cls_, vals in res.items():
        y_c = vals["y_c"]
        classes[cls_] = y_c.mean()
        for method, r in vals["stops"].items():
            if "@" in method and f"@ {tr:.2f}" not in method:
                continue
            d.setdefault(method, []).append(r["recall"])
    return pd.DataFrame(d).set_index(np.array(list(classes.keys()))), pd.Series(classes)


def latex_results(df, clss, exclude, target_rec=0.8, exclude_costs=True, precision=3):
    def bold_best_it_second(col):
        l = [""] * len(col)
        s = col.argsort()
        l[s[0]] = "textbf:--rwrap"
        l[s[1]] = "underline:--rwrap"
        return l

    c = clss.groupby(lambda i: i.split("_")[0]).mean()
    d = df.groupby(lambda i: i.split("_")[0]).mean()
    new_d = {}
    cut = pd.qcut(c, q=3)
    all_d = pd.DataFrame(d.mean())
    d = d.groupby(cut).mean().set_index(np.array(["Low", "Medium", "High"])).T
    d["All"] = all_d
    d = d[["All", "Low", "Medium", "High"]].T
    for b, j in d.T.to_dict().items():
        for (m, e), v in j.items():
            if exclude.search(m) or (exclude_costs and e == "Cost"):
                continue
            sub_d = new_d.setdefault((b, e), {})
            sub_d[m] = v
    mapper = {
        f"SLDQuant 0.0 & a=0.3 @ {target_rec:.2f} w\\ m=0": r"\sal",
        f"SLDQuant 0.0 & a=0.3 @ {target_rec:.2f} w\\ m=1": r"\salr",
        f"SLDQuant 2 & a=0.3 @ {target_rec:.2f} w\\ m=0": r"\sal{}QuantCI",
        f"QuantCI 0 @ {target_rec:.2f}": "Quant",
        f"QuantCI 2.0 @ {target_rec:.2f}": "QuantCI",
        f"CHM @ {target_rec:.2f}": "CHM",
        f"QBCB @ {target_rec:.2f}": "QBCB",
        f"IPP @ {target_rec:.2f}": "IPP",
    }
    return (
        pd.DataFrame(new_d)
        .sort_index()
        .style.format(precision=precision)
        .format_index(lambda i: mapper.get(i, i))
        .apply(bold_best_it_second)
        .to_latex(hrules=True)
    )


def boxplots(plot_name, res, target_rec, legend_loc="lower right", d_name="RCV1-v2"):
    d, _ = create_recall_df(res, target_rec)
    d = d.groupby(lambda i: i.split("_")[0]).mean()

    mapper = {
        f"SLDQuant 0.0 & a=0.3 @ {target_rec:.2f} w\\ m=0": r"$\mathrm{SAL}_\tau$ (ours)",
        f"SLDQuant 0.0 & a=0.3 @ {target_rec:.2f} w\\ m=1": r"$\mathrm{SAL}_\tau^{R}$ (ours)",
        f"QuantCI 2.0 @ {target_rec:.2f}": "QuantCI",
        f"CHM @ {target_rec:.2f}": "CHM",
        f"QBCB @ {target_rec:.2f}": "QBCB",
        f"IPP @ {target_rec:.2f}": "IPP",
        "Knee": "Knee",
        "BudgetKnee": "Budget",
    }
    plt.figure(figsize=(12, 10))
    ax = sns.boxplot(data=d.rename(columns=mapper)[list(mapper.values())].sort_index(axis=1))
    xl = np.linspace(*ax.get_xlim(), len(ax.get_xticks()))
    ax.plot(xl, [target_rec] * len(xl), label="Target recall")
    plt.title(f"Estimated recall. Target recall = {target_rec}. Dataset {d_name}")
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(plot_name)


def proc(t):
    dm, cm = compute_tar_metrics(res, YangIdealizedCost(), t)
    dc, cc = compute_different_cost_structures(res, cost_structs, target_recall=t)
    r1 = latex_results(dm, cm, exclude=re.compile(r"Quant(CI)? 1"), target_rec=t)
    r2 = latex_results(dc, cc, exclude=re.compile(r"Quant(CI)? 1"), target_rec=t, precision=0)
    return t, r1, r2


if __name__ == "__main__":
    from run_experiments import *
    from multiprocessing import Pool
    import sys
    import re

    manager = TmtManager()
    manager.set_entry_by_name(sys.argv[1])
    res = {k: v for k, v in next(manager.load_results())[1].items() if "y_c" in v}
    cost_structs = [
        YangIdealizedCost(),
        YangIdealizedCost.with_expensive_training(),
        YangIdealizedCost.with_minecore_like(),
    ]

    target_recs = [0.80, 0.90, 0.95]
    with Pool(len(target_recs)) as p:
        w = []
        for t in target_recs:
            w.append(p.apply_async(proc, (t,)))
        for wi in w:
            t, r1, r2 = wi.get()
            print("#" * 10 + f" R = {t} " + "#" * 10)
            print(r1, r2, sep="\n")

    # Multiindex dict can be like
    # {(Method1, metric1): [1,2,3], (Method1, metric2): [1,2,3] ... }
    # n_runs = 20
    # target_rec = .8
    # name = 'sr_exp'
    # dfs, clss = [], []
    # for i in range(1, n_runs+1):
    #     manager = TmtManager()
    #     manager.set_entry_by_name(f'{name}{i}')
    #     res = next(manager.load_results())[1]
    #     d, c = compute_tar_metrics(res, YangIdealizedCost(), target_rec)
    #     dfs.append(d)
    #     clss.append(c)
    #
    # clss = pd.concat(clss).groupby(level=0).mean()  # <- get classes average prevalence (unique)
    # dfs = pd.concat(dfs).groupby(level=0).mean()
    # cut = pd.qcut(clss, q=4)
    # d = dfs.groupby(cut).mean().set_index(np.array(['Low', 'Medium-Low', 'Medium', 'High']))
    # new_d = {}
    # for b, j in d.T.to_dict().items():
    #     for (m, e), v in j.items():
    #         sub_d = new_d.setdefault((b, e), {})
    #         sub_d[m] = v
    # print(pd.DataFrame(new_d).sort_index().style.format(precision=3).
    #       format_index(lambda i: i.removesuffix(f' @ {target_rec}')).
    #       highlight_min(props='textbf:--rwrap').to_latex(hrules=True))
    # cknee = cormack_knee.KneeStopping(target_recall=target_rec)
    # quant = quantci.QuantStopping(target_recall=target_rec)
    # quant_1 = copy.deepcopy(quant)
    # quant_1.nstd = 1.
    # quant_2 = copy.deepcopy(quant)
    # quant_2.nstd = 2.

    # adj_sld = SLDQuantStopping(target_recall=target_rec, nstd=0., dataset_length=10_000)
    # adj_sld_1 = copy.deepcopy(adj_sld)
    # adj_sld_1.nstd = 1
    # adj_sld_2 = copy.deepcopy(adj_sld)
    # adj_sld_2.nstd = 2

    # stoppings = [
    #    cknee,
    #    quant,
    #    quant_2,
    #    adj_sld,
    #    adj_sld_1,
    #    adj_sld_2,
    # ]
    # manager = TmtManager()
    # manager.set_entry_by_name('target_rec_8090100_fix_5')
    # res = next(manager.load_results())[1]
    # ms = []

    # class_df = pd.DataFrame({k: v['y_c'] for k, v in res.items()}).mean().astype(float)
    # cut = pd.qcut(class_df, q=4)
    # cost = YangIdealizedCost()
    # print(latex_table_it_recall(stoppings, target_rec))
