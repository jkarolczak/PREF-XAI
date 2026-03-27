import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from cycler import cycler
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.dms import get_dm
from lib.metrics import set_jaccard
from lib.model import (MLPWithEmbedding,
                       train)
from lib.optimization.epsilon import solve_epsilon
from lib.optimization.hit_and_run import solve_hr, pruss_to_centroid, rules_on_first_pos
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes,
                             remove_redundant_rules)
from lib.tools import set_seed

pd.set_option("display.precision", 16)


def set_plt_properties() -> None:
    font_size = 8
    colour = "black"
    line_width = 0.5
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
        "figure.titlesize": font_size,

        "boxplot.boxprops.color": colour,
        "boxplot.boxprops.linewidth": line_width,
        "boxplot.capprops.color": colour,
        "boxplot.capprops.linewidth": line_width,
        "boxplot.whiskerprops.color": colour,
        "boxplot.whiskerprops.linewidth": line_width,
        "boxplot.medianprops.color": colour,
        "boxplot.medianprops.linewidth": 3 * line_width,
        "boxplot.flierprops.markeredgecolor": colour,

        "lines.linewidth": line_width,
        "axes.prop_cycle": cycler(color=[colour]),

        "xtick.major.width": line_width,
        "ytick.major.width": line_width,
    })


@click.command()
@click.argument("dataset_name", type=str, default="churn_banking")
@click.argument("class_", type=int, default=1)
@click.argument("idx_", type=int, default=0)
@click.option("--retrain", "-r", is_flag=True, help="A flag indicating whether to retrain the model.")
@click.option("--verbose", "-v", is_flag=True, help="A flag indicating whether to print the logs.")
@click.option("--confirm_measure", type=str, default=None)
@click.option("--confirm_threshold", type=float, default=None)
@click.option("--expert_func", type=str, default=None)
@click.option("--expert_rules", type=int, default=None)
@click.option("--exclude", type=int, default=0)
@click.option("--hr_iters", type=int, default=10000)
def main(dataset_name: TDatasetName, class_: int, idx_: int, retrain: bool, verbose: bool, confirm_measure: str | None,
         confirm_threshold: float | None, expert_func: str | None, expert_rules: int | None, exclude: int,
         hr_iters: int) -> None:
    set_seed()
    set_plt_properties()
    cm = 1 / 2.54

    conf = get_config(dataset_name)
    if confirm_measure:
        conf.rules.confirm_measure = confirm_measure
    if confirm_threshold:
        conf.rules.confirm_threshold = confirm_threshold
    if expert_func:
        conf.expert.func = expert_func
    if expert_rules:
        conf.expert.n_rules = expert_rules
    if exclude:
        excluded_preferences = conf.expert.columns_preferences[-exclude:]
        conf.expert.columns_preferences = conf.expert.columns_preferences[:-exclude]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load(dataset_name)

    model = MLPWithEmbedding(input_dim=dataset.train.x.shape[1], cat_idxs=dataset.cat_idxs, cat_dims=dataset.cat_dims,
                             **conf.model).to(device=device)

    if retrain:
        train(model=model, dataset=dataset, device=device, verbose=verbose, **conf.train)
        torch.save(model.state_dict(), f"models/{dataset_name}.pth")
    else:
        model.load_state_dict(torch.load(f"models/{dataset_name}.pth", weights_only=True))

    proba_train = model(dataset.train.as_torch(device=device)[0])[:, 1].detach().cpu().numpy()
    y_hat_train = (proba_train > conf.train.threshold).astype(int)

    rules = generate_rules(dataset, y_hat_train, **conf.rules)
    proba_test = model(dataset.test.as_torch(device=device)[0])[:, 1].detach().cpu().numpy()
    y_hat_test = (proba_test > conf.train.threshold).astype(int)
    idx = dataset.test.y[y_hat_test == class_].index[idx_]
    idx_i = dataset.test.y.index.get_loc(idx)
    sub_rules = get_rules_covering_instance(rules, dataset.test.x.loc[idx], cls=dataset.test.y.loc[idx])
    sub_rules = remove_redundant_rules(sub_rules)
    sub_rules_wrapped = wrap_rules_to_attributes(sub_rules)
    dm = get_dm(conf.expert.type)(rules=sub_rules_wrapped, **conf.expert)
    reference_ranking = dm.reference_ranking
    full_reference_ranking = dm.create_ranking(sub_rules_wrapped)
    if exclude:
        exclude_cols_template = [col for idx, col in enumerate(sub_rules_wrapped.columns) if idx in excluded_preferences]
        sub_rules_wrapped = sub_rules_wrapped.drop(columns=exclude_cols_template, errors="ignore")

    pruss_hr = solve_hr(sub_rules_wrapped, dm, n_iters=hr_iters)
    occurrences = [[] for _ in reference_ranking]
    for prus in pruss_hr:
        ranking = prus.create_ranking(sub_rules_wrapped)
        for j, i in enumerate(reference_ranking):
            pos = ranking.index.get_loc(i)
            occurrences[j].append(pos)

    xticks = np.arange(len(sub_rules_wrapped), step=3, dtype=int)
    xlabels = xticks + 1
    yticks = [i + 1 for i in range(len(reference_ranking))]

    plt.figure(figsize=(14 * cm, 5.5 * cm), dpi=1000)

    plt.xlabel("Possible Rank Observed in Hit-and-Run")
    plt.ylabel("Rank in Decision Maker Reference", y=0.40)

    plt.grid(axis="both", color="lightgray", linestyle="-", linewidth=0.5, which="both", zorder=-5)
    plt.xlim((-1, len(sub_rules_wrapped)))
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks[::-1], yticks)

    violins = plt.violinplot(occurrences[::-1], orientation="horizontal", widths=1.5, side="high")

    for partname in ("cbars", "cmins", "cmaxes"):
        vp = violins[partname]
        vp.set_edgecolor("black")

    for pc in violins["bodies"]:
        pc.set_facecolor("grey")
        pc.set_alpha(1)
        pc.set_zorder(2)
    plt.tight_layout()
    fig_file = f"outputs/figures/h&r_{dataset_name}_{class_}_{idx_}"
    if exclude:
        fig_file += f"_exclude_{exclude}"
    plt.savefig(f"{fig_file}.pdf")

    prus_epsilon = solve_epsilon(sub_rules_wrapped, dm)
    prus_hr_centroid = pruss_to_centroid(pruss_hr)
    hr_rules_on_first_pos = rules_on_first_pos(pruss_hr, sub_rules_wrapped)

    directory = Path(f"outputs/reports/instance/{conf.expert.type}")
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{dataset_name}_{conf.rules.confirm_measure}_{class_}_{idx_}.txt"
    with file.open("w") as fp:
        fp.write("Config (conf):\n")
        fp.write(f"{json.dumps(conf, indent=4)}\n\n")

        fp.write("Instance selected by the user:\n")
        fp.write(f"Class: {class_}\n")
        fp.write(f"Index: {idx_}\n")
        fp.write(f"{dataset.test.x.loc[idx]}\n\n")
        fp.write(f"Model output: {proba_test[idx_i]:.8f}\n")
        fp.write(f"Model class prediction: {y_hat_test[idx_i]}\n")
        fp.write(f"Ground truth class: {dataset.test.y.loc[idx]}\n\n")

        fp.write("Expert ranking:\n")
        fp.write(f" - Ranking of the attributes: {dataset.train.x.columns[conf.expert.columns_preferences].to_list()}\n")
        fp.write(f"{sub_rules_wrapped.iloc[reference_ranking]}\n\n")

        for solution, method in zip([prus_epsilon, prus_hr_centroid], ["Epsilon", "Hit-and-Run Centroid"]):
            fp.write("\n\n\n========================================\n")
            fp.write(f"{method}:\n")
            fp.write("=========================================\n")

            sub_ranking = solution.create_ranking(sub_rules_wrapped.iloc[reference_ranking])
            full_ranking = solution.create_ranking(sub_rules_wrapped)

            fp.write("Prus:\n")
            fp.write(f"{solution}\n\n")

            fp.write("Prus expert's preferences re-ranking:\n")
            fp.write(sub_ranking.to_string())

            fp.write("\n\nPrus all covering rules re-ranking:\n")
            fp.write(full_ranking.to_string())

            kendall_ranking = kendalltau(full_ranking.index, full_reference_ranking.index).statistic.item()
            kendall_weights = kendalltau(dm.prus.to_np(), solution.to_np()).statistic.item()
            jaccard_5 = set_jaccard(sub_ranking.index[:5], reference_ranking[:5])
            jaccard_10 = set_jaccard(sub_ranking.index[:10], reference_ranking[:10])
            fp.write("\n\nMetrics:\n")
            fp.write(f" - Jaccard 5: {jaccard_5:.4f}\n")
            fp.write(f" - Jaccard 10: {jaccard_10:.4f}\n")
            fp.write(f" - Kendall Tau Ranking: {kendall_ranking:.4f}\n")
            fp.write(f" - Kendall Tau Weights: {kendall_weights:.4f}\n")

            human_readable = sub_rules.loc[full_ranking.index].drop(columns=["support", "confidence"])
            human_readable["prus"] = full_ranking["prus"]
            human_readable = human_readable.loc[:, (human_readable != 0).any(axis=0)]
            names = []
            for col in human_readable.columns:
                if "_" in col:
                    pre, val = col.split("_")
                    if "-" in val:
                        left, right = val.split("-")
                        names.append(f"{pre}={float(left):.4f}-{float(right):.4f}")
                    else:
                        names.append(f"{pre}={val}")
                else:
                    names.append(col)
            human_readable = human_readable.replace(True, pd.Series(names, human_readable.columns)).replace(False, "")
            human_readable = human_readable.replace(r"\.0000", "", regex=True)

            fp.write("\n\nHuman readable ranking:\n")
            fp.write(human_readable.to_string())

        fp.write("\n\n\n========================================\n")
        fp.write("Hit-and-Run First Rules:\n")
        fp.write("=========================================\n")

        jaccard_5 = set_jaccard(hr_rules_on_first_pos, full_reference_ranking.index[:5])
        jaccard_10 = set_jaccard(hr_rules_on_first_pos, full_reference_ranking.index[:10])
        fp.write("\n\nMetrics:\n")
        fp.write(f" - Jaccard 5: {jaccard_5:.4f}\n")
        fp.write(f" - Jaccard 10: {jaccard_10:.4f}\n")

        fp.write(sub_rules_wrapped.iloc[list(hr_rules_on_first_pos)].to_string())


if __name__ == "__main__":
    main()
