from pathlib import Path
from typing import Literal

import click
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.experts import get_expert
from lib.model import (MLPWithEmbedding,
                       train)
from lib.optimization.hit_and_run import write_initial, aggregate_results
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes,
                             remove_redundant_rules)
from lib.tools import set_seed

pd.set_option("display.precision", 16)


@click.command()
@click.argument("dataset_name", type=str, default="churn")
@click.argument("class_", type=int, default=1)
@click.argument("idx_", type=int, default=0)
@click.argument("mode", type=click.Choice(["init", "results"]), default="init")
@click.option("--retrain", "-r", is_flag=True, help="A flag indicating whether to retrain the model.")
@click.option("--verbose", "-v", is_flag=True, help="A flag indicating whether to print the logs.")
@click.option("--confirm_measure", type=str, default=None)
@click.option("--confirm_threshold", type=float, default=None)
@click.option("--expert_func", type=str, default=None)
@click.option("--expert_rules", type=int, default=None)
@click.option("--exclude", type=int, default=0)
def main(dataset_name: TDatasetName, class_: int, idx_: int, mode: Literal["init", "results"], retrain: bool, verbose: bool,
         confirm_measure: str | None, confirm_threshold: float | None, expert_func: str | None,
         expert_rules: int | None, exclude: int) -> None:
    set_seed()
    conf = get_config(dataset_name)
    if confirm_measure:
        conf["rules"]["confirm_measure"] = confirm_measure
    if confirm_threshold:
        conf["rules"]["confirm_threshold"] = confirm_threshold
    if expert_func:
        conf["expert"]["func"] = expert_func
    if expert_rules:
        conf["expert"]["n_rules"] = expert_rules
    if exclude:
        excluded_preferences = conf["expert"]["columns_preferences"][-exclude:]
        conf["expert"]["columns_preferences"] = conf["expert"]["columns_preferences"][:-exclude]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load(dataset_name)

    model = MLPWithEmbedding(
        input_dim=dataset.train.x.shape[1],
        cat_idxs=dataset.cat_idxs,
        cat_dims=dataset.cat_dims,
        **conf.model
    ).to(device=device)

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
    sub_rules = get_rules_covering_instance(rules, dataset.test.x.loc[idx], cls=dataset.test.y.loc[idx])
    sub_rules_wrapped = wrap_rules_to_attributes(sub_rules)
    sub_rules_wrapped = remove_redundant_rules(sub_rules_wrapped)
    idcs = get_expert(conf["expert"]["func"])(rules=sub_rules_wrapped, rules_wrapped=sub_rules_wrapped, dataset=dataset,
                                              **conf["expert"])
    if exclude:
        exclude_cols_template = [col for idx, col in enumerate(sub_rules_wrapped.columns) if idx in excluded_preferences]
        sub_rules_wrapped = sub_rules_wrapped.drop(columns=exclude_cols_template, errors="ignore")

    path_base = f"outputs/{dataset_name}_{class_}_{idx_}"
    if exclude:
        path_base += f"_{exclude}"
    path_trust = Path(f"{path_base}_trust.txt")
    path_results = Path(f"{path_base}_results.txt")

    match mode:
        case "init":
            write_initial(sub_rules_wrapped, idcs, path_trust)
            Path(path_results).touch(exist_ok=True)
        case "results":
            occurrences = aggregate_results(sub_rules_wrapped, idcs, path_results)

            xticks = np.arange(len(sub_rules_wrapped), step=3, dtype=int)
            xlabels = xticks + 1
            yticks = [i + 1 for i in range(len(idcs))]

            plt.figure(figsize=(8, 6), dpi=300)

            plt.title("Hit and Run (1000 iterations)")
            plt.xlabel("Observed position in the ranking of all matching rules ($\\mathcal{S}$)")
            plt.ylabel("Position of reference rules ranked by the DM ($\\mathcal{T}$)")

            plt.grid(axis="both", color="lightgray", linestyle="-", linewidth=0.5, which="both", zorder=-5)
            plt.xlim((-2, len(sub_rules_wrapped)))
            plt.xticks(xticks, xlabels)
            plt.yticks(yticks[::-1], yticks)

            violins = plt.violinplot(occurrences[::-1],
                                     orientation="horizontal",
                                     widths=1.5,
                                     side="high")

            for partname in ("cbars", "cmins", "cmaxes"):
                vp = violins[partname]
                vp.set_edgecolor(cmc.batlowS.colors[0])

            for pc in violins["bodies"]:
                pc.set_facecolor(cmc.batlowS.colors[4])
                pc.set_alpha(1)
                pc.set_zorder(2)

            plt.savefig(f"hit_and_run_{dataset_name}_{class_}_{idx_}_{exclude}.png")


if __name__ == "__main__":
    main()
