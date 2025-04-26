import click
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.expert import get_expert
from lib.model import (MLPWithEmbedding,
                       train)
from lib.optimization.extreme_ranking import solve_extreme_ranking
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes, remove_redundant_rules)
from lib.tools import set_seed

pd.set_option("display.precision", 16)


@click.command()
@click.argument("dataset_name", type=str, default="churn")
@click.argument("class_", type=int, default=1)
@click.argument("idx_", type=int, default=0)
@click.option("--retrain", "-r", is_flag=True, help="A flag indicating whether to retrain the model.")
@click.option("--verbose", "-v", is_flag=True, help="A flag indicating whether to print the logs.")
@click.option("--confirm_measure", type=str, default=None)
@click.option("--confirm_threshold", type=float, default=None)
@click.option("--expert_func", type=str, default=None)
def main(dataset_name: TDatasetName, class_: int, idx_: int, retrain: bool, verbose: bool, confirm_measure: str | None,
         confirm_threshold: float | None, expert_func: str | None) -> None:
    set_seed()
    conf = get_config(dataset_name)
    conf["rules"]["confirm_measure"] = confirm_measure or conf["rules"]["confirm_measure"]
    conf["rules"]["confirm_threshold"] = confirm_threshold or conf["rules"]["confirm_threshold"]
    conf["expert"]["func"] = expert_func or conf["expert"]["func"]
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

    idcs = get_expert(conf["expert"]["func"])(rules=sub_rules, rules_wrapped=sub_rules_wrapped, dataset=dataset,
                                              **conf["expert"])
    index_ = sub_rules_wrapped.iloc[idcs].index
    solution = solve_extreme_ranking(sub_rules_wrapped, idcs)
    submask = (solution["highest"] > solution["lowest"])
    solution["lowest"][submask] += 1.25
    solution["highest"][submask] -= 0.25
    solution["lowest"] -= 1
    solution["highest"] -= 1

    # get indices of the rules selected by the user in solution
    selected_indices = [i for i, idx in enumerate(sub_rules_wrapped.index) if idx in index_]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    n = len(solution)
    # Plot range as bars
    ax.barh(y=np.arange(n)[::-1] - 2,
            width=solution["highest"] - solution["lowest"],
            left=solution["lowest"],
            color=cmc.batlowS.colors[4], zorder=5)

    # Mark DM selected rules
    ax.scatter(np.zeros((len(selected_indices), 1)) - 1, [n - x - 3 for x in selected_indices], marker="*",
               color=cmc.batlowS.colors[0], zorder=5)

    ticks = np.arange(len(sub_rules_wrapped), step=3, dtype=int)
    # ticks = np.linspace(0, len(solution) - 1, 5, dtype=int)
    # Add labels
    ax.set_xlabel("Allowable position")
    ax.set_ylabel("Rule Index")
    ax.set_title("Extreme ranking")
    ax.set_xticks(ticks, ticks + 1)
    ax.set_yticks(ticks[::-1], ticks + 1)
    plt.xlim((-2, len(solution)))
    plt.grid(axis="both", color="lightgray", linestyle="-", linewidth=0.5, which="both", zorder=-5)

    plt.savefig(f"extreme_ranking_{dataset_name}_{class_}_{idx_}.png")

    print(solution)
    print(solution.iloc[selected_indices])


if __name__ == "__main__":
    main()
