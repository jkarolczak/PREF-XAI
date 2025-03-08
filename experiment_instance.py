import json
from pathlib import Path

import click
import pandas as pd
import torch

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.expert import get_expert
from lib.model import (MLPWithEmbedding,
                       train)
from lib.optimization import solve
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes)
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
    if confirm_measure:
        conf["rules"]["confirm_measure"] = confirm_measure
    if confirm_threshold:
        conf["rules"]["confirm_threshold"] = confirm_threshold
    if expert_func:
        conf["expert"]["func"] = expert_func
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
    idx_i = dataset.test.y.index.get_loc(idx)
    sub_rules = get_rules_covering_instance(rules, dataset.test.x.loc[idx], cls=dataset.test.y.loc[idx])
    sub_rules_wrapped = wrap_rules_to_attributes(sub_rules)

    idcs = get_expert(conf["expert"]["func"])(rules=sub_rules, rules_wrapped=sub_rules_wrapped, dataset=dataset,
                                              **conf["expert"])

    solution = solve(sub_rules_wrapped, idcs)

    sub_ranking = solution.create_ranking(sub_rules_wrapped.iloc[idcs])
    full_ranking = solution.create_ranking(sub_rules_wrapped)

    directory = Path(f"reports/instance/{conf['expert']['func']}")
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{dataset_name}_{conf['rules']['confirm_measure']}_{class_}_{idx_}.txt"
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

        fp.write("Rules statistics:\n")
        fp.write(f" - Rules of class 0: {rules[rules['y_0'] == 1].shape[0]}\n")
        fp.write(f" - Rules of class 1: {rules[rules['y_1'] == 1].shape[0]}\n")
        fp.write(f" - Rules covering the instance: {sub_rules_wrapped.shape[0]}\n\n")

        fp.write("Expert ranking:\n")
        fp.write(f" - Ranking of the attributes: {dataset.train.x.columns[conf['expert']['columns_preferences']].to_list()}\n")
        fp.write(f"{sub_rules_wrapped.iloc[idcs]}\n\n")

        fp.write("Trust:\n")
        fp.write(f"{solution}\n\n")

        fp.write("Trust expert's preferences re-ranking:\n")
        fp.write(sub_ranking.to_string())

        fp.write("\n\nTrust all covering rules re-ranking:\n")
        fp.write(full_ranking.to_string())


if __name__ == "__main__":
    main()
