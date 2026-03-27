import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from tqdm import tqdm

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.dms import get_dm
from lib.model import (MLPWithEmbedding,
                       train)
from lib.optimization.epsilon import solve_epsilon
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes, remove_redundant_rules)
from lib.tools import set_seed

pd.set_option("display.precision", 16)


@click.command()
@click.argument("dataset_name", type=str, default="churn")
@click.option("--retrain", "-r", is_flag=True, help="A flag indicating whether to retrain the model.")
@click.option("--verbose", "-v", is_flag=True, help="A flag indicating whether to print the logs.")
def main(dataset_name: TDatasetName, retrain: bool, verbose: bool) -> None:
    set_seed()
    conf = get_config(dataset_name)
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

    new_pos = {0: {i: [] for i in range(10)},
               1: {i: [] for i in range(10)}}
    sub_len = {0: [], 1: []}
    kendall = {0: [], 1: []}

    for class_ in tqdm([0, 1], desc="Class"):
        for idx_ in tqdm(range(90), desc="Instance"):
            idx = dataset.test.y[y_hat_test == class_].index[idx_]
            sub_rules = get_rules_covering_instance(rules, dataset.test.x.loc[idx], cls=dataset.test.y.loc[idx])
            sub_rules = remove_redundant_rules(sub_rules)
            print(sub_rules.shape)
            if len(sub_rules) <= 10:
                continue

            sub_rules_wrapped = wrap_rules_to_attributes(sub_rules)
            sub_len[class_].append(sub_rules.shape[0])

            dm_cls = get_dm(conf["expert"]["type"])
            dm = dm_cls(rules=sub_rules_wrapped, **conf["expert"])

            idcs = dm.reference_ranking()

            prus_epsilon = solve_epsilon(sub_rules_wrapped, idcs)

            # kendall distance between indices and prus_epsilon weights
            weights = prus_epsilon.w_j
            weights_ranking = np.argsort(weights)[::-1]

            kendall[class_].append(
                kendalltau(
                    weights_ranking,
                    conf["expert"]["columns_preferences"]
                )
            )

            if np.any(prus_epsilon.to_np()[-3:] != 0):
                print(dataset_name, class_, idx_)
                print(prus_epsilon)
            sub_ranking = prus_epsilon.create_ranking(sub_rules_wrapped.iloc[idcs])
            full_ranking = prus_epsilon.create_ranking(sub_rules_wrapped)

            try:
                assert all([r1 == r2 for r1, r2 in zip(sub_rules_wrapped.iloc[idcs].index, sub_ranking.index)])
            except Exception as e:
                continue

            for i, position in enumerate(sub_ranking.index):
                new_pos[class_][i].append(full_ranking.index.get_loc(position))

    quantiles = {cls: {q: [] for q in [0, 25, 50, 75, 100]} for cls in [0, 1]}
    for cls in [0, 1]:
        for i, positions in new_pos[cls].items():
            for q in (0, 25, 50, 75, 100):
                try:
                    quantiles[cls][q].append(np.percentile(positions, q))
                except:
                    pass
    for cls in [0, 1]:
        for q in (0, 25, 50, 75, 100):
            if q in quantiles[cls]:
                quantiles[cls][q] = sorted(quantiles[cls][q])

    directory = Path(f"reports/mass/{conf['expert']['type']}")
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{dataset_name}_{conf['rules']['confirm_measure']}.txt"
    with file.open("w") as fp:
        fp.write("Config:\n")
        fp.write(f"{json.dumps(conf, indent=4)}\n\n")

        fp.write("Rules statistics:\n")
        fp.write(f" - Rules of class 0: {rules[rules['y_0'] == 1].shape[0]}\n")
        fp.write(f" - Rules of class 1: {rules[rules['y_1'] == 1].shape[0]}\n")
        fp.write(f" - Number of rules covering instances of class 0: {np.mean(sub_len[0]).item():.4f} "
                 f"(+/- {np.std(sub_len[0]).item():.4f})\n")
        fp.write(f" - Number of rules covering instances of class 1: {np.mean(sub_len[1]).item():.4f} "
                 f"(+/- {np.std(sub_len[1]).item():.4f})\n\n")

        fp.write(
            f"Kendall's tau between prus_epsilon weights and user preferences for class 0: {np.mean(kendall[0]).item():.4f}\n")
        fp.write(
            f"Kendall's tau between prus_epsilon weights and user preferences for class 1: {np.mean(kendall[1]).item():.4f}\n\n")

        fp.write("Positions of the rules selected by the expert in the final ranking:\n")
        for cls in [0, 1]:
            fp.write(f" - Class {cls}: \n")
            for i, positions in new_pos[cls].items():
                fp.write(f"   - Rule {i}: ")
                for q in (0, 25, 50, 75, 100):
                    if q in quantiles[cls]:
                        fp.write(f"q{q}: {quantiles[cls][q][i] + 1:3.2f} ")
                fp.write("\n")


if __name__ == "__main__":
    main()
