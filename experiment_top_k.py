import json
from pathlib import Path
from typing import Optional, List

import click
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from tqdm import tqdm

from lib.config import get_config
from lib.data import (load, TDatasetName)
from lib.dms import get_dm
from lib.metrics import set_jaccard
from lib.model import MLPWithEmbedding
from lib.optimization.epsilon import solve_epsilon
from lib.optimization.hit_and_run import (solve_hr,
                                          trusts_to_centroid, rules_on_first_pos)
from lib.rules import generate_rules
from lib.rules.utils import (get_rules_covering_instance,
                             wrap_rules_to_attributes,
                             remove_redundant_rules)
from lib.tools import set_seed

pd.set_option("display.precision", 16)


def process_single_instance(
        idx_: int,
        class_: int,
        x_loc: pd.Series,
        y_loc: int,
        rules: pd.DataFrame,
        conf: dict
) -> Optional[List[dict]]:
    sub_rules = get_rules_covering_instance(rules, x_loc, cls=y_loc)

    sub_rules = wrap_rules_to_attributes(sub_rules)
    sub_rules = remove_redundant_rules(sub_rules)

    n_attributes = sub_rules.shape[1] - len([col for col in sub_rules.columns if col.startswith("y_")] +
                                            ["support", "confirmation", "complexity",
                                             "confidence", "coverage(y)"])
    if len(sub_rules) <= 20:
        return None

    dm_cls = get_dm(conf["expert"]["type"])

    instance_results = []

    for k in range(1, n_attributes):
        dm = dm_cls(rules=sub_rules, **conf["expert"])

        ranking_expert = dm.create_ranking(sub_rules)

        temp_dict = {
            "class": class_,
            "idx": idx_,
            "k": k,
            "covering_rules": len(sub_rules),
        }

        try:
            trusts_hr = solve_hr(sub_rules, dm, n_iters=conf["optimization"]["hr_iters"], top_k=k)
            trust_hr_centroid = trusts_to_centroid(trusts_hr)
            ranking_hr_centroid = trust_hr_centroid.create_ranking(sub_rules)
            hr_rules_on_first_pos = rules_on_first_pos(trusts_hr, sub_rules)
            discovered_rules = ranking_hr_centroid.index.get_loc(dm.reference_ranking[0])
            temp_dict.update({
                "jaccard_5_ranking_hr": set_jaccard(ranking_hr_centroid.index[:5], ranking_expert.index[:5]),
                "jaccard_5_hr_first_pos": set_jaccard(hr_rules_on_first_pos, ranking_expert.index[:5]),
                "jaccard_10_ranking_hr": set_jaccard(ranking_hr_centroid.index[:10], ranking_expert.index[:10]),
                "jaccard_10_hr_first_pos": set_jaccard(hr_rules_on_first_pos, ranking_expert.index[:10]),
                "kendall_ranking_hr": kendalltau(ranking_expert.index, ranking_hr_centroid.index).statistic.item(),
                "kendall_weights_hr": kendalltau(dm.attributes_preference, -trust_hr_centroid.to_np()).statistic,
                "discovered_rules_hr": discovered_rules,
            })
        except Exception as e:
            if "Optimization problem did not converge" in str(e):
                temp_dict["hr_error"] = "Optimization problem did not converge"
            else:
                raise e

        try:
            trust_epsilon = solve_epsilon(sub_rules, dm, top_k=k)
            ranking_epsilon = trust_epsilon.create_ranking(sub_rules)
            discovered_rules = ranking_epsilon.index.get_loc(dm.reference_ranking[0])
            temp_dict.update({
                "jaccard_5_ranking_epsilon": set_jaccard(ranking_epsilon.index[:5], ranking_expert.index[:5]),
                "jaccard_10_ranking_epsilon": set_jaccard(ranking_epsilon.index[:10], ranking_expert.index[:10]),
                "kendall_ranking_epsilon": kendalltau(ranking_expert.index, ranking_epsilon.index).statistic.item(),
                "kendall_weights_epsilon": kendalltau(dm.attributes_preference, -trust_epsilon.to_np()).statistic,
                "discovered_rules_epsilon": discovered_rules,
            })
        except Exception as e:
            if "Optimization problem did not converge" in str(e):
                temp_dict["epsilon_error"] = "Optimization problem did not converge"
            else:
                raise e
        instance_results.append(temp_dict)
    return instance_results


@click.command()
@click.argument("dataset_name", type=str, default="churn")
@click.option("--n_jobs", default=-1, help="Number of parallel jobs. -1 uses all cores.")
def main(dataset_name: TDatasetName, n_jobs: int) -> None:
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

    model.load_state_dict(torch.load(f"models/{dataset_name}.pth", weights_only=True))

    proba_train = model(dataset.train.as_torch(device=device)[0])[:, 1].detach().cpu().numpy()
    y_hat_train = (proba_train > conf.train.threshold).astype(int)

    rules = generate_rules(dataset, y_hat_train, **conf.rules)

    proba_test = model(dataset.test.as_torch(device=device)[0])[:, 1].detach().cpu().numpy()
    y_hat_test = (proba_test > conf.train.threshold).astype(int)

    tasks = []

    for class_ in [0, 1]:
        relevant_indices = dataset.test.y[y_hat_test == class_].index

        num_instances = min(300, len(relevant_indices))

        for idx_ in range(num_instances):
            real_idx = relevant_indices[idx_]

            tasks.append((
                idx_,
                class_,
                dataset.test.x.loc[real_idx],
                dataset.test.y.loc[real_idx],
                rules,
                conf
            ))

    print(f"Processing {len(tasks)} instances using {n_jobs} jobs...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_instance)(*args) for args in tqdm(tasks, desc="Parallel Processing")
    )

    stats = [item for sublist in results if sublist is not None for item in sublist]

    path = Path(f"outputs/mass/{dataset_name}_top_k.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved {len(stats)} results to {path}")


if __name__ == "__main__":
    main()
