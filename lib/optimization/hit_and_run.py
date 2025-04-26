from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lib.optimization.epsilon import solve_epsilon
from lib.optimization.trust import Trust


def write_initial(rules: pd.DataFrame, user_preferences: list[int], path: Path) -> None:
    rules = rules.copy()
    trust: Trust = solve_epsilon(rules, user_preferences)

    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["confidence", "coverage(y)"]

    with path.open("w") as fp:
        for i, row in rules.iloc[user_preferences, :].iterrows():
            values = row.drop(special_columns).to_numpy()
            fp.write(f"Data {','.join([str(v) for v in values[:-2].astype(int)])}")
            fp.write(f",{','.join([str(v) for v in values[-2:].astype(float)])}\n")
        fp.write("Data " + ",".join([str(x) for x in trust.to_np()]) + "\n")


def aggregate_results(rules: pd.DataFrame, user_preferences: list[int], path: Path) -> list[list[int]]:
    rules = rules.copy()
    rules_preferences = rules.iloc[user_preferences]

    with open(path, "r") as fp:
        lines = fp.readlines()

    occurrences = [[] for _ in range(len(rules_preferences))]

    for line in tqdm(lines):
        prefix, weights, sum_ = line.split(";")
        weights = [float(w) for w in weights.strip().split(" ")]
        trust = Trust(
            rules_preferences,
            w_j=weights[:-3],
            w_supp=weights[-3],
            w_confirm=weights[-2],
            w_complex=weights[-1]
        )

        if np.allclose(
                rules_preferences.index.values,
                trust.create_ranking(rules_preferences).index.values
        ):
            ranking = trust.create_ranking(rules)

            # find positions of preferred rules in the ranking
            for i, position in enumerate(rules_preferences.index):
                occurrences[i].append(ranking.index.get_loc(position))

    return occurrences
