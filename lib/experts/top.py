from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def top_3(rules: pd.DataFrame, n_rules: int, columns_preferences: list[int] | None = None, *args, **kwargs) -> list[int]:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + [
        "support", "confirmation", "complexity", "confidence", "coverage(y)"]

    ranking = [None for _ in range(n_rules)]
    ranking[:3] = select_top_3_rules(rules, columns_preferences, *args, **kwargs)
    ranking[-3:] = select_top_3_rules(rules, columns_preferences[::-1], *args, **kwargs)[::-1]

    jaccard_distances = pairwise_distances(
        rules.drop(columns=special_columns, errors="ignore").to_numpy().astype(bool),
        metric="jaccard"
    )

    if any([any([r1 == r2 for r2 in ranking[-3:]]) for r1 in ranking[:3]]):
        ranking[-3:] = [None for _ in range(3)]
        idcs = ranking[:3]
        end = n_rules
    else:
        idcs = ranking[:3] + ranking[-3:]
        end = n_rules - 3

    for i in range(3, end):
        distances = np.min(jaccard_distances[idcs], axis=0)
        candidate_indices = np.argsort(-distances)
        for idx in candidate_indices:
            if idx not in idcs:
                idcs.append(idx.item())
                ranking[i] = idx.item()
                break
    return ranking


def select_top_3_rules(rules: pd.DataFrame, columns_preferences: list[int], *args, **kwargs) -> list[int]:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + [
        "support", "confirmation", "complexity", "confidence", "coverage(y)"]
    columns = [col for col in rules.columns if col not in special_columns]
    unpreferred_columns = [col for i, col in enumerate(columns) if i in columns_preferences[-3:]]
    rules = rules.copy()
    ranking = []

    for combinations_length in range(3):
        combs = combinations(columns_preferences[:3], combinations_length + 1)
        for comb in combs:
            sub_columns = [c for i, c in enumerate(columns) if i in comb]

            mask = rules[sub_columns].all(axis=1) & ~rules.drop(columns=sub_columns).any(axis=1)
            selected_rows = rules[mask]
            selected_rows = selected_rows.sort_values(by=["complexity", "support"], ascending=[True, False])

            for i, row in selected_rows.iterrows():
                if i not in ranking:
                    ranking.append(i)
                if len(ranking) >= 3:
                    return ranking
    for combinations_length in [3, 2, 1]:
        combs = combinations(columns_preferences[:3], combinations_length)
        for comb in combs:
            sub_columns = [c for i, c in enumerate(columns) if i in comb]

            mask = rules[sub_columns].all(axis=1) & ~rules[unpreferred_columns].any(axis=1)
            selected_rows = rules[mask]
            selected_rows = selected_rows.sort_values(by=["complexity", "support"],
                                                      ascending=[True, False])

            for i, row in selected_rows.iterrows():
                if i not in ranking:
                    ranking.append(i)
                if len(ranking) >= 3:
                    return ranking

    rules_copy = rules.copy()
    if len(ranking) > 0:
        rules_copy = rules_copy.drop(index=ranking)
    sub_columns = [c for i, c in enumerate(columns) if i in columns_preferences[:3]]
    rules_copy["preference"] = rules_copy[sub_columns].sum(axis=1)
    rules_copy["unpreference"] = rules_copy[unpreferred_columns].sum(axis=1)
    rules_copy = rules_copy.sort_values(by=["unpreference", "preference", "complexity", "support"],
                                        ascending=[True, False, True, False])

    ranking += rules_copy.index.tolist()[:3 - len(ranking)]
    return ranking
