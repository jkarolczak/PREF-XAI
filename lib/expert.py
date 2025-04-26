from itertools import combinations
from typing import (Literal,
                    TypeAlias)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from lib.data import IDataset
from lib.rules.utils import get_instances_covered_by_rule

TExpert: TypeAlias = Literal["diversify_by_properties", "diversify_by_spanning_examples"]


def sort_by_columns_preferences(idcs: list[int], rules: pd.DataFrame,
                                columns_preferences: list[int]) -> list[int]:
    """Sorts rules by columns preferences.

    Args:
        idcs (list[int]): List of selected indices
        rules (pd.DataFrame): DataFrame of rules.
        columns_preferences (list[int]): List of columns preferences.

    Returns:
        list[int]: Sorted list of indices.
    """

    columns_weights = {col: 1 - i / len(columns_preferences) for i, col in enumerate(columns_preferences)}
    columns_weights = np.array([columns_weights[col] for col in sorted(columns_weights.keys())])
    order = np.argsort(
        np.array([columns_weights @ rules.iloc[i, columns_preferences].to_numpy() for i in idcs]))
    idcs = [idcs[i] for i in order]
    return idcs


def diversify_by_properties(rules_wrapped: pd.DataFrame, n_rules: int, columns_preferences: list[int] | None = None,
                            idcs: list[int] | None = None, *args, **kwargs) -> list[int]:
    """Selects the most diversified rules based on Jaccard index between the rules.

    Args:
        rules_wrapped (pd.DataFrame): DataFrame of rules.
        n_rules (int): The number of rules to select.
        columns_preferences (list[int], optional): List of columns preferences. Defaults to None.
        idcs (list[int], optional): List of selected indices to start from. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of diversified rules.
    """
    special_columns = [col for col in rules_wrapped.columns if col.startswith("y_")] + ["support", "confirmation",
                                                                                        "complexity",
                                                                                        "confidence", "coverage(y)"]
    jaccard_similarities = pairwise_distances(
        rules_wrapped.drop(columns=special_columns, errors="ignore").to_numpy().astype(bool),
        metric="jaccard")

    idcs = idcs or [rules_wrapped.drop(columns=special_columns, errors="ignore").to_numpy().sum(axis=1).argmax().item()]
    for _ in range(1, n_rules):
        min_distances = np.min(jaccard_similarities[idcs], axis=0)
        nearest_idcs = np.argwhere(min_distances == np.max(min_distances, axis=0)).squeeze(1)
        if any(mask := rules_wrapped.iloc[nearest_idcs]["support"] >= (
                2 / rules_wrapped.drop(columns=special_columns, errors="ignore").shape[1])):
            nearest_idcs = nearest_idcs[mask.values.tolist()]
        farthest_point = np.random.choice(nearest_idcs.tolist()).item()
        idcs.append(farthest_point)
    if columns_preferences is not None:
        idcs = sort_by_columns_preferences(idcs, rules_wrapped, columns_preferences)

    return idcs


def diversify_by_spanning_examples(rules: pd.DataFrame, rules_wrapped: pd.DataFrame, dataset: IDataset, n_rules: int,
                                   columns_preferences: list[int] | None = None, *args, **kwargs) -> list[int]:
    """Selects rules by maximizing the diversity of the examples covered by the rules.

    Args:
        rules (pd.DataFrame): DataFrame of rules.
        rules_wrapped (pd.DataFrame): Wrapped rules.
        dataset (IDataset): Dataset.
        n_rules (int): The number of rules to select.
        columns_preferences (list[int], optional): List of columns preferences. Defaults to None.

    Returns:
        list[int]: List of selected rule indices.
    """

    rules = rules.copy()
    covered_examples = []
    for i in range(len(rules)):
        rule = rules.iloc[[i]]
        covered_examples.append(
            get_instances_covered_by_rule(rule, dataset)
        )
    covered_examples = np.array([[idx in ex for idx in range(len(dataset.x))] for ex in covered_examples])
    jaccard_similarities = pairwise_distances(covered_examples, metric="jaccard")

    idcs = [np.random.choice(list(range(len(rules))))]

    for _ in range(1, n_rules):
        min_distances = np.min(jaccard_similarities[idcs], axis=0)
        farthest_point = np.argmax(min_distances)

        if farthest_point not in idcs:
            idcs.append(farthest_point)
        else:
            sorted_indices = np.argsort(-min_distances)
            for idx in sorted_indices:
                if idx not in idcs:
                    idcs.append(idx)
                    break

    if columns_preferences is not None:
        idcs = sort_by_columns_preferences(idcs, rules_wrapped, columns_preferences)

    return idcs


def select_top_3_rules(rules: pd.DataFrame, rules_wrapped: pd.DataFrame,
                       columns_preferences: list[int], *args, **kwargs) -> list[int]:
    special_columns = [col for col in rules_wrapped.columns if col.startswith("y_")] + [
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


def natural(rules: pd.DataFrame, rules_wrapped: pd.DataFrame, n_rules: int,
            columns_preferences: list[int] | None = None, *args, **kwargs) -> list[int]:
    special_columns = [col for col in rules_wrapped.columns if col.startswith("y_")] + [
        "support", "confirmation", "complexity", "confidence", "coverage(y)"]

    ranking = [None for _ in range(n_rules)]
    ranking[:3] = select_top_3_rules(rules, rules_wrapped, columns_preferences, *args, **kwargs)
    ranking[-3:] = select_top_3_rules(rules, rules_wrapped, columns_preferences[::-1], *args, **kwargs)[::-1]

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


def get_expert(name: str) -> callable:
    """Returns the expert function based on the name.

    Args:
        name (str): The name of the expert function.

    Returns:
        callable: The expert function.
    """

    return globals()[name]
