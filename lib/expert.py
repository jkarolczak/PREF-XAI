from typing import (Literal,
                    TypeAlias)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from lib.data import IDataset
from lib.rules.utils import get_instances_covered_by_rule

TExpert: TypeAlias = Literal["diversify_by_properties", "diversify_by_spanning_examples"]


def sort_by_columns_preferences(idcs: list[int], rules: pd.DataFrame,
                                columns_preferences: list[int]) -> pd.DataFrame:
    """Sorts rules by columns preferences.

    Args:
        idcs (list[int]): List of selected indices
        rules (pd.DataFrame): DataFrame of rules.
        columns_preferences (list[int]): List of columns preferences.

    Returns:
        pd.DataFrame: DataFrame of sorted rules.
    """

    columns_weights = {col: 1 - i / len(columns_preferences) for i, col in enumerate(columns_preferences)}
    columns_weights = np.array([columns_weights[col] for col in sorted(columns_weights.keys())])
    order = np.argsort(
        np.array([columns_weights @ rules.iloc[i, columns_preferences].to_numpy() for i in idcs]))
    idcs = [idcs[i] for i in order]
    return idcs


def diversify_by_properties(rules_wrapped: pd.DataFrame, n_rules: int, columns_preferences: list[int] | None = None,
                            *args, **kwargs) -> list[int]:
    """Selects the most diversified rules based on Jaccard index between the rules.

    Args:
        rules_wrapped (pd.DataFrame): DataFrame of rules.
        n_rules (int): The number of rules to select.
        columns_preferences (list[int], optional): List of columns preferences. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of diversified rules.
    """
    special_columns = [col for col in rules_wrapped.columns if col.startswith("y_")] + ["support", "confirmation",
                                                                                        "complexity",
                                                                                        "confidence", "coverage(y)"]
    jaccard_similarities = pairwise_distances(
        rules_wrapped.drop(columns=special_columns, errors="ignore").to_numpy().astype(bool),
        metric="jaccard")

    idcs = [rules_wrapped.drop(columns=special_columns, errors="ignore").to_numpy().sum(axis=1).argmax().item()]
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
            pd.DataFrame: DataFrame of diversified rules.
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


def get_expert(name: str) -> callable:
    """Returns the expert function based on the name.

    Args:
        name (str): The name of the expert function.

    Returns:
        callable: The expert function.
    """

    return globals()[name]
