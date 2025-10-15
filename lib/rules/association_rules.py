import concurrent.futures
from pathlib import Path

import pandas as pd


class RulesCache:
    cache_path = Path("cache/rules")

    @staticmethod
    def build_file_name(name: str, n_bins: int, support_threshold: float, confirm_measure: str,
                        confirm_threshold: float) -> str:
        return (f"name={name}"
                f"_n-bins={n_bins}"
                f"_min-support={support_threshold:.5f}"
                f"_confirm={confirm_measure}"
                f"_min_confirm={confirm_threshold:.5f}.csv")

    @classmethod
    def create_cache_dir(cls) -> None:
        cls.cache_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def contains(cls, name: str, n_bins: int, support_threshold: float, confirm_measure: str,
                 confirm_threshold: float) -> bool:
        cls.create_cache_dir()
        file_name = cls.build_file_name(name, n_bins, support_threshold, confirm_measure, confirm_threshold)
        return (cls.cache_path / file_name).exists()

    @classmethod
    def load(cls, name: str, n_bins: int, support_threshold: float, confirm_measure: str,
             confirm_threshold: float) -> pd.DataFrame:
        file_name = cls.build_file_name(name, n_bins, support_threshold, confirm_measure, confirm_threshold)
        return pd.read_csv(cls.cache_path / file_name)

    @classmethod
    def save(cls, itemsets: pd.DataFrame, name: str, n_bins: int, support_threshold: float, confirm_measure: str,
             confirm_threshold: float) -> None:
        file_name = cls.build_file_name(name, n_bins, support_threshold, confirm_measure, confirm_threshold)
        cls.create_cache_dir()
        itemsets.to_csv(cls.cache_path / file_name, index=False)


def generate_association_rules_new(frequent_itemsets: pd.DataFrame, df: pd.DataFrame, confirm_threshold: float,
                                   confirm_measure: str) -> pd.DataFrame:
    rules = []
    itemsets = frequent_itemsets["itemsets"].tolist()

    def process_itemset(itemset):
        local_rules = []
        for consequent in itemset:
            if consequent.startswith("y_"):
                antecedent = itemset - {consequent}
                if antecedent:
                    a = df[(df[list(antecedent)] == 1).all(axis=1) & (df[consequent] == 1)].shape[0]
                    b = df[(df[list(antecedent)] == 0).any(axis=1) & (df[consequent] == 1)].shape[0]
                    c = df[(df[list(antecedent)] == 1).all(axis=1) & (df[consequent] == 0)].shape[0]
                    d = df[(df[list(antecedent)] == 0).any(axis=1) & (df[consequent] == 0)].shape[0]

                    confidence = a / (a + c)
                    match confirm_measure:
                        case "confidence":
                            measure_value = confidence
                        case "S":
                            measure_value = (a / (a + c)) - (b / (b + d))
                        case "N":
                            measure_value = (a / (a + b)) - (c / (c + d))
                        case _:
                            raise ValueError(f"Unsupported confirmation measure: {confirm_measure}")

                    if measure_value >= confirm_threshold:
                        local_rules.append({
                            "antecedents": antecedent,
                            "consequent": consequent,
                            "support": (a + c) / (a + b + c + d),
                            "confirmation": measure_value,
                            "confidence": confidence,
                            "coverage(y)": a,
                        })

        return local_rules

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_itemset, itemsets)
        for result in results:
            rules.extend(result)

    return pd.DataFrame(rules)


def generate_association_rules(frequent_itemsets: pd.DataFrame, confirm_threshold: float,
                               confirm_measure: str) -> pd.DataFrame:
    """
    Generate association rules with consequents starting with 'y_' based on a chosen measure.

    Args:
        frequent_itemsets (pd.DataFrame): DataFrame of frequent itemsets with their support values.
        confirm_threshold (float): Minimum threshold for the confirmation measure.
        confirm_measure (str): Measure to evaluate rules (e.g., "confidence", "S", "N").

    Returns:
        pd.DataFrame: DataFrame of association rules with their metrics.
    """
    rules = []
    itemsets = frequent_itemsets["itemsets"].tolist()
    support_dict = {frozenset(itemset): support for itemset, support in
                    zip(frequent_itemsets["itemsets"], frequent_itemsets["support"])}
    total_support = sum(support_dict.values())

    for itemset in itemsets:
        for consequent in itemset:
            if consequent.startswith("y_"):
                antecedent = itemset - {consequent}
                if antecedent:
                    support_antecedent = support_dict[frozenset(antecedent)]
                    support_consequent = support_dict[frozenset([consequent])]
                    support_itemset = support_dict[frozenset(itemset)]

                    a = support_itemset
                    c = support_antecedent - support_itemset
                    b = support_consequent - support_itemset
                    d = total_support - (a + b + c)

                    match confirm_measure:
                        case "confidence":
                            measure_value = a / (a + c)
                        case "S":
                            measure_value = (a / (a + c)) - (b / (b + d))
                        case "N":
                            measure_value = (a / (a + b)) - (c / (c + d))
                        case _:
                            raise ValueError(f"Unsupported confirmation measure: {confirm_measure}")

                    if measure_value >= confirm_threshold:
                        rules.append({
                            "antecedents": antecedent,
                            "consequent": consequent,
                            "support": support_itemset,
                            confirm_measure: measure_value,

                        })

    return pd.DataFrame(rules)
