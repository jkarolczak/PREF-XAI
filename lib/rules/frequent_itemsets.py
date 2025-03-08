from pathlib import Path

import pandas as pd


class ItemsetsCache:
    cache_path = Path("cache/itemsets")

    @staticmethod
    def build_file_name(name: str, n_bins: int, support_threshold: float) -> str:
        return f"name={name}_n-bins={n_bins}_min-support={support_threshold:.5f}.csv"

    @classmethod
    def create_cache_dir(cls) -> None:
        cls.cache_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def contains(cls, name: str, n_bins: int, support_threshold: float) -> bool:
        file_name = ItemsetsCache.build_file_name(name, n_bins, support_threshold)
        cls.create_cache_dir()
        return (cls.cache_path / file_name).exists()

    @classmethod
    def load(cls, name, n_bins: int, support_threshold: float) -> pd.DataFrame:
        file_name = ItemsetsCache.build_file_name(name, n_bins, support_threshold)
        df = pd.read_csv(cls.cache_path / file_name)
        df["itemsets"] = df["itemsets"].apply(lambda x: frozenset(eval(x)))
        return df

    @classmethod
    def save(cls, itemsets: pd.DataFrame, name: str, n_bins: int, support_threshold: float) -> None:
        file_name = ItemsetsCache.build_file_name(name, n_bins, support_threshold)
        cls.create_cache_dir()
        itemsets["itemsets"] = itemsets["itemsets"].apply(lambda x: frozenset(list(x)))
        itemsets.to_csv(cls.cache_path / file_name, index=False)


def generate_frequent_itemsets(df: pd.DataFrame, support_threshold: float) -> pd.DataFrame:
    """
    Generate frequent itemsets starting with 2-itemsets containing at least one column starting with 'y_'
    and merging itemsets iteratively using Apriori algorithm.
    Args:
        df (pd.DataFrame): Binary dataframe of transactions.
        support_threshold (float): Minimum support threshold.
    Returns:
        pd.DataFrame: Frequent itemsets with their support values.
    """
    from mlxtend.frequent_patterns import apriori

    frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames=True, low_memory=True)
    return frequent_itemsets
