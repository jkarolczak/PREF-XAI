import numpy as np
import pandas as pd

from lib.data import IDataset
from lib.rules.association_rules import (generate_association_rules_new,
                                         RulesCache)
from lib.rules.frequent_itemsets import (generate_frequent_itemsets,
                                         ItemsetsCache)
from lib.rules.utils import (complexity,
                             dumminize,
                             decode_rules,
                             remove_redundant_rules)


def generate_rules(dataset: IDataset, y_hat: np.ndarray, support_threshold: float, confirm_threshold: float,
                   confidence_threshold: float, n_bins: int, confirm_measure: str = "S") -> pd.DataFrame:
    df_dumminized = dumminize(dataset, y_hat, n_bins)

    if ItemsetsCache.contains(dataset.name, n_bins, support_threshold):
        frequent_itemsets = ItemsetsCache.load(dataset.name, n_bins, support_threshold)
    else:
        frequent_itemsets = generate_frequent_itemsets(df=df_dumminized, support_threshold=support_threshold)
        ItemsetsCache.save(frequent_itemsets, name=dataset.name, n_bins=n_bins, support_threshold=support_threshold)

    if RulesCache.contains(dataset.name, n_bins, support_threshold, confirm_measure, confirm_threshold):
        rules = RulesCache.load(dataset.name, n_bins, support_threshold, confirm_measure, confirm_threshold)
    else:
        rules = generate_association_rules_new(frequent_itemsets, df_dumminized, confirm_threshold=confirm_threshold,
                                               confirm_measure=confirm_measure)
        RulesCache.save(rules, name=dataset.name, n_bins=n_bins, support_threshold=support_threshold,
                        confirm_measure=confirm_measure, confirm_threshold=confirm_threshold)
    
    support = rules["support"]
    confirm = rules["confirmation"]
    confidence = rules["confidence"]
    coverage = rules["coverage(y)"]
    rules = decode_rules(rules, df_dumminized)
    rules["complexity"] = complexity(rules, normalization_factor=dataset.train.x.shape[1])
    rules["support"] = support
    rules["confirmation"] = confirm
    rules["confidence"] = confidence
    rules["coverage(y)"] = coverage
    rules = rules[rules["confidence"] >= confidence_threshold].reset_index(drop=True)
    return rules
