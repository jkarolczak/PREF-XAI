import numpy as np
import pandas as pd

from lib.data import IDataset


def complexity(rules: pd.DataFrame, normalization_factor: int = 1) -> pd.Series:
    """Calculates complexity of rules.

    Args:
        rules (pd.DataFrame): rules
        normalization_factor (int): normalization factor (maximal possible number of antecedents)

    Returns:

    """
    return (rules.sum(axis=1) - 1) / normalization_factor


def dumminize(dataset: IDataset, y_hat: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Transforms dataset into a format suitable for association rules generation.

    Each categorical feature is transformed into a set of binary features using one-hot encoding. Each numerical feature is
    transformed into a set of binary features by assigning to respective quantile.

    Args:
        dataset (IDataset): dataset
        y_hat (np.ndarray): predictions
        n_bins (int): number of quantiles to split numerical features

    Returns:
        pd.DataFrame: transformed dataset

    """
    cat_cols = list(dataset.train.x.columns[dataset.cat_idxs])
    num_cols = list(dataset.train.x.columns[dataset.num_idxs])

    df = dataset.train.x.copy()
    df = pd.get_dummies(df, columns=cat_cols)

    for col in num_cols:
        quantiles = [df[col].quantile(q=q).item() for q in [i / n_bins for i in range(n_bins + 1)]]
        quantiles = sorted(list(set(quantiles)))
        quantiles[0] = np.floor(quantiles[0])
        quantiles[-1] = np.ceil(quantiles[-1])
        for i in range(len(quantiles) - 1):
            df[f"{col}_{quantiles[i]}-{quantiles[i + 1]}"] = (
                    (df[col] >= quantiles[i]) & (df[col] < quantiles[i + 1])
            )
        df = df.drop(col, axis=1)

    df["y"] = y_hat
    df = pd.get_dummies(df, columns=["y"])

    return df


def decode_rules(rules: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Decodes rules. Each rule is transformed from a set of feature names to a set of binary features as columns.

    Args:
        rules (pd.DataFrame): rules
        df (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: decoded rules
    """

    column_rules = pd.DataFrame(index=rules.index, columns=df.columns)
    for antecedent in df.columns:
        column_rules[antecedent] = rules["antecedents"].apply(lambda x: antecedent in x)

    for c in [c for c in column_rules.columns if c.startswith("y_")]:
        column_rules[c] = rules["consequent"] == c

    return column_rules


def column_to_range(c: str) -> dict[str, float | str]:
    """Transforms a column name to a dictionary with range values.

    Args:
        c (str): columns name

    Returns:
        dict[str, float | str]: dictionary with range values
    """

    return {
        "column": c,
        "attribute": c.split("_")[0],
        "range_min": float(c.split("_")[1].split("-")[0]),
        "range_max": float(c.split("_")[1].split("-")[1]),
    }


def column_to_value(c: str) -> dict[str, float | str]:
    """Transforms a column name to a dictionary with a single value.

    Args:
        c (str): column name

    Returns:
        dict[str, float | str]: dictionary with a single value

    """
    attribute, value = c.split("_")
    try:
        value = float(value)
    except ValueError:
        value = value
    return {
        "column": c,
        "attribute": attribute,
        "value": value,
    }


def rules_criteria_as_list(rules: pd.DataFrame) -> list[dict[str, float | str]]:
    """Transforms rules DataFrame to a list of dictionaries with criteria.

    Args:
        rules (pd.DataFrame): rules DataFrame

    Returns:
        list[dict[str, float | str]]: list of dictionaries with criteria
    """

    rules_list = [column_to_range(c) if "-" in c else column_to_value(c) for c in rules.columns]
    return rules_list


def instance_to_criteria(instance: pd.Series, criteria_list: list[dict[str, float | str]], columns: list[str]) -> pd.DataFrame:
    """Transforms instance to a DataFrame with criteria.

    Args:
        instance (pd.Series): instance
        criteria_list (list[dict[str, float | str]]): list of dictionaries with criteria
        columns (list[str]): columns names

    Returns:
        pd.DataFrame: DataFrame with criteria
    """

    criteria = []
    for c in criteria_list:
        if "range_min" in c:
            if c["range_min"] <= instance[c["attribute"]] < c["range_max"]:
                criteria.append(c["column"])
        else:
            if c["value"] == instance[c["attribute"]]:
                criteria.append(c["column"])
    criteria_df = pd.DataFrame(columns=columns)
    criteria_df.loc[0] = [False] * len(columns)
    for c in criteria:
        criteria_df[c] = True
    return criteria_df


def rule_to_columns_list(rule: pd.Series) -> list[str]:
    """Transforms a rule to a list of columns.

    Args:
        rule (pd.Series): rule

    Returns:
        list[str]: list of columns
    """

    return list(rule[rule].index)


def get_rules_covering_instance(rules: pd.DataFrame, instance: pd.Series, cls: int) -> pd.DataFrame:
    """Returns rules covering an instance.

    Args:
        rules (pd.DataFrame): rules
        instance (pd.Series): instance
        cls (int): class

    Returns:
        pd.DataFrame: rules covering an instance
    """

    blacklist = ["support", "complexity", "confirmation", "y_0", "y_1", "confidence", "coverage(y)"]
    columns = list(filter(lambda c: c not in blacklist, rules.columns))
    criteria_list = rules_criteria_as_list(rules.drop(columns=blacklist))
    instance_as_criteria = instance_to_criteria(instance, criteria_list, columns)

    rules = rules.copy()
    rules = rules[rules[f"y_{cls}"] == 1]

    rules = rules[
        rules.drop(columns=blacklist).apply(
            lambda r: all(instance_as_criteria[c].values for c in rule_to_columns_list(r)),
            axis=1
        )
    ]
    return rules


def get_instances_covered_by_rule(rule: pd.DataFrame, dataset: IDataset) -> list[int]:
    """Returns instances covered by a rule.

    Args:
        rule (pd.DataFrame): rule
        dataset (IDataset): dataset

    Returns:
        pd.DataFrame: instances covered by a rule
    """

    blacklist = ["support", "complexity", "confirmation", "y_0", "y_1", "confidence", "coverage(y)"]
    columns = list(filter(lambda c: c not in blacklist, dataset.train.x.columns))

    criteria_list = rules_criteria_as_list(rule.drop(columns=blacklist))
    column_list = rule_to_columns_list(rule.drop(columns=blacklist).iloc[0])

    instances = []
    for i, instance in dataset.train.x.iterrows():
        instance_as_criteria = instance_to_criteria(instance, criteria_list, columns)
        for c in column_list:
            if c not in instance_as_criteria.columns:
                break
        instances.append(i)
    return instances


def get_attribute_names(rules: pd.DataFrame) -> list[str]:
    """Returns attribute names from rules composed of attribute names and values.

    Args:
        rules (pd.DataFrame): rules

    Returns:
        list[str]: attribute names
    """

    attributes = []
    for c in rules.columns:
        attribute = c.split("_")[0]
        if attribute not in attributes:
            attributes.append(attribute)
    return attributes


def wrap_rules_to_attributes(rules: pd.DataFrame) -> pd.DataFrame:
    """Wraps rules composed of attribute names and values to a DataFrame with attribute names only.

    Args:
        rules (pd.DataFrame): rules

    Returns:
        pd.DataFrame: rules with attribute names only
    """

    columns = ["confidence", "coverage(y)", "support", "complexity", "confirmation", "y_0", "y_1"]
    rules_limited = rules.drop(columns=columns)
    rules_limited = rules_limited.apply(lambda r: [c.split("_")[0] for c in r.index if r[c]], axis=1)

    attributes = get_attribute_names(rules.drop(columns=columns))

    output = pd.DataFrame(columns=attributes + columns)
    for i, item in rules_limited.items():
        output.loc[i, list(set(attributes) - set(item))] = False
        output.loc[i, item] = True
        output.loc[i, columns] = rules.loc[i, columns]
    return output
