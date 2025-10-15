import numpy as np
import pandas as pd


def build_deck(columns_preferences: list[int], new_cards: int = 10) -> list[int]:
    dividers = np.sort(np.random.choice(np.arange(new_cards - 4), size=3, replace=False))
    parts = np.diff(np.concatenate([[0], dividers, [new_cards - 4]])) + 1
    np.random.shuffle(parts)
    m0 = np.floor(len(columns_preferences) / 2 - 1).astype(int).item()
    m1 = np.ceil(len(columns_preferences) / 2 + 1).astype(int).item()
    return (
            columns_preferences[:3] +
            [None] * parts[0] +
            columns_preferences[3:m0] +
            [None] * parts[1] +
            columns_preferences[m0:m1] +
            [None] * parts[2] +
            columns_preferences[m1:-3] +
            [None] * parts[3] +
            columns_preferences[-3:]
    )


def deck_to_weights(deck: list[int], n: int) -> np.ndarray:
    limit = np.random.randint(2, 5)
    weights = np.linspace(limit, -limit, len(deck))
    weights = [w for w, d in zip(weights, deck) if d is not None]
    i = 0
    result = [None for _ in range(n)]
    for d in deck:
        if d is not None:
            result[d] = weights[i].item()
            i += 1
    return np.array(result, dtype=np.float32)


def cards_deck(rules: pd.DataFrame, n_rules: int, columns_preferences: list[int] | None = None, *args, **kwargs) -> list[int]:
    rules = rules.copy()
    special_columns = [col for col in rules.columns if col.startswith("y_")] + [
        "support", "confirmation", "complexity", "confidence", "coverage(y)"]

    deck = build_deck(columns_preferences)
    weights = deck_to_weights(deck, n=len(columns_preferences))

    rules["weights"] = np.nanmean(
        rules.drop(columns=special_columns, errors="ignore").replace(0, np.nan).to_numpy() * weights,
        axis=1
    ) + 1e-6

    rules_weights = np.abs(rules["weights"].to_numpy(dtype=np.float32))

    idcs = np.random.choice(rules.index.tolist(), size=n_rules, replace=False,
                            p=rules_weights / rules_weights.sum()).tolist()

    sub_rules = rules.loc[idcs].sort_values(by=["weights"], ascending=False)

    return sub_rules.index.tolist()
