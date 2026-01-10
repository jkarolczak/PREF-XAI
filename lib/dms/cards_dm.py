import numpy as np
import pandas as pd

from lib.dms.interface_dm import IDM


def build_deck(columns_preferences: list[int]) -> list[int]:
    n_new_cards = len(columns_preferences)
    n_gaps = n_new_cards - 1

    if n_new_cards < 2:
        return columns_preferences
    counts = np.random.multinomial(n_new_cards, [1 / n_gaps] * n_gaps)
    deck = [columns_preferences[0]]
    for count, next_card in zip(counts, columns_preferences[1:]):
        deck.extend([None] * count)
        deck.append(next_card)
    return deck


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


class CardsDM(IDM):
    def __init__(self, rules: pd.DataFrame, n_reference_rules: int, columns_preferences: list[int], *args,
                 **kwargs) -> None:
        super().__init__(rules, n_reference_rules, *args, **kwargs)

        deck = build_deck(columns_preferences)
        weights = deck_to_weights(deck, n=len(columns_preferences))

        self.rules["weights"] = np.nanmean(self.rules.replace(0, np.nan).to_numpy() * weights, axis=1) + 1e-6
        rules_weights = np.abs(self.rules["weights"].to_numpy(dtype=np.float32)) ** 2
        idcs = np.random.choice(self.rules.index.tolist(), size=self.n_reference_rules, replace=False,
                                p=rules_weights / rules_weights.sum()).tolist()

        self._reference_ranking = self.rules.loc[idcs].sort_values(by=["weights"], ascending=False).index.tolist()
        self._attribute_ranking = list(np.argsort(-weights).tolist())

    def create_ranking(self, rules: pd.DataFrame) -> pd.DataFrame:
        rules = rules.copy()
        rules["weights"] = np.nanmean(
            rules.drop(columns=self.special_columns, errors="ignore").replace(0, np.nan).to_numpy() *
            self.rules["weights"].to_numpy(dtype=np.float32),
            axis=1
        ) + 1e-6
        return rules.sort_values(by=["weights"], ascending=False)


class RandomCardsDM(CardsDM):
    def __init__(self, rules: pd.DataFrame, n_reference_rules: int, *args, **kwargs) -> None:
        columns_preferences = np.random.permutation(len(rules.columns) - len(
            [col for col in rules.columns if col.startswith("y_")] + ["confidence", "coverage(y)"]
        )).tolist()
        super().__init__(rules, n_reference_rules, columns_preferences, *args, **kwargs)
