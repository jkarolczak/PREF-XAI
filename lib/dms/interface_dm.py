from abc import (abstractmethod,
                 ABC)

import pandas as pd


class IDM(ABC):
    def __init__(self, rules: pd.DataFrame, n_reference_rules: int, *args, **kwargs) -> None:
        self.rules = rules.copy()
        self.special_columns = [col for col in rules.columns if col.startswith("y_")] + [
            "confidence", "coverage(y)"]
        self.rules = self.rules.drop(columns=self.special_columns)
        self.n_reference_rules = n_reference_rules

    @property
    def reference_ranking(self) -> list[int]:
        if (ranking := getattr(self, "_reference_ranking", None)) is None:
            raise Exception("DM has to initialize reference ranking in the constructor")
        return ranking

    @abstractmethod
    def create_ranking(self, rules: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement create_ranking method")

    @property
    def attributes_preference(self) -> list[int]:
        if (ranking := getattr(self, "_attribute_ranking", None)) is None:
            raise Exception("DM has to initialize attribute ranking in the constructor")
        return ranking
