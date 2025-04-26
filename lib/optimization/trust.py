from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trust:
    rules: pd.DataFrame
    w_j: list[float]
    w_supp: float
    w_confirm: float
    w_complex: float

    def compute(self, rule: pd.Series):
        return (
                sum([self.w_j[i] * rule.iloc[i] for i in range(len(self.w_j))]) +
                self.w_supp * rule["support"] +
                self.w_confirm * rule["confirmation"] -
                self.w_complex * rule["complexity"]
        )

    def create_ranking(self, rules: pd.DataFrame) -> pd.DataFrame:
        rules = rules.copy()
        rules["trust"] = rules.apply(self.compute, axis=1)
        return rules.sort_values("trust", ascending=False)

    def __repr__(self) -> str:
        nl = "\n"
        return (f"Trust(w_j=[{nl}{(',' + nl).join([f'  w_{i}: {w_j:.16f}' for i, w_j in enumerate(self.w_j)])}{nl}], "
                f"w_supp={self.w_supp:.16f}, w_confirm={self.w_confirm:.16f}, w_complex={self.w_complex:.16f})")

    def print(self) -> None:
        print("Trust(")
        for i, w in enumerate(self.w_j):
            print(f"  w_{i}: {w:.16f},")
        print(f"  w_supp: {self.w_supp:.16f}, w_confirm: {self.w_confirm:.16f}, w_complex: {self.w_complex:.16f}")
        print(")")

    def to_np(self) -> np.ndarray:
        return np.array([*self.w_j, self.w_supp, self.w_confirm, self.w_complex])
