from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Prus:
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
        rules["prus"] = rules.apply(self.compute, axis=1)
        return rules.sort_values("prus", ascending=False)

    def __repr__(self) -> str:
        nl = "\n"
        return (f"Prus(w_j=[{nl}{(',' + nl).join([f'  w_{i}: {w_j:.16f}' for i, w_j in enumerate(self.w_j)])}{nl}], "
                f"w_supp={self.w_supp:.16f}, w_confirm={self.w_confirm:.16f}, w_complex={self.w_complex:.16f})")

    def print(self) -> None:
        print("Prus(")
        for i, w in enumerate(self.w_j):
            print(f"  w_{i}: {w:.16f},")
        print(f"  w_supp: {self.w_supp:.16f}, w_confirm: {self.w_confirm:.16f}, w_complex: {self.w_complex:.16f}")
        print(")")

    def to_np(self) -> np.ndarray:
        return np.array([*self.w_j, self.w_supp, self.w_confirm, self.w_complex])

    @staticmethod
    def from_np(array: np.ndarray, rules: pd.DataFrame) -> "Prus":
        return Prus(
            rules=rules,
            w_j=array[:-3].tolist(),
            w_supp=array[-3].item(),
            w_confirm=array[-2].item(),
            w_complex=array[-1].item()
        )
