from dataclasses import dataclass

import pandas as pd
import pulp as pl


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


def solve(rules: pd.DataFrame, user_preferences: list[int]) -> Trust:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["support", "confirmation", "complexity",
                                                                                "confidence", "coverage(y)"]

    # Extract user preferences only
    rules = rules.copy()
    rules = rules.iloc[user_preferences, :]

    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()

    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    num_conditions = conditions.shape[1]

    # PULP problem
    problem = pl.LpProblem("Maximize_Epsilon", pl.LpMaximize)

    # Define variables
    w_j = [pl.LpVariable(f"w_j_{i}", lowBound=0, upBound=1) for i in range(num_conditions)]
    w_supp = pl.LpVariable("w_supp", lowBound=0, upBound=1)
    w_confirm = pl.LpVariable("w_confirm", lowBound=0, upBound=1)
    w_complex = pl.LpVariable("w_complex", lowBound=0, upBound=1)

    # Trust calculation definition
    trust = [
        pl.lpSum(w_j[i] * conditions[j, i] for i in range(num_conditions)) +
        w_supp * support[j] +
        w_confirm * confirmation[j] -
        w_complex * complexity[j]
        for j in range(len(user_preferences))
    ]

    # Constraints: Sum of weights
    problem += pl.lpSum(w_j) + w_supp + w_confirm + w_complex == 1, "Sum_of_Weights"

    # Constraints: User preferences
    epsilon = pl.LpVariable("epsilon", lowBound=0, upBound=1)
    for i in range(len(user_preferences) - 1):
        problem += trust[i] >= trust[i + 1] + epsilon, f"Preference_{i}"

    # Objective function
    problem += epsilon, "Maximize_Epsilon"

    # Solve the problem
    pl.GUROBI(msg=False).solve(problem)

    if problem.status != 1:
        raise Exception("Optimization problem did not converge")

    return Trust(
        rules,
        [w_j[i].varValue for i in range(num_conditions)],
        w_supp.varValue,
        w_confirm.varValue,
        w_complex.varValue
    )
