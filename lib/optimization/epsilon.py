import pandas as pd
import pulp as pl

from lib.dms.interface_dm import IDM
from lib.optimization.trust import Trust


def solve_epsilon(
        rules: pd.DataFrame,
        dm: IDM,
        top_k: int | None = None
) -> Trust:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["support", "confirmation", "complexity",
                                                                                "confidence", "coverage(y)"]
    # Extract user preferences only
    rules = rules.copy()
    rules = rules.iloc[dm.reference_ranking, :]

    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()

    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    n_conditions = conditions.shape[1]

    if top_k is not None:
        active_indices = set(dm.attributes_preference[:top_k])
    else:
        active_indices = set(range(n_conditions))

    # PULP problem
    problem = pl.LpProblem("Maximize_Epsilon", pl.LpMaximize)

    # Define variables
    w_j = [
        pl.LpVariable(f"w_j_{i}", lowBound=0, upBound=1) if i in active_indices else 0.0
        for i in range(n_conditions)
    ]
    w_supp = pl.LpVariable("w_supp", lowBound=0, upBound=1)
    w_confirm = pl.LpVariable("w_confirm", lowBound=0, upBound=1)
    w_complex = pl.LpVariable("w_complex", lowBound=0, upBound=1)

    # Trust calculation definition
    trust = [
        pl.lpSum(w_j[i] * conditions[j, i] for i in range(n_conditions)) +
        w_supp * support[j] +
        w_confirm * confirmation[j] -
        w_complex * complexity[j]
        for j in range(len(dm.reference_ranking))
    ]

    # Constraints: Sum of weights
    problem += pl.lpSum(w_j) + w_supp + w_confirm + w_complex == 1, "Sum_of_Weights"

    # Constraints: User preferences
    epsilon = pl.LpVariable("epsilon", lowBound=0, upBound=1)
    for i in range(len(dm.reference_ranking) - 1):
        problem += trust[i] >= trust[i + 1] + epsilon, f"Preference_{i}"

    # Objective function
    problem += epsilon, "Maximize_Epsilon"

    # Solve the problem

    if pl.GUROBI(msg=False).solve(problem) != 1:
        raise Exception("Optimization problem did not converge")

    w_j_values = [
        item.varValue if isinstance(item, pl.LpVariable) else item
        for item in w_j
    ]

    return Trust(
        rules,
        w_j_values,
        w_supp.varValue,
        w_confirm.varValue,
        w_complex.varValue
    )
