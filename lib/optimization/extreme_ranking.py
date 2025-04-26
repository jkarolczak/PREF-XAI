from typing import Literal

import numpy as np
import pandas as pd
import pulp as pl

M = 1000
INEQ = 1e-6


def _rules_to_values(rules: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["support", "confirmation", "complexity",
                                                                                "confidence", "coverage(y)"]
    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()
    return conditions, support, confirmation, complexity


def _solve_boundary_rule(
        examined_rule: pd.Series,
        other_rules: pd.DataFrame,
        user_ranking: pd.DataFrame,
        direction: Literal["highest", "lowest"] = "highest"
) -> int:
    rules = pd.concat([pd.DataFrame(examined_rule).T, other_rules], axis=0)

    all_conditions, all_support, all_confirmation, all_complexity = _rules_to_values(rules)
    ranking_conditions, ranking_support, ranking_confirmation, ranking_complexity = _rules_to_values(user_ranking)

    n_conditions = all_conditions.shape[1]
    n_other_rules = other_rules.shape[0]
    n_ranking_rules = user_ranking.shape[0]

    # PULP problem
    problem = pl.LpProblem("Minimize_Violations", pl.LpMinimize)

    # Define variables
    v_i = [pl.LpVariable(f"v_{i}", cat="Binary") for i in range(n_other_rules)]
    w_j = [pl.LpVariable(f"w_{j}", cat="Continuous", lowBound=0, upBound=1) for j in range(n_conditions)]
    w_supp = pl.LpVariable("w_supp", cat="Continuous", lowBound=0, upBound=1)
    w_confirm = pl.LpVariable("w_confirm", cat="Continuous", lowBound=0, upBound=1)
    w_complex = pl.LpVariable("w_complex", cat="Continuous", lowBound=0, upBound=1)

    ranking_trust = [
        pl.lpSum(w_j[j] * ranking_conditions[i, j] for j in range(n_conditions)) +
        w_supp * ranking_support[i] +
        w_confirm * ranking_confirmation[i] -
        w_complex * ranking_complexity[i]
        for i in range(n_ranking_rules)
    ]

    # Trust calculation definition
    all_trust = [
        pl.lpSum(w_j[j] * all_conditions[i, j] for j in range(n_conditions)) +
        w_supp * all_support[i] +
        w_confirm * all_confirmation[i] -
        w_complex * all_complexity[i]
        for i in range(n_other_rules + 1)
    ]

    # Constraints: Sum of weights
    problem += pl.lpSum(w_j) + w_supp + w_confirm + w_complex == 1, "Sum_of_Weights"

    # Constraints: User preferences
    for i in range(n_other_rules):
        if direction == "highest":
            problem += all_trust[0] >= all_trust[i + 1] + INEQ - M * v_i[i], f"Violation_{i}"
        elif direction == "lowest":
            problem += all_trust[i + 1] <= all_trust[0] + INEQ - M * v_i[i], f"Violation_{i}"

    for i in range(n_ranking_rules - 1):
        problem += ranking_trust[i] >= ranking_trust[i + 1] + INEQ, f"User_Preference_{i}"

    # Objective function
    problem += pl.lpSum(v_i), "Minimize_Violations"

    solver = pl.GUROBI(msg=False)

    if solver.solve(problem) != pl.LpStatusOptimal:
        raise Exception("Optimization problem did not converge")

    violations = problem.objective.value()

    if direction == "highest":
        return violations + 1
    elif direction == "lowest":
        return n_other_rules - violations


def _solve_range_rule(
        examined_rule: pd.Series,
        other_rules: pd.DataFrame,
        user_ranking: pd.DataFrame
) -> tuple[int, int]:
    return (
        _solve_boundary_rule(examined_rule, other_rules, user_ranking, direction="highest"),
        _solve_boundary_rule(examined_rule, other_rules, user_ranking, direction="lowest")
    )


def solve_extreme_ranking(rules: pd.DataFrame, user_preferences: list[int]) -> pd.DataFrame:
    rules = rules.copy()
    user_ranking = rules.iloc[user_preferences]
    result = []

    for i, rule in rules.iterrows():
        range_rule = _solve_range_rule(rule, rules.drop(i), user_ranking)
        result.append(range_rule)

    rules = pd.concat([rules, pd.DataFrame(result, index=rules.index, columns=["highest", "lowest"])], axis=1)
    rules = rules.sort_values(by=["highest", "lowest"], ascending=[True, True])
    return rules
