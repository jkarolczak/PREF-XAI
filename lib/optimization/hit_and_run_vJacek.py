import numpy as np
import pandas as pd
import pulp as pl

from lib.optimization.trust import Trust


def solve_initial(rules: pd.DataFrame, special_columns: list[str]) -> Trust:
    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()

    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    num_conditions = conditions.shape[1]

    # PULP problem (now feasibility)
    problem = pl.LpProblem("Feasible_Epsilon", pl.LpMinimize)

    # Variables
    w_j = [pl.LpVariable(f"w_j_{i}", lowBound=0, upBound=1) for i in range(num_conditions)]
    w_supp = pl.LpVariable("w_supp", lowBound=0, upBound=1)
    w_confirm = pl.LpVariable("w_confirm", lowBound=0, upBound=1)
    w_complex = pl.LpVariable("w_complex", lowBound=0, upBound=1)

    # Trust values
    trust = [
        pl.lpSum(w_j[i] * conditions[j, i] for i in range(num_conditions)) +
        w_supp * support[j] +
        w_confirm * confirmation[j] -
        w_complex * complexity[j]
        for j in range(len(rules))
    ]

    # Constraint: weights sum to 1
    problem += pl.lpSum(w_j) + w_supp + w_confirm + w_complex == 1, "Sum_of_Weights"

    # Epsilon constraint: just needs to be strictly positive
    epsilon = pl.LpVariable("epsilon", lowBound=0.0001, upBound=1)  # or small positive lower bound
    for i in range(len(rules) - 1):
        problem += trust[i] >= trust[i + 1] + epsilon, f"Preference_{i}"

    # Dummy objective: could also be just `pl.lpSum(0)` if you want
    problem += 0, "Dummy_Objective"

    # Solve feasibility
    pl.GUROBI(msg=False).solve(problem)

    if problem.status != 1:
        raise Exception("No feasible solution found")

    return Trust(
        rules,
        [w_j[i].varValue for i in range(num_conditions)],
        w_supp.varValue,
        w_confirm.varValue,
        w_complex.varValue
    )


def solve_simplex_boundary(origin: np.ndarray, direction: np.ndarray, rules: pd.DataFrame,
                           special_columns: list[str]) -> Trust:
    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()

    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    n_rules, n_conditions = conditions.shape

    # PULP problem
    problem = pl.LpProblem("Maximize_C", pl.LpMaximize)

    # Define variables
    c = pl.LpVariable("c", lowBound=0)

    # Trust calculation definition
    trust = [
        pl.lpSum(
            conditions[j, i] * (origin[i] + c * direction[i])
            for i in range(n_conditions)
        ) +
        support[j] * (origin[-3] + c * direction[-3]) +
        confirmation[j] * (origin[-2] + c * direction[-2]) -
        complexity[j] * (origin[-1] + c * direction[-1])
        for j in range(n_rules)
    ]

    # Constraints: Sum of weights
    problem += (
            pl.lpSum(origin[i] + c * direction[i] for i in range(n_conditions)) +
            origin[-3] + c * direction[-3] +
            origin[-2] + c * direction[-2] +
            origin[-1] + c * direction[-1] == 1
    ), "Sum_of_Weights"

    # Constraints: User preferences
    for i in range(n_rules - 1):
        problem += trust[i] >= trust[i + 1], f"Preference_{i}"

    # Objective function
    problem += c, "Maximize_C"

    # Solve the problem
    pl.GUROBI(msg=False).solve(problem)

    if problem.status != 1:
        print(problem)
        raise Exception("Optimization problem did not converge")

    return Trust(
        rules,
        [
            origin[i] + direction[i] * c.varValue for i in range(n_conditions)
        ],
        origin[-3] + direction[-3] * c.varValue,  # w_supp
        origin[-2] + direction[-2] * c.varValue,  # w_confirm
        origin[-1] + direction[-1] * c.varValue  # w_complex
    )


def solve_hit_and_run(rules: pd.DataFrame, user_preferences: list[int], n_trials: int = 100):
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["support", "confirmation", "complexity",
                                                                                "confidence", "coverage(y)"]
    rules = rules.copy()
    rules = rules.iloc[user_preferences, :]
    trust = solve_initial(rules, special_columns=special_columns)
    direction = np.random.randn(len(rules.columns) - len(special_columns))
    direction /= np.linalg.norm(direction)
    history = []

    print(trust)
    for _ in range(n_trials):
        origin = trust.to_np()
        next_trust = solve_simplex_boundary(origin=origin, direction=direction, rules=rules,
                                            special_columns=special_columns)
        history.append(next_trust)
        print(_, next_trust)
        # vector with origin between initial and current trust, and direction orthogonal to that vector
        origin = trust.to_np()
        edge = next_trust.to_np()
        trust = next_trust

        p = np.random.rand()
        origin = p * origin + (1 - p) * edge
        # find orthogonal direction to the vector between initial and current trust
        direction = edge - origin
        direction /= np.linalg.norm(direction)
        direction[np.random.randint(0, len(direction))] *= -1

    print(history)
