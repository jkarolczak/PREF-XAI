import numpy as np
import pandas as pd
import pulp as pl

from lib.optimization.trust import Trust

EPS = 1e-6


def solve_hr(rules: pd.DataFrame, user_preferences: list[int], n_iters: int = 1000) -> list[Trust]:
    rules = rules.copy()
    trust = solve_initial(rules, user_preferences)
    trust_history = [trust]
    trust_vec = trust.to_np()
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["confidence", "coverage(y)"]
    rules = rules.drop(columns=special_columns)
    rules_preferences = rules.iloc[user_preferences]
    rules_np = rules_preferences.values

    n_rules = rules_np.shape[0]
    n_attributes = rules_np.shape[1]
    n_independent_w = n_attributes - 1
    n_random_draws = n_attributes - 2

    preference = np.zeros((n_rules, n_rules))
    for i in range(n_rules - 1):
        preference[i, i + 1] = 1

    draws = np.zeros(n_attributes)
    rank = np.zeros(n_attributes, dtype=int)
    values = np.zeros(n_attributes)
    alfa = np.zeros(n_attributes)
    trust_vec_temp = np.zeros(n_attributes)

    for q in range(n_iters):
        draws[:n_random_draws] = np.random.rand(n_random_draws)

        rank[:n_random_draws] = np.sum(draws[:n_random_draws, None] >= draws[:n_random_draws], axis=1) - 1
        for j in range(n_random_draws):
            values[rank[j]] = draws[j]

        alfa[0] = values[0]
        alfa[1:n_random_draws] = values[1:n_random_draws] - values[0:n_random_draws - 1]
        alfa[n_random_draws] = 1.0 - values[n_random_draws - 1]

        for j in range(n_independent_w):
            sqrt_val = np.sqrt(np.abs(alfa[j]))
            alfa[j] = sqrt_val if np.random.rand() > 0.5 else -sqrt_val

        lambda_ = 1e6
        alfa_total = np.sum(alfa[:n_independent_w])

        for i in range(n_independent_w):
            if alfa[i] < 0:
                candidate = -trust_vec[i] / alfa[i]
                if lambda_ > candidate > 0:
                    lambda_ = candidate

        if alfa_total > 0:
            ro = trust_vec[n_attributes - 1] / alfa_total
            if ro < lambda_:
                lambda_ = ro

        for t in range(n_rules):
            for r in range(n_rules):
                if preference[t, r] != 0:
                    diff = np.sum(alfa[:n_independent_w] * (rules_np[t, :n_independent_w] - rules_np[r, :n_independent_w]))
                    diff += alfa_total * (rules_np[t, n_attributes - 1] - rules_np[r, n_attributes - 1])

                    diffnum = np.sum(
                        trust_vec[:n_independent_w] * (rules_np[t, :n_independent_w] - rules_np[r, :n_independent_w]))
                    diffnum -= trust_vec[n_attributes - 1] * (rules_np[t, n_attributes - 1] - rules_np[r, n_attributes - 1])

                    if diff < 0:
                        candidate = -diffnum / diff

                        if candidate < lambda_:
                            lambda_ = candidate

        trust_vec_temp[:n_independent_w] = trust_vec[:n_independent_w] + lambda_ * alfa[:n_independent_w]
        trust_vec_temp[n_attributes - 1] = 1.0 - np.sum(trust_vec_temp[:n_independent_w])
        trust_vec_temp = np.maximum(trust_vec_temp, 0)
        trust_vec_temp /= np.sum(trust_vec_temp)

        b = np.random.rand()
        trust_vec = b * trust_vec + (1 - b) * trust_vec_temp

        trust_history.append(Trust(rules,
                                   trust_vec[:-3].tolist(),
                                   trust_vec[-3].item(),
                                   trust_vec[-2].item(),
                                   trust_vec[-1].item()))

    return trust_history


def solve_initial(rules: pd.DataFrame, user_preferences: list[int]) -> Trust:
    special_columns = [col for col in rules.columns if col.startswith("y_")] + ["support", "confirmation", "complexity",
                                                                                "confidence", "coverage(y)"]

    # Extract user preferences only
    rules = rules.copy()
    rules = rules.iloc[user_preferences, :]

    support = rules["support"].to_numpy()
    confirmation = rules["confirmation"].to_numpy()
    complexity = rules["complexity"].to_numpy()

    conditions = rules[[col for col in rules.columns if col not in special_columns]].to_numpy()
    n_conditions = conditions.shape[1]

    # PULP problem
    problem = pl.LpProblem("Maximize_Epsilon", pl.LpMaximize)

    # Define variables
    w_j = [pl.LpVariable(f"w_j_{i}", lowBound=0, upBound=1) for i in range(n_conditions)]
    w_supp = pl.LpVariable("w_supp", lowBound=0, upBound=1)
    w_confirm = pl.LpVariable("w_confirm", lowBound=0, upBound=1)
    w_complex = pl.LpVariable("w_complex", lowBound=0, upBound=1)

    # Trust calculation definition
    trust = [
        pl.lpSum(w_j[i] * conditions[j, i] for i in range(n_conditions)) +
        w_supp * support[j] +
        w_confirm * confirmation[j] -
        w_complex * complexity[j]
        for j in range(len(user_preferences))
    ]

    # Constraints: Sum of weights
    problem += pl.lpSum(w_j) + w_supp + w_confirm + w_complex == 1, "Sum_of_Weights"

    # Constraints: User preferences
    for i in range(len(user_preferences) - 1):
        problem += trust[i] >= trust[i + 1] + EPS, f"Preference_{i}"

    # Constraints: Maximize w_j to ensure being within the hypervolume
    epsilon = pl.LpVariable("epsilon", lowBound=0, upBound=1)
    for i in range(n_conditions):
        problem += w_j[i] >= epsilon, f"Maximize_w_{i}"
    problem += w_supp >= epsilon, "Maximize_w_supp"
    problem += w_confirm >= epsilon, "Maximize_w_confirm"
    problem += w_complex >= epsilon, "Maximize_w_complex"

    # Objective function
    problem += epsilon, "Maximize_Epsilon"

    # Solve the problem
    if pl.GUROBI(msg=False).solve(problem) != 1:
        raise Exception("Optimization problem did not converge")

    return Trust(
        rules,
        [w_j[i].varValue for i in range(n_conditions)],
        w_supp.varValue,
        w_confirm.varValue,
        w_complex.varValue
    )
