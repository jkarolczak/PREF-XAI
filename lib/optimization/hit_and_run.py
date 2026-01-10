import numpy as np
import pandas as pd
import pulp as pl

from lib.dms.interface_dm import IDM
from lib.optimization.trust import Trust

EPS = 1e-6


def solve_hr(
        rules: pd.DataFrame,
        dm: IDM,
        top_k: int | None = None,
        n_iters: int = 10000
) -> list[Trust]:
    initial_trust = solve_initial(rules, dm, top_k=top_k)
    special_columns = [col for col in rules.columns if col.startswith("y_")] + \
                      ["support", "confirmation", "complexity", "confidence", "coverage(y)"]
    condition_cols = [col for col in rules.columns if col not in special_columns]
    n_total_conditions = len(condition_cols)

    if top_k is not None:
        raw_indices = dm.attributes_preference[:top_k]
        active_indices = {i for i in raw_indices if i < n_total_conditions}
    else:
        active_indices = set(range(len(condition_cols)))
    active_map = sorted(active_indices)
    rules_subset = rules.iloc[dm.reference_ranking].copy()
    active_col_names = [condition_cols[i] for i in active_map]
    cond_vals = rules_subset[active_col_names].to_numpy()
    supp_vals = rules_subset["support"].to_numpy().reshape(-1, 1)
    conf_vals = rules_subset["confirmation"].to_numpy().reshape(-1, 1)
    main_attrs = np.hstack([cond_vals, supp_vals, conf_vals])
    comp_vals = rules_subset["complexity"].to_numpy()

    n_main = main_attrs.shape[1]
    full_init_w = initial_trust.to_np()
    w_active_indices = active_map
    w_fixed_indices = [n_total_conditions, n_total_conditions + 1, n_total_conditions + 2]
    w = full_init_w[w_active_indices + w_fixed_indices]

    history = [initial_trust]

    for _ in range(n_iters):
        estr = np.random.rand(n_main - 1)
        estr.sort()

        alfa = np.zeros(n_main)
        alfa[0] = estr[0]

        if len(estr) > 1:
            alfa[1:n_main - 1] = estr[1:] - estr[:-1]

        alfa[n_main - 1] = 1.0 - estr[-1]

        signs = np.where(np.random.rand(n_main) > 0.5, 1.0, -1.0)
        alfa = signs * np.sqrt(np.abs(alfa))

        lam = 1e6
        alfat = np.sum(alfa)

        neg_mask = alfa < -EPS
        if np.any(neg_mask):
            bound = -w[:n_main][neg_mask] / alfa[neg_mask]
            lam = min(lam, np.min(bound))

        if alfat > EPS:
            lam = min(lam, w[n_main] / alfat)

        for i in range(len(dm.reference_ranking) - 1):
            t_main = main_attrs[i]
            r_main = main_attrs[i + 1]
            t_comp = comp_vals[i]
            r_comp = comp_vals[i + 1]
            delta_main = t_main - r_main
            delta_comp = t_comp - r_comp
            slope = np.dot(alfa, delta_main) + alfat * delta_comp
            curr_margin = np.dot(w[:n_main], delta_main) - w[n_main] * delta_comp

            if slope < -EPS:
                limit = -curr_margin / slope
                if lam > limit:
                    lam = limit

        wf_main = w[:n_main] + lam * alfa
        wf_last = 1.0 - np.sum(wf_main)
        wf = np.append(wf_main, wf_last)

        b = np.random.rand()
        w = b * w + (1.0 - b) * wf

        w_j_values = [0.0] * n_total_conditions
        num_active = len(active_map)
        current_active_weights = w[:num_active]

        for idx, val in zip(active_map, current_active_weights):
            w_j_values[idx] = val

        w_supp = w[-3].item()
        w_conf = w[-2].item()
        w_comp = w[-1].item()

        history.append(Trust(rules, w_j_values, w_supp, w_conf, w_comp))

    return history


def solve_initial(
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
        raw_indices = dm.attributes_preference[:top_k]
        active_indices = {i for i in raw_indices if i < n_conditions}
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
    for i in range(len(dm.reference_ranking) - 1):
        problem += trust[i] >= trust[i + 1] + EPS, f"Preference_{i}"

    # Constraints: Maximize w_j to ensure being within the hypervolume
    epsilon = pl.LpVariable("epsilon", lowBound=0, upBound=1)
    for i in active_indices:
        problem += w_j[i] >= epsilon, f"Maximize_w_{i}"
    problem += w_supp >= epsilon, "Maximize_w_supp"
    problem += w_confirm >= epsilon, "Maximize_w_confirm"
    problem += w_complex >= epsilon, "Maximize_w_complex"

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


def trusts_to_centroid(trusts: list[Trust]) -> Trust:
    arrays = [t.to_np() for t in trusts]
    centroid_array = np.mean(arrays, axis=0)
    return Trust.from_np(centroid_array, trusts[0].rules)


def rules_on_first_pos(trusts: list[Trust], rules: pd.DataFrame) -> set[int]:
    if not trusts or rules.empty:
        return set()

    special_columns = [col for col in rules.columns if col.startswith("y_")] + \
                      ["support", "confirmation", "complexity", "confidence", "coverage(y)"]
    condition_cols = [col for col in rules.columns if col not in special_columns]

    cond_vals = rules[condition_cols].to_numpy(dtype=float)
    supp_vals = rules["support"].to_numpy(dtype=float).reshape(-1, 1)
    conf_vals = rules["confirmation"].to_numpy(dtype=float).reshape(-1, 1)
    comp_vals = -1.0 * rules["complexity"].to_numpy(dtype=float).reshape(-1, 1)

    rules_matrix = np.hstack([cond_vals, supp_vals, conf_vals, comp_vals])

    w_list = []
    for t in trusts:
        row = t.to_np()
        w_list.append(row)

    weights = np.array(w_list)
    scores = rules_matrix @ weights.T
    max_scores = scores.max(axis=0)
    is_winner = scores >= (max_scores - EPS)
    rule_wins_any = is_winner.any(axis=1)

    return set(rules.index[rule_wins_any].tolist())
