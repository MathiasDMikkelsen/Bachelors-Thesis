import numpy as np
from scipy.optimize import root

# a. Baseline parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
t = 24.0
d0 = 0.5
epsilon_c = 0.995  # Labor share parameter for clean firm
epsilon_d = 0.92   # Labor share parameter for dirty firm
sigma = 0.5        # Elasticity sigma (labor vs composite)
p_c = 1.0          # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
varphi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

# b. Abatement extension parameters
# SET YOUR CALIBRATED ABATEMENT PRICE HERE
p_a = 15.0        # Abatement price
varsigma = 2.0    # Elasticity between pollution and abatement
epsilon_z = 0.7  # Share parameter in Z/A nest

# c. Rho definitions
a = (sigma - 1) / sigma         # rho for top-level CES
b = (varsigma - 1) / varsigma    # rho for inner Z/A nest
# Handle Cobb-Douglas cases
if sigma == 1.0: a = -np.inf
if varsigma == 1.0: b = -np.inf


def solve(tau_w, tau_z, g):
    """
    Solve general equilibrium with abatement.
    tau_w: array of income tax rates (length n)
    tau_z: scalar environmental tax
    g:     government consumption
    Returns: solution vector, results dict, success flag
    """
    assert len(tau_w) == n, f"tau_w must have length n={n}"
    assert 0 < epsilon_z < 1, "epsilon_z must be between 0 and 1"

    def system_eqns(y):
        # Unpack unknowns
        t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l = y
        z_c, z_d = np.exp(log_z_c), np.exp(log_z_d)
        a_c, a_d = np.exp(log_a_c), np.exp(log_a_d)

        # Feasibility checks
        if min(t_c, t_d, z_c, z_d, a_c, a_d, w, p_d, tau_z) <= 0:
            return np.ones(9) * 1e6

        # --- Inner composite Z/A ---
        if np.isinf(b):
            inner_c = z_c**epsilon_z * a_c**(1 - epsilon_z)
            inner_d = z_d**epsilon_z * a_d**(1 - epsilon_z)
        else:
            base_c = epsilon_z * z_c**b + (1 - epsilon_z) * a_c**b
            base_d = epsilon_z * z_d**b + (1 - epsilon_z) * a_d**b
            if base_c <= 0 or base_d <= 0:
                return np.ones(9) * 1e6
            inner_c = base_c**(1/b)
            inner_d = base_d**(1/b)

        # --- Top-level CES f_j ---
        if np.isinf(a):
            f_c = t_c**epsilon_c * inner_c**(1 - epsilon_c)
            f_d = t_d**epsilon_d * inner_d**(1 - epsilon_d)
        else:
            base_fc = epsilon_c * t_c**a + (1 - epsilon_c) * inner_c**a
            base_fd = epsilon_d * t_d**a + (1 - epsilon_d) * inner_d**a
            if base_fc <= 0 or base_fd <= 0:
                return np.ones(9) * 1e6
            f_c = base_fc**(1/a)
            f_d = base_fd**(1/a)

        # --- Firm FOCs ---
        MPL_c = epsilon_c * (f_c / t_c)**(1 - a)
        MPL_d = epsilon_d * (f_d / t_d)**(1 - a)

        MPZ_c = (1 - epsilon_c) * epsilon_z \
                * (f_c / inner_c)**(1 - a) * (inner_c / z_c)**(1 - b)
        MPZ_d = (1 - epsilon_d) * epsilon_z \
                * (f_d / inner_d)**(1 - a) * (inner_d / z_d)**(1 - b)

        MPA_c = (1 - epsilon_c) * (1 - epsilon_z) \
                * (f_c / inner_c)**(1 - a) * (inner_c / a_c)**(1 - b)
        MPA_d = (1 - epsilon_d) * (1 - epsilon_z) \
                * (f_d / inner_d)**(1 - a) * (inner_d / a_d)**(1 - b)

        # --- Household behavior ---
        h_i = phi * w * (1 - tau_w) * t+(1-tau_w)*(a_d + a_c) * varphi * p_a + l - p_d * d0
        if np.any(h_i <= 0):
            return np.ones(9) * 1e6
        d_i = (beta / (p_d * (alpha + beta + gamma))) * h_i + d0
        leisure_denom = (alpha + beta + gamma) * (1 - tau_w) * phi * w
        if np.any(leisure_denom <= 0):
            return np.ones(9) * 1e6
        l_i = (gamma / leisure_denom) * h_i
        if np.any(l_i <= 0) or np.any(l_i >= t):
            return np.ones(9) * 1e6

        agg_labor = np.sum(phi * (t - l_i))
        agg_d = np.sum(d_i)

        # --- Equilibrium equations ---
        eq1 = t_c + t_d - agg_labor
        eq2 = agg_d + 0.5 * g / p_d - f_d
        eq3 = p_c * MPL_c - w
        eq4 = p_c * MPZ_c - tau_z
        eq5 = p_d * MPL_d - w
        eq6 = p_d * MPZ_d - tau_z
        income_tax_revenue = np.sum(tau_w * w * phi * (t - l_i)+tau_w*(a_d + a_c) * varphi * p_a)
        env_tax_revenue = tau_z * (z_c + z_d)
        eq7 = n * l - (income_tax_revenue + env_tax_revenue - g)
        eq8 = p_c * MPA_c - p_a
        eq9 = p_d * MPA_d - p_a

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9])

    # Initial guess
    y0 = np.array([10.0, 10.0,
                   np.log(1.0), np.log(1.0),
                   np.log(0.5), np.log(0.5),
                   0.8, 1.5, 1.0])

    sol = root(system_eqns, y0, method='lm', options={'ftol':1e-20})
    if not sol.success:
        return None, None, False

    # Unpack
    t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l = sol.x
    z_c, z_d = np.exp(log_z_c), np.exp(log_z_d)
    a_c, a_d = np.exp(log_a_c), np.exp(log_a_d)

    # Recompute composites and output for results
    if np.isinf(b):
        inner_c = z_c**epsilon_z * a_c**(1 - epsilon_z)
        inner_d = z_d**epsilon_z * a_d**(1 - epsilon_z)
    else:
        inner_c = (epsilon_z * z_c**b + (1 - epsilon_z) * a_c**b)**(1/b)
        inner_d = (epsilon_z * z_d**b + (1 - epsilon_z) * a_d**b)**(1/b)
    if np.isinf(a):
        f_c = t_c**epsilon_c * inner_c**(1 - epsilon_c)
        f_d = t_d**epsilon_d * inner_d**(1 - epsilon_d)
    else:
        f_c = (epsilon_c * t_c**a + (1 - epsilon_c) * inner_c**a)**(1/a)
        f_d = (epsilon_d * t_d**a + (1 - epsilon_d) * inner_d**a)**(1/a)

    # Households again for results
    h_i = phi * w * (1 - tau_w) * t + l + (1-tau_w)*(a_d + a_c) * varphi * p_a - p_d * d0
    d_i = (beta / (p_d * (alpha + beta + gamma))) * h_i + d0
    l_i = (gamma / ((alpha + beta + gamma) * (1 - tau_w) * phi * w)) * h_i

    # Aggregates
    agg_c = np.sum((alpha / (p_c * (alpha + beta + gamma))) * h_i)
    agg_d = np.sum(d_i)
    agg_labor = np.sum(phi * (t - l_i))

    # Profits
    profit_c = p_c * f_c - w * t_c - tau_z * z_c - p_a * a_c
    profit_d = p_d * f_d - w * t_d - tau_z * z_d - p_a * a_d

    # Utilities & budget errors
    utilities = np.empty(n)
    budget_errors = np.empty(n)
    for i in range(n):
        c_i = (alpha / (p_c * (alpha + beta + gamma))) * h_i[i]
        utilities[i] = alpha * np.log(c_i) +\
                       beta * np.log(d_i[i] - d0) +\
                       gamma * np.log(l_i[i])
        income_i = phi[i] * w * (1 - tau_w[i]) * (t - l_i[i])+ (1-tau_w[i])*(a_d + a_c) * varphi[i] * p_a + l
        expend_i = p_c * c_i + p_d * d_i[i]
        budget_errors[i] = income_i - expend_i

    results = {
        't_c': t_c, 't_d': t_d, 'z_c': z_c, 'z_d': z_d, 'a_c': a_c, 'a_d': a_d,
        'w': w, 'p_d': p_d, 'l': l,
        'f_c': f_c, 'f_d': f_d,
        'c_agents': (alpha / (p_c * (alpha + beta + gamma))) * h_i,
        'd_agents': d_i, 'l_agents': l_i,
        'agg_c': agg_c, 'agg_d': agg_d, 'agg_labor': agg_labor,
        'profit_c': profit_c, 'profit_d': profit_d,
        'budget_errors': budget_errors,
        'utilities': utilities,
        'system_residuals': sol.fun
    }

    return sol.x, results, True


# --- Example Run ---
if __name__ == "__main__":
    print("--- Running direct test of inner_solver.py (Baseline + Abatement) ---")
    test_tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
    test_tau_z = 3.0
    test_g = 5.0
    print(f"Using fixed parameters: p_a={p_a}, varsigma={varsigma}, epsilon_z={epsilon_z}")
    print(f"Testing with tau_w = {test_tau_w}, tau_z = {test_tau_z}, G = {test_g}")

    solution_vec, results_dict, converged_flag = solve(test_tau_w, test_tau_z, test_g)

    if converged_flag:
        print("\nInner solver converged successfully.")
        print(f"Solution vector [t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l]:\n{solution_vec}")
        print("\nResiduals check (should be close to zero):")
        residuals = results_dict['system_residuals']
        for idx, val in enumerate(residuals, start=1):
            print(f"  eq{idx} residual: {val:.6e}")
        print(f"\nProfits (should be near zero): Pr_c={results_dict['profit_c']:.4e}, Pr_d={results_dict['profit_d']:.4e}")
    else:
        print("\nInner solver FAILED to converge.")
        if results_dict is not None:
            print(f"Solver message: {results_dict.get('sol', 'No additional info')}")
    print("--- End of direct test ---")