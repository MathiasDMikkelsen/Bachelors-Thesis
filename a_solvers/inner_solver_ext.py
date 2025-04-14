# inner_solver.py (Baseline Modified for Abatement)

import numpy as np
from scipy.optimize import root

# a. Baseline parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
t = 24.0
d0 = 0.5
epsilon_c = 0.995 # Labor share parameter for clean firm (top level)
epsilon_d = 0.92  # Labor share parameter for dirty firm (top level)
sigma = 0.5       # Elasticity sigma (top level, labor vs composite Z/A)
p_c = 1.0         # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
varphi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)      # Number of households/types

# b. New parameters for Abatement extension
# Calibrate p_a once offline, then use the value here       # <<<<<<<< USER: SET YOUR CALIBRATED ABATEMENT PRICE HERE >>>>>>>>
p_a = 15.0
varsigma = 2.0   # Elasticity varsigma (ς) between pollution (z) and abatement (a)
epsilon_z = 0.7   # Share parameter for pollution (z) in the z/a nest (must be 0 < eps_z < 1)

# c. Define rho values from elasticities for convenience
rho = (sigma - 1) / sigma
rho_za = (varsigma - 1) / varsigma # rho for z/a nest (ς)

# Ensure elasticities are valid
if sigma == 1.0: rho = -np.inf # Handle Cobb-Douglas case if needed (though rho = -1 here)
if varsigma == 1.0: rho_za = -np.inf # Handle Cobb-Douglas case if needed

def solve(tau_w, tau_z, g):
    
    """
    Solves the inner system for general equilibrium with abatement.
    tau_w: array of length n for proportional income tax rates
    tau_z: scalar environmental tax
    g: scalar government consumption
    p_a, varsigma, epsilon_z are read from global scope above
    """
    assert len(tau_w) == n, f"tau_w must have length n={n}"
    assert 0 < epsilon_z < 1, "epsilon_z must be between 0 and 1"

    def system_eqns(y):
        
        """Defines the system of 9 equilibrium equations."""
        # Unpack unknowns: Now includes log_a_c, log_a_d
        t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)
        a_c = np.exp(log_a_c)
        a_d = np.exp(log_a_d)

        # --- Input Validation ---
        if w <= 0 or p_d <= 0 or p_a <= 0 or tau_z <= 0: return np.ones(9) * 1e6
        if t_c <= 0 or t_d <= 0 or z_c <= 0 or z_d <= 0 or a_c <= 0 or a_d <= 0: return np.ones(9) * 1e6

        # --- Calculate Nested Production ---
        # Inner nest: Pollution/Abatement composite Y_j
        # Handle potential division by zero if rho_za is zero (varsigma=1) -> Cobb-Douglas
        if np.isinf(rho_za): # Cobb-Douglas case for Z/A nest
            Y_c = (z_c**epsilon_z) * (a_c**(1 - epsilon_z))
            Y_d = (z_d**epsilon_z) * (a_d**(1 - epsilon_z))
        else: # CES case for Z/A nest
             # Ensure base of power is positive for non-integer exponents if rho_za is tricky
             term_zc = epsilon_z * (z_c**rho_za)
             term_ac = (1 - epsilon_z) * (a_c**rho_za)
             term_zd = epsilon_z * (z_d**rho_za)
             term_ad = (1 - epsilon_z) * (a_d**rho_za)
             if term_zc <= 0 or term_ac <= 0 or term_zd <= 0 or term_ad <= 0: return np.ones(9) * 1e6 # Base must be > 0
             Y_c = (term_zc + term_ac)**(1/rho_za)
             Y_d = (term_zd + term_ad)**(1/rho_za)

        if Y_c <= 0 or Y_d <= 0: return np.ones(9) * 1e6 # Composite must be positive

        # Outer nest: Final Output f_j
        # Handle potential division by zero if rho is zero (sigma=1) -> Cobb-Douglas
        if np.isinf(rho): # Cobb-Douglas case for top level
            f_c = (t_c**epsilon_c) * (Y_c**(1 - epsilon_c))
            f_d = (t_d**epsilon_d) * (Y_d**(1 - epsilon_d))
        else: # CES case for top level
            term_tc = epsilon_c * (t_c**rho)
            term_Yc = (1 - epsilon_c) * (Y_c**rho)
            term_td = epsilon_d * (t_d**rho)
            term_Yd = (1 - epsilon_d) * (Y_d**rho)
            if term_tc <= 0 or term_Yc <= 0 or term_td <= 0 or term_Yd <= 0: return np.ones(9) * 1e6
            f_c = (term_tc + term_Yc)**(1/rho)
            f_d = (term_td + term_Yd)**(1/rho)

        if f_c <= 0 or f_d <= 0: return np.ones(9) * 1e6 # Output must be positive

        # --- Household Behavior (Unchanged) ---
        h_i = phi*w*(1-tau_w)*t + l +(np.exp(log_a_d)+np.exp(log_a_c))*varphi*p_a - p_d*d0
        if np.any(h_i <= 0): return np.ones(9) * 1e6
        d_agents = (beta/(p_d*(alpha+beta+gamma))) * h_i + d0
        leisure_denom = (alpha+beta+gamma)*(1-tau_w)*phi*w
        if np.any(leisure_denom <= 0): return np.ones(9) * 1e6
        l_agents = (gamma/leisure_denom) * h_i
        if np.any(l_agents <= 0) or np.any(l_agents >= t): return np.ones(9) * 1e6
        agg_labor = np.sum(phi*(t - l_agents))
        agg_d = np.sum(d_agents)

        # --- Equilibrium Equations (Now 9) ---
        # Eq1: Labor market clearing (unchanged structure)
        eq1 = t_c + t_d - agg_labor
        # Eq2: Dirty good market clearing (unchanged structure, uses new f_d)
        eq2 = agg_d + 0.5*g/p_d - f_d
        # Eq3: Firm c FOC for t_c: p_c * MPL_c = w (p_c=1)
        MPL_c = epsilon_c * (f_c / t_c)**(1-rho) # 1-rho = 1/sigma
        eq3 = p_c * MPL_c - w
        # Eq4: Firm c FOC for z_c: p_c * MPZ_c = tau_z (p_c=1)
        MPZ_c = (1-epsilon_c)*epsilon_z * (f_c/Y_c)**(1-rho) * (Y_c/z_c)**(1-rho_za)
        eq4 = p_c * MPZ_c - tau_z
        # Eq5: Firm d FOC for t_d: p_d * MPL_d = w
        MPL_d = epsilon_d * (f_d / t_d)**(1-rho)
        eq5 = p_d * MPL_d - w
        # Eq6: Firm d FOC for z_d: p_d * MPZ_d = tau_z
        MPZ_d = (1-epsilon_d)*epsilon_z * (f_d/Y_d)**(1-rho) * (Y_d/z_d)**(1-rho_za)
        eq6 = p_d * MPZ_d - tau_z
        # Eq7: Government budget constraint (unchanged structure)
        income_tax_revenue = np.sum(tau_w * w * phi * (t - l_agents))
        env_tax_revenue = tau_z * (z_c + z_d) # Tax is on NET pollution z
        eq7 = n*l - (income_tax_revenue + env_tax_revenue - g)
        # Eq8: Firm c FOC for a_c: p_c * MPA_c = p_a (p_c=1)
        MPA_c = (1-epsilon_c)*(1-epsilon_z) * (f_c/Y_c)**(1-rho) * (Y_c/a_c)**(1-rho_za)
        eq8 = p_c * MPA_c - p_a
        # Eq9: Firm d FOC for a_d: p_d * MPA_d = p_a
        MPA_d = (1-epsilon_d)*(1-epsilon_z) * (f_d/Y_d)**(1-rho) * (Y_d/a_d)**(1-rho_za)
        eq9 = p_d * MPA_d - p_a

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9])

    # --- Initial Guess (Now 9 variables) ---
    # Adjust guess based on expected magnitudes
    y0 = np.array([10.0, 10.0,  # t_c, t_d
                   np.log(1.0), np.log(1.0),  # log_z_c, log_z_d
                   np.log(0.5), np.log(0.5),  # log_a_c, log_a_d (guess abatement starts lower)
                   0.8, 1.5, 1.0])  # w, p_d, l

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'ftol': 1e-20})

    # Check for convergence
    if not sol.success:
        # print(f"Warning: Inner solver failed to converge. Message: {sol.message}") # Optional debug
        return None, None, False

    # Unpack solution (Now 9 variables)
    t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)
    a_c = np.exp(log_a_c)
    a_d = np.exp(log_a_d)

    # --- Recalculate quantities and checks (similar to baseline, but ensure validity) ---
    if w <= 0 or p_d <= 0 or t_c <= 0 or t_d <= 0 or z_c <= 0 or z_d <= 0 or a_c <= 0 or a_d <= 0:
         return None, None, False

    # Recalculate Y and f (using the solved values)
    if np.isinf(rho_za): Y_c = (z_c**epsilon_z) * (a_c**(1 - epsilon_z)); Y_d = (z_d**epsilon_z) * (a_d**(1 - epsilon_z))
    else: Y_c = (epsilon_z * (z_c**rho_za) + (1 - epsilon_z) * (a_c**rho_za))**(1/rho_za); Y_d = (epsilon_z * (z_d**rho_za) + (1 - epsilon_z) * (a_d**rho_za))**(1/rho_za)
    if Y_c <= 0 or Y_d <= 0: return None, None, False
    if np.isinf(rho): f_c = (t_c**epsilon_c) * (Y_c**(1 - epsilon_c)); f_d = (t_d**epsilon_d) * (Y_d**(1 - epsilon_d))
    else: f_c = (epsilon_c * (t_c**rho) + (1 - epsilon_c) * (Y_c**rho))**(1/rho); f_d = (epsilon_d * (t_d**rho) + (1 - epsilon_d) * (Y_d**rho))**(1/rho)
    if f_c <= 0 or f_d <= 0: return None, None, False

    h_i = phi*w*(1-tau_w)*t + l +(np.exp(log_a_d)+np.exp(log_a_c))*varphi*p_a - p_d*d0
    if np.any(h_i <= 0): return None, None, False
    c_agents = (alpha/(p_c*(alpha+beta+gamma))) * h_i # p_c = 1
    d_agents = (beta/(p_d*(alpha+beta+gamma))) * h_i + d0
    leisure_denom = (alpha+beta+gamma)*(1-tau_w)*phi*w
    if np.any(leisure_denom <= 0): return None, None, False
    l_agents = (gamma/leisure_denom) * h_i

    # Check validity for utility calc
    valid_c = np.all(c_agents > 0)
    valid_d = np.all(d_agents > d0)
    valid_l = np.all(l_agents > 0) and np.all(l_agents < t)
    utilities = np.full(n, -1e9) # Default to large negative
    if valid_c and valid_d and valid_l:
         utilities = (alpha * np.log(c_agents) +
                      beta * np.log(d_agents - d0) +
                      gamma * np.log(l_agents))

    # Aggregates
    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi*(t - l_agents))

    # Profits
    profit_c = p_c * f_c - w*t_c - tau_z*z_c - p_a*a_c # p_c=1
    profit_d = p_d * f_d - w*t_d - tau_z*z_d - p_a*a_d

    # Budgets
    budget_errors = np.zeros(n)
    for i in range(n):
        income = phi[i]*w*(1-tau_w[i])*(t - l_agents[i]) + l + (np.exp(log_a_d)+np.exp(log_a_c))*varphi[i]*p_a
        expenditure = p_c * c_agents[i] + p_d*d_agents[i] # p_c=1
        budget_errors[i] = income - expenditure

    # Assemble results
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "a_c": a_c, "a_d": a_d, # Added abatement
        "w": w, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities, # log(u_tilde_i)
        "sol": sol,
        "system_residuals": sol.fun
    }

    return sol.x, results, sol.success

# --- Example Run (Optional for testing this file directly) ---
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
        print(f"Solution vector [t_c,t_d,log_z_c,log_z_d,log_a_c,log_a_d,w,p_d,l]:\n{solution_vec}")
        # print("\nSelected Results:")
        # print(f"  w = {results_dict['w']:.4f}, p_d = {results_dict['p_d']:.4f}, l = {results_dict['l']:.4f}")
        # print(f"  Pollution z_c = {results_dict['z_c']:.4f}, z_d = {results_dict['z_d']:.4f}")
        # print(f"  Abatement a_c = {results_dict['a_c']:.4f}, a_d = {results_dict['a_d']:.4f}")
        # print("  Household log(blue utilities):", results_dict['utilities'])
        print(f"\nResiduals check (should be close to zero):")
        residuals = results_dict["system_residuals"]
        for idx, res_val in enumerate(residuals, start=1):
            print(f"  eq{idx} residual: {res_val:.6e}")
        print(f"\nProfits (should be near zero): Pr_c={results_dict['profit_c']:.4e}, Pr_d={results_dict['profit_d']:.4e}")
    else:
        print("\nInner solver FAILED to converge.")
        if results_dict is not None and 'sol' in results_dict:
             print(f"Solver message: {results_dict['sol'].message}")

    print("--- End of direct test ---")