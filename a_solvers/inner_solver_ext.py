import numpy as np
from scipy.optimize import root
import warnings

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r_c = -1.0 # Example for clean sector
r_d = -1.0 # Example for dirty sector
t = 24.0
d0 = 0.5
epsilon_c = 0.9
epsilon_d = 0.45
p_c = 1.0 # Numeraire

# --- MODIFIED: Define parameters for 2 households (n=2) ---
n = 2
n_d = 1 # Number of dirty sector households
n_c = 1 # Number of clean sector households
# --- REMOVED: phi parameter definitions ---
# phi_d_val = 1.0
# phi_c_val = 1.0
# phi = np.array([phi_d_val, phi_c_val])
# --- Productivity (phi) is implicitly 1 for both households ---
# -----------------------------------------------------------

def solve(tau_w, tau_z, g):
    """
    Solves the general equilibrium for the 2-household (phi=1), 2-sector model.

    Args:
        tau_w (np.array): Array of wage tax rates, length 2 [tau_w_d, tau_w_c].
        tau_z (float): Tax rate on polluting input z.
        g (float): Government consumption requirement.

    Returns:
        tuple: (solution_vector, results_dict, convergence_flag)
    """
    if len(tau_w) != n:
        raise ValueError(f"Length of tau_w ({len(tau_w)}) must be {n}")
    tau_w_d = tau_w[0] # Tax for dirty worker
    tau_w_c = tau_w[1] # Tax for clean worker

    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y

        try:
            z_c = np.exp(log_z_c)
            z_d = np.exp(log_z_d)
        except OverflowError:
            return np.full(8, 1e10)

        t_c_safe = t_c + 1e-9
        t_d_safe = t_d + 1e-9
        z_c_safe = z_c + 1e-9
        z_d_safe = z_d + 1e-9
        p_d_safe = p_d + 1e-9
        w_c_safe = w_c + 1e-9
        w_d_safe = w_d + 1e-9

        # --- Firm side ---
        try:
            if isinstance(r_c, complex) or isinstance(r_d, complex):
                 raise ValueError("r values cannot be complex")

            f_c = (epsilon_c * (t_c_safe**r_c) + (1 - epsilon_c) * (z_c_safe**r_c))**(1/r_c)
            f_d = (epsilon_d * (t_d_safe**r_d) + (1 - epsilon_d) * (z_d_safe**r_d))**(1/r_d)

            if isinstance(f_c, complex) or isinstance(f_d, complex):
                 return np.full(8, 1e9)

        except (ValueError, OverflowError, ZeroDivisionError):
             return np.full(8, 1e8)

        # --- Household side (phi=1 for both) ---
        # Calculate "full income potential net of committed expenditure" (Y_tilde)
        # --- MODIFIED: Removed phi_d and phi_c (implicitly 1) ---
        Y_tilde_d = w_d_safe * (1 - tau_w_d) * t + l - p_d_safe * d0
        Y_tilde_c = w_c_safe * (1 - tau_w_c) * t + l - p_d_safe * d0
        # -------------------------------------------------------

        if Y_tilde_d <= 1e-9 or Y_tilde_c <= 1e-9:
            return np.full(8, 1e7)

        denom_shares = alpha + beta + gamma

        # Demand for dirty good
        d_agent_d = (beta / (p_d_safe * denom_shares)) * Y_tilde_d + d0
        d_agent_c = (beta / (p_d_safe * denom_shares)) * Y_tilde_c + d0

        # Demand for leisure
        # --- MODIFIED: Removed phi_d and phi_c from price_leisure ---
        price_leisure_d = w_d_safe * (1 - tau_w_d)
        price_leisure_c = w_c_safe * (1 - tau_w_c)
        # ----------------------------------------------------------
        # Check if net wage is positive before dividing
        if price_leisure_d <= 1e-9 or price_leisure_c <= 1e-9:
             return np.full(8, 1e6) # Cannot calculate leisure if net wage is non-positive

        l_agent_d = (gamma / (denom_shares * price_leisure_d)) * Y_tilde_d
        l_agent_c = (gamma / (denom_shares * price_leisure_c)) * Y_tilde_c

        l_agent_d = np.clip(l_agent_d, 1e-9, t - 1e-9)
        l_agent_c = np.clip(l_agent_c, 1e-9, t - 1e-9)

        # Labor supply (household time units = effective labor units since phi=1)
        labor_supply_d = t - l_agent_d
        labor_supply_c = t - l_agent_c

        # --- System of Equations (8 equations) ---
        try:
            # --- MODIFIED: Labor market clearing uses labor_supply directly ---
            # Eq1: Clean labor market: demand (t_c) = supply (labor_supply_c)
            eq1 = t_c - labor_supply_c
            # Eq2: Dirty labor market: demand (t_d) = supply (labor_supply_d)
            eq2 = t_d - labor_supply_d
            # -------------------------------------------------------------------

            # Eq3: Dirty good market clearing
            agg_d = d_agent_d + d_agent_c
            eq3 = (agg_d + 0.5 * g / p_d_safe) - f_d

            # Eq4: Firm C FOC Labor
            MP_L_c = epsilon_c * (t_c_safe**(r_c - 1)) * (f_c**(1 - r_c))
            eq4 = w_c - MP_L_c

            # Eq5: Firm C FOC Z
            MP_Z_c = (1 - epsilon_c) * (z_c_safe**(r_c - 1)) * (f_c**(1 - r_c))
            eq5 = tau_z - MP_Z_c

            # Eq6: Firm D FOC Labor
            MP_L_d = epsilon_d * (t_d_safe**(r_d - 1)) * (f_d**(1 - r_d))
            eq6 = w_d - MP_L_d * p_d

            # Eq7: Firm D FOC Z
            MP_Z_d = (1 - epsilon_d) * (z_d_safe**(r_d - 1)) * (f_d**(1 - r_d))
            eq7 = tau_z - MP_Z_d * p_d

            if any(np.isnan([eq4, eq5, eq6, eq7])) or any(np.isinf([eq4, eq5, eq6, eq7])):
                return np.full(8, 1e6)

            # Eq8: Government budget constraint
            # --- MODIFIED: wage_tax_rev uses labor_supply directly ---
            wage_tax_rev = tau_w_d * w_d_safe * labor_supply_d + tau_w_c * w_c_safe * labor_supply_c
            # ------------------------------------------------------
            z_tax_rev = tau_z * (z_c + z_d)
            gov_spending = g + n * l # n=2
            eq8 = (wage_tax_rev + z_tax_rev) - gov_spending

            residuals = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])
            if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                 return np.full(8, 1e5)
            return residuals

        except (ValueError, OverflowError, ZeroDivisionError):
             return np.full(8, 1e4)


    # Initial guess for 8 variables
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.7, 0.6, 1.5, 0.1])

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing ---
    if not sol.success:
        warnings.warn(f"Solver did not converge! Message: {sol.message}", RuntimeWarning)
        # Return NaNs (structure matches results dict)
        nan_results = {key: np.nan for key in [
            "t_c", "t_d", "z_c", "z_d", "w_c", "w_d", "p_d", "l",
            "f_c", "f_d", "c_agents", "d_agents", "l_agents", "labor_supply_agents",
            "agg_c", "agg_d", "agg_labor_c", "agg_labor_d", "agg_labor_total",
            "profit_c", "profit_d", "budget_errors", "utilities",
            "wage_vector", "sol_object", "system_residuals"]}
        nan_results["sol_object"] = sol
        nan_results["system_residuals"] = np.full(8, np.nan)
        for key in ["c_agents", "d_agents", "l_agents", "labor_supply_agents", "budget_errors", "utilities", "wage_vector"]:
             nan_results[key] = np.full(n, np.nan) # n=2
        return np.full(8, np.nan), nan_results, False


    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    # Recalculate production
    f_c = (epsilon_c * (t_c**r_c) + (1 - epsilon_c) * (z_c**r_c))**(1/r_c)
    f_d = (epsilon_d * (t_d**r_d) + (1 - epsilon_d) * (z_d**r_d))**(1/r_d)

    wage_vector = np.array([w_d, w_c]) # Order [dirty, clean]

    # Recalculate household variables (phi=1)
    tau_w_d = tau_w[0]
    tau_w_c = tau_w[1]

    # --- MODIFIED: Removed phi from calculations ---
    Y_tilde_d = w_d * (1 - tau_w_d) * t + l - p_d * d0
    Y_tilde_c = w_c * (1 - tau_w_c) * t + l - p_d * d0
    denom_shares = alpha + beta + gamma

    c_agent_d = (alpha / (p_c * denom_shares)) * Y_tilde_d
    c_agent_c = (alpha / (p_c * denom_shares)) * Y_tilde_c
    c_agents = np.array([c_agent_d, c_agent_c])

    d_agent_d = (beta / (p_d * denom_shares)) * Y_tilde_d + d0
    d_agent_c = (beta / (p_d * denom_shares)) * Y_tilde_c + d0
    d_agents = np.array([d_agent_d, d_agent_c])

    price_leisure_d = w_d * (1 - tau_w_d)
    price_leisure_c = w_c * (1 - tau_w_c)
    # Add checks for safety during post-processing too
    price_leisure_d = price_leisure_d if price_leisure_d > 1e-9 else 1e-9
    price_leisure_c = price_leisure_c if price_leisure_c > 1e-9 else 1e-9

    l_agent_d = (gamma / (denom_shares * price_leisure_d)) * Y_tilde_d
    l_agent_c = (gamma / (denom_shares * price_leisure_c)) * Y_tilde_c
    l_agent_d = np.clip(l_agent_d, 1e-9, t - 1e-9)
    l_agent_c = np.clip(l_agent_c, 1e-9, t - 1e-9)
    l_agents = np.array([l_agent_d, l_agent_c])

    # Labor supply (effective = actual since phi=1)
    labor_supply_agents = t - l_agents
    # ---------------------------------------------

    # Aggregates
    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor_c = labor_supply_agents[1] # Labor supply clean = agent C's supply
    agg_labor_d = labor_supply_agents[0] # Labor supply dirty = agent D's supply
    agg_labor_total = agg_labor_d + agg_labor_c

    # Profits
    profit_c = p_c * f_c - w_c * t_c - tau_z * z_c
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d

    # Budget errors
    budget_errors = np.zeros(n)
    # --- MODIFIED: Removed phi from income calc ---
    income_d = w_d * (1 - tau_w_d) * labor_supply_agents[0] + l
    expenditure_d = p_c * c_agents[0] + p_d * d_agents[0]
    budget_errors[0] = income_d - expenditure_d
    income_c = w_c * (1 - tau_w_c) * labor_supply_agents[1] + l
    expenditure_c = p_c * c_agents[1] + p_d * d_agents[1]
    budget_errors[1] = income_c - expenditure_c
    # ------------------------------------------

    # Utilities
    utilities = np.zeros(n)
    if c_agents[0] > 1e-9 and (d_agents[0] - d0) > 1e-9 and l_agents[0] > 1e-9:
         utilities[0] = alpha*np.log(c_agents[0]) + beta*np.log(d_agents[0]-d0) + gamma*np.log(l_agents[0])
    else: utilities[0] = -np.inf
    if c_agents[1] > 1e-9 and (d_agents[1] - d0) > 1e-9 and l_agents[1] > 1e-9:
         utilities[1] = alpha*np.log(c_agents[1]) + beta*np.log(d_agents[1]-d0) + gamma*np.log(l_agents[1])
    else: utilities[1] = -np.inf

    system_residuals = system_eqns(sol.x)

    # Assemble results dictionary
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d,
        "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
        "wage_vector": wage_vector,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents,
        "d_agents": d_agents,
        "l_agents": l_agents,
        "labor_supply_agents": labor_supply_agents, # Actual labor supply
        # "eff_labor_supply_agents": REMOVED, # No longer different from labor_supply_agents
        "agg_c": agg_c, "agg_d": agg_d,
        "agg_labor_c": agg_labor_c,
        "agg_labor_d": agg_labor_d,
        "agg_labor_total": agg_labor_total,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "sol_object": sol,
        "system_residuals": system_residuals
    }

    return sol.x, results, sol.success

# --- Example Run ---
tau_w_example = np.array([0.10, 0.15]) # Example: [tax_d, tax_c]
tau_z_example = 4.0
g_example = 5.0

try:
    solution, results, converged = solve(tau_w_example, tau_z_example, g_example)

    print("\nSolver Convergence Status:", converged)
    if results and results.get("sol_object"): # Use .get for safety
        print("Solver Status Code:", results["sol_object"].status)
        print("Solver Message:", results["sol_object"].message)

    if converged:
        print("\nSolution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
        print(solution)

        print("\nProduction Summary:")
        print(f"  Sector C: Labor Demand (t_c) = {results['t_c']:.4f}, Z Input (z_c) = {results['z_c']:.4f}, Output (f_c) = {results['f_c']:.4f}")
        print(f"  Sector D: Labor Demand (t_d) = {results['t_d']:.4f}, Z Input (z_d) = {results['z_d']:.4f}, Output (f_d) = {results['f_d']:.4f}")

        print("\nWages & Prices:")
        print(f"  Clean Wage (w_c): {results['w_c']:.4f}")
        print(f"  Dirty Wage (w_d): {results['w_d']:.4f}")
        print(f"  Dirty Good Price (p_d): {results['p_d']:.4f}")
        print(f"  Lump-sum Transfer (l): {results['l']:.4f}")

        print("\nHousehold Details (Agent D: index 0, Agent C: index 1):")
        # --- MODIFIED: Removed phi printout ---
        print(f"  Agent D (Dirty): tau_w={tau_w_example[0]:.2f}, c={results['c_agents'][0]:.4f}, d={results['d_agents'][0]:.4f}, leisure={results['l_agents'][0]:.4f}, labor={results['labor_supply_agents'][0]:.4f}, utility={results['utilities'][0]:.4f}")
        print(f"  Agent C (Clean): tau_w={tau_w_example[1]:.2f}, c={results['c_agents'][1]:.4f}, d={results['d_agents'][1]:.4f}, leisure={results['l_agents'][1]:.4f}, labor={results['labor_supply_agents'][1]:.4f}, utility={results['utilities'][1]:.4f}")
        # ------------------------------------

        print("\nAggregate Quantities:")
        print(f"  Aggregate c = {results['agg_c']:.4f}")
        print(f"  Aggregate d = {results['agg_d']:.4f}")
        print(f"  Total Labor = {results['agg_labor_total']:.4f} (Clean: {results['agg_labor_c']:.4f}, Dirty: {results['agg_labor_d']:.4f})") # Labor = Effective Labor
        print(f"  Aggregate Pollution (z_c+z_d) = {results['z_c'] + results['z_d']:.4f}")

        print("\nChecks:")
        print(f"  Profit C = {results['profit_c']:.4e}")
        print(f"  Profit D = {results['profit_d']:.4e}")
        print(f"  Budget Error (Agent D) = {results['budget_errors'][0]:.4e}")
        print(f"  Budget Error (Agent C) = {results['budget_errors'][1]:.4e}")
        print("\nSystem Residuals:")
        for i, res in enumerate(results['system_residuals']):
            print(f"  Eq {i+1}: {res:.4e}")

except ValueError as ve:
    print(f"\nError during setup: {ve}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()