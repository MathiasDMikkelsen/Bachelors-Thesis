import numpy as np
from scipy.optimize import root

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0
d0 = 0.25
epsilon_c = 0.995
epsilon_d = 0.582302
p_c = 1.0  # Numeraire

# --- MODIFIED SECTION ---
# Productivity weights split by sector
# CHANGED: 3 households in dirty sector, 2 in clean sector.
# Clarification: phi_d sums to 1, phi_c sums to 1.
# These represent productivity/efficiency units for households in each sector.
# Example values are chosen, ensure they sum to 1 within each sector.
phi_d = np.array([0.053164, 0.322405, (1 - 0.053164 - 0.322405)])  # Dirty sector households (3 types, sum = 1.0)
phi_c = np.array([0.295018, (1 - 0.295018)])                          # Clean sector households (2 types, sum = 1.0)
# --- END MODIFIED SECTION ---

phi = np.concatenate([phi_d, phi_c])  # total phi for all 5 households
n = len(phi)      # Total number of household types (3 + 2 = 5)
n_d = len(phi_d)  # Number of dirty sector households (3)
n_c = len(phi_c)  # Number of clean sector households (2)

def solve(tau_w, tau_z, g):

    # Ensure tau_w matches the structure of phi (total number of households)
    if len(tau_w) != n:
        raise ValueError(f"Length of tau_w ({len(tau_w)}) must match total number of households ({n})")

    def system_eqns(y):
        # Unpack variables to solve for
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y

        # Ensure positivity for z inputs
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        # --- Firm side ---
        # Production functions (CES)
        f_c = (epsilon_c * ((t_c + 1e-9)**r) + (1 - epsilon_c) * ((z_c + 1e-9)**r))**(1/r)
        f_d = (epsilon_d * ((t_d + 1e-9)**r) + (1 - epsilon_d) * ((z_d + 1e-9)**r))**(1/r)

        # --- Household side ---
        # Construct wage vector based on sector-specific wages:
        # First n_d elements receive w_d; next n_c elements receive w_c.
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

        # Calculate "full income net of committed expenditure" (Y_tilde)
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0

        # Prevent non-positive values in Y_tilde (which would break the log utility)
        if np.any(Y_tilde <= 0):
            return np.full(8, 1e6)

        # Household demands based on Y_tilde
        denom_shares = alpha + beta + gamma

        # Demand for dirty good (includes baseline d0)
        d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0

        # Demand for leisure (l_agents) -- price of leisure is the net wage rate
        price_leisure = phi * wage * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)

        # Labor supply by household type (total time minus leisure)
        labor_supply_agents = t - l_agents

        # --- Aggregation ---
        agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d])
        agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:])
        agg_d = np.sum(d_agents)

        # --- System of Equations ---
        # Eq1: Clean sector labor market clearing
        eq1 = t_c - agg_labor_c
        # Eq2: Dirty sector labor market clearing
        eq2 = t_d - agg_labor_d
        # Eq3: Dirty good market clearing
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d

        # Firm FOCs for the clean sector:
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq4 = w_c - MP_L_c
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq5 = tau_z - MP_Z_c

        # Firm FOCs for the dirty sector:
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq6 = w_d - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq7 = tau_z - MP_Z_d * p_d

        # Government budget constraint:
        total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    # Initial guess: [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    # Solve the system using the Levenberg-Marquardt method
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing ---
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

    # Create wage vector (first n_d entries for dirty-sector; next n_c entries for clean-sector)
    wage_vector = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

    # Recalculate household variables using wage_vector
    Y_tilde = phi * wage_vector * (1 - tau_w) * t + l - p_d * d0
    denom_shares = alpha + beta + gamma

    c_agents = (alpha / (p_c * denom_shares)) * Y_tilde
    d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0
    price_leisure = phi * wage_vector * (1 - tau_w)
    l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
    labor_supply_agents = t - l_agents

    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi * labor_supply_agents)

    profit_c = p_c * f_c - w_c * t_c - tau_z * z_c
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d

    # Budget errors calculated for each household type
    budget_errors = np.zeros(n)
    for i in range(n):
        income_i = phi[i] * wage_vector[i] * (1 - tau_w[i]) * labor_supply_agents[i] + l
        expenditure_i = p_c * c_agents[i] + p_d * d_agents[i]
        budget_errors[i] = income_i - expenditure_i

    utilities = np.zeros(n)
    for i in range(n):
        if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
            utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
        else:
            utilities[i] = -np.inf

    residuals = system_eqns(sol.x)

    # Assemble and return results including the wage vector
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d,
        "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
        "wage_vector": wage_vector,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents, "labor_supply_agents": labor_supply_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "residuals": residuals,
        "sol": sol,
        "Y_tilde": Y_tilde  # Also return Y_tilde for inspection if needed
    }

    return sol.x, results, sol.success

# Inputs (Ensure tau_w has n=5 elements: 3 for dirty-sector, 2 for clean-sector)
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_z = 4
g = 5.0

try:
    solution, results, converged = solve(tau_w, tau_z, g)

    print(f"Model Setup: {n_d} dirty households, {n_c} clean households (Total: {n})")
    print("solution status:", results["sol"].status)
    print("solution message:", results["sol"].message)
    print("convergence:", converged)
    if not converged:
         print("Solver did NOT converge. Results might be unreliable.")
         print("Final residuals:", results["residuals"])

    print("\nSolution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
    print(solution)

    print("\nProduction summary:")
    print(f"Sector c: t_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}, f_c = {results['f_c']:.4f}")
    print(f"Sector d: t_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}, f_d = {results['f_d']:.4f}")
    
    print("\nWage Vector (ordered by households):")
    print(results["wage_vector"])

    print("\nHousehold demands, leisure, and labor supply:")
    for i in range(n_d):
         print(f"Household D{i+1} (phi={phi[i]:.3f}): c={results['c_agents'][i]:.4f}, d={results['d_agents'][i]:.4f}, leisure={results['l_agents'][i]:.4f}, labor={results['labor_supply_agents'][i]:.4f}")
    for i in range(n_c):
         idx = n_d + i  # Proper index for clean households
         print(f"Household C{i+1} (phi={phi[idx]:.3f}): c={results['c_agents'][idx]:.4f}, d={results['d_agents'][idx]:.4f}, leisure={results['l_agents'][idx]:.4f}, labor={results['labor_supply_agents'][idx]:.4f}")

    print("\nHousehold utilities:")
    for i in range(n_d):
         print(f"Household D{i+1}: utility = {results['utilities'][i]:.4f}")
    for i in range(n_c):
         idx = n_d + i
         print(f"Household C{i+1}: utility = {results['utilities'][idx]:.4f}")

    print("\nBudget constraint errors:")
    for i in range(n_d):
         print(f"Household D{i+1}: error = {results['budget_errors'][i]:.4e}")
    for i in range(n_c):
         idx = n_d + i
         print(f"Household C{i+1}: error = {results['budget_errors'][idx]:.4e}")

    print("\nResiduals of equilibrium equations:")
    for i, res in enumerate(results["residuals"], 1):
         print(f"Equation {i}: residual = {res:.4e}")

except Exception as e:
    print(f"An error occurred during solving or post-processing: {e}")