import numpy as np
from scipy.optimize import root

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
# t = 24.0 # Original time endowment
t = 24.0 # Changed time endowment as per provided code
d0 = 0.5
epsilon_c = 0.995
# epsilon_d = 0.92 # Original value
epsilon_d = 0.582302 # Changed value as per provided code
p_c = 1.0 # Numeraire

# --- MODIFIED SECTION ---
# Productivity weights split by sector
# CHANGED: 3 households in dirty sector, 2 in clean sector.
# Clarification: phi_d sums to 1, phi_c sums to 1.
# These represent productivity/efficiency units for households in each sector.
# Example values are chosen, ensure they sum to 1 within each sector.
# Using values from provided code
phi_d = np.array([0.053164, 0.322405, (1-0.053164-0.322405)])  # Dirty sector households (3 types, sum = 1.0)
phi_c = np.array([0.295018, (1-0.295018)])       # Clean sector households (2 types, sum = 1.0)

# --- END MODIFIED SECTION ---

phi = np.concatenate([phi_d, phi_c])  # total phi for all 5 households (sum = 2.0)
n = len(phi) # Total number of household types (now 3 + 2 = 5)
n_d = len(phi_d) # Number of household types in dirty sector (now 3)
n_c = len(phi_c) # Number of household types in clean sector (now 2)


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
        # Add small epsilon to avoid division by zero if t_c/t_d or z_c/z_d are zero during iteration
        f_c = (epsilon_c * ((t_c + 1e-9)**r) + (1 - epsilon_c) * ((z_c + 1e-9)**r))**(1/r)
        f_d = (epsilon_d * ((t_d + 1e-9)**r) + (1 - epsilon_d) * ((z_d + 1e-9)**r))**(1/r)

        # --- Household side ---
        # Construct wage vector based on sector-specific wages
        # First n_d elements get w_d, next n_c elements get w_c
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

        # Calculate "full income net of committed expenditure" (Y_tilde)
        # Y_tilde = Potential Labour Income + Lump-sum Transfer - Committed Expenditure for d
        # tau_w corresponds element-wise to phi
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0

        # Check for non-positive Y_tilde which prevents log utility calculation
        if np.any(Y_tilde <= 0):
             # Return large residuals if Y_tilde is non-positive, guiding solver away
             return np.full(8, 1e6)

        # Calculate household demands based on Y_tilde (modified approach like Code 2)
        # Denominator for shares: alpha + beta + gamma
        denom_shares = alpha + beta + gamma

        # Demand for dirty good (d): includes baseline d0
        d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0

        # Demand for leisure (l_agents)
        # Price of leisure is the net wage rate: phi_i * wage_i * (1 - tau_w_i)
        price_leisure = phi * wage * (1 - tau_w)
        # Add small epsilon to avoid division by zero if price_leisure is zero
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde

        # Ensure leisure doesn't exceed total time or go below zero (numerical safety)
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)

        # Labor supply by household type (t - leisure)
        labor_supply_agents = t - l_agents

        # --- Aggregation ---
        # Aggregate effective labor supply by sector (weighted by productivity phi)
        # Slicing uses n_d which is now correctly set to 3
        agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d])
        agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:]) # Slicing starts correctly after n_d elements

        # Aggregate demand for dirty good
        agg_d = np.sum(d_agents)

        # --- System of Equations ---
        # Eq1: Labor market clearing (clean sector)
        eq1 = t_c - agg_labor_c
        # Eq2: Labor market clearing (dirty sector)
        eq2 = t_d - agg_labor_d
        # Eq3: Goods market clearing (dirty good)
        # Assumes 0.5*g is spent on dirty good d
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d # Added epsilon to p_d

        # Firm FOCs (marginal products = input prices)
        # Eq4: FOC Labor (clean sector) w_c = MP_L_c * p_c (p_c=1)
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq4 = w_c - MP_L_c
        # Eq5: FOC Z (clean sector) tau_z = MP_Z_c * p_c (p_c=1)
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq5 = tau_z - MP_Z_c
        # Eq6: FOC Labor (dirty sector) w_d = MP_L_d * p_d
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq6 = w_d - MP_L_d * p_d
        # Eq7: FOC Z (dirty sector) tau_z = MP_Z_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq7 = tau_z - MP_Z_d * p_d

        # Eq8: Government budget constraint
        # tau_w, phi, wage, labor_supply_agents all have length n
        total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        # l is per capita lump sum transfer, n*l is total transfer
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    # Initial guess: [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]
    # Keep the same guess structure, solver might need adjustments if it fails
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing --- (No changes needed here, relies on n, n_d, n_c which are updated)
    # Unpack solution
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l_sol = sol.x # Renamed l to l_sol to avoid clash with global l
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    # Recalculate variables based on solution
    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

    wage_sol = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Renamed wage -> wage_sol

    # Recalculate demands using the consistent Y_tilde approach
    Y_tilde = phi * wage_sol * (1 - tau_w) * t + l_sol - p_d * d0 # Use wage_sol, l_sol
    denom_shares = alpha + beta + gamma

    c_agents = (alpha / (p_c * denom_shares)) * Y_tilde
    d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0
    price_leisure = phi * wage_sol * (1 - tau_w) # Use wage_sol
    l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, t - 1e-9) # Ensure bounds
    labor_supply_agents = t - l_agents

    # Recalculate aggregates
    agg_c = np.sum(c_agents) # Note: Aggregating demands directly (not weighted by phi)
    agg_d = np.sum(d_agents) # Note: Aggregating demands directly
    # Aggregate effective labor = sum(phi_i * L_i)
    agg_labor = np.sum(phi * labor_supply_agents)

    # Profits (Note: Firms pay for actual labor t_c, t_d, not effective labor)
    profit_c = p_c * f_c - w_c * t_c - tau_z * z_c # p_c = 1
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d

    # Check budget constraints for each household type
    budget_errors = np.zeros(n)
    # Calculate actual income for results dictionary
    incomes_sol = np.zeros(n)
    for i in range(n):
        # Earned income = productivity * wage * (1-tax) * labor_time + transfer
        income_i = phi[i] * wage_sol[i] * (1 - tau_w[i]) * labor_supply_agents[i] + l_sol # Use wage_sol, l_sol
        incomes_sol[i] = income_i # Store income
        # Expenditure = p_c*c + p_d*d
        expenditure_i = p_c * c_agents[i] + p_d * d_agents[i]
        budget_errors[i] = income_i - expenditure_i

    # Calculate utilities
    utilities = np.zeros(n)
    for i in range(n):
        # Check for positive arguments in logs
        if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
            utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
        else:
            # Assign very low utility if constraints are violated
            utilities[i] = -np.inf

    # Calculate residuals at the solution
    residuals = system_eqns(sol.x)

    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l_sol, # Use l_sol
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents, "labor_supply_agents": labor_supply_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "residuals": residuals,
        "sol": sol,
        "Y_tilde": Y_tilde, # Also return Y_tilde for inspection if needed
        "incomes": incomes_sol # Add calculated incomes to results
    }

    return sol.x, results, sol.success

# Inputs (Example Usage)
# Ensure tau_w has n=5 elements (3 for dirty, 2 for clean sector)
# The meaning changes: first 3 are for dirty, last 2 for clean
# Using values from provided code
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_z = 0.2
g = 5.0

# Solve model
try:
    solution, results, converged = solve(tau_w, tau_z, g)

    # Output
    print(f"Model Setup: {n_d} dirty households, {n_c} clean households (Total: {n})")
    print("solution status:", results["sol"].status)
    print("solution message:", results["sol"].message)
    print("convergence:", converged)
    if not converged:
         print("Solver did NOT converge. Results might be unreliable.")
         print("Final residuals:", results["residuals"])

    print("\nsolution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
    print(solution)

    print("\nproduction summary:")
    print(f"sector c: t_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}, f_c = {results['f_c']:.4f}")
    print(f"sector d: t_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}, f_d = {results['f_d']:.4f}")
    print(f"Wages: w_c = {results['w_c']:.4f}, w_d = {results['w_d']:.4f}")
    print(f"Dirty Good Price: p_d = {results['p_d']:.4f}")
    print(f"Lump-sum Transfer: l = {results['l']:.4f}")


    print("\nhousehold demands, leisure, and labor supply:")
    # Label households based on sector
    for i in range(n_d):
         print(f"household D{i+1} (phi={phi[i]:.3f}): c={results['c_agents'][i]:.4f}, d={results['d_agents'][i]:.4f}, leisure={results['l_agents'][i]:.4f}, labor={results['labor_supply_agents'][i]:.4f}")
    for i in range(n_c):
         idx = n_d + i # Get the correct index in the full arrays
         print(f"household C{i+1} (phi={phi[idx]:.3f}): c={results['c_agents'][idx]:.4f}, d={results['d_agents'][idx]:.4f}, leisure={results['l_agents'][idx]:.4f}, labor={results['labor_supply_agents'][idx]:.4f}")

    print("\nhousehold utilities:")
    for i in range(n_d):
         print(f"household D{i+1}: utility = {results['utilities'][i]:.4f}")
    for i in range(n_c):
         idx = n_d + i
         print(f"household C{i+1}: utility = {results['utilities'][idx]:.4f}")

    print("\nbudget constraint errors:")
    for i in range(n_d):
         print(f"household D{i+1}: error = {results['budget_errors'][i]:.4e}")
    for i in range(n_c):
         idx = n_d + i
         print(f"household C{i+1}: error = {results['budget_errors'][idx]:.4e}")

    print("\nresiduals of equilibrium equations:")
    for i, res in enumerate(results["residuals"], 1):
         print(f"Equation {i}: residual = {res:.4e}")

    # --- NEW CODE SECTION: Income Distribution ---
    if converged: # Only print income distribution if solver converged
        print("\nHousehold Income Distribution:")

        # Get incomes calculated during post-processing
        incomes = results['incomes']

        # Calculate total income
        total_income = np.sum(incomes)

        # Print income distribution table
        print(f"Total Disposable Income: {total_income:.4f}")
        print("------------------------------------------------------")
        print("Household | Sector | Productivity (phi) | Income   | Share (%)")
        print("------------------------------------------------------")
        # Dirty sector households
        for i in range(n_d):
            share = (incomes[i] / total_income) * 100 if total_income > 1e-9 else 0
            print(f"D{i+1:<8} | Dirty  | {phi[i]:<18.5f} | {incomes[i]:<8.4f} | {share:<8.2f}")
        # Clean sector households
        for i in range(n_c):
            idx = n_d + i
            share = (incomes[idx] / total_income) * 100 if total_income > 1e-9 else 0
            print(f"C{i+1:<8} | Clean  | {phi[idx]:<18.5f} | {incomes[idx]:<8.4f} | {share:<8.2f}")
        print("------------------------------------------------------")
    else:
        print("\nIncome distribution not calculated because the solver did not converge.")
    # --- END NEW CODE SECTION ---


except ValueError as ve:
    print(f"\nInput Error: {ve}")
except Exception as e:
    # Print full traceback for debugging unexpected errors
    import traceback
    print(f"\nAn error occurred during solving or post-processing:")
    print(traceback.format_exc())