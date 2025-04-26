import numpy as np
from scipy.optimize import root

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
# --- MODIFIED: Separate r values for each sector ---
# r relates to the elasticity of substitution (sigma = 1 / (1 - r))
# Choose appropriate values based on your model assumptions.
r_c = -1.0  # Example value for the clean sector (e.g., sigma_c = 0.5)
r_d = -(1-0.8)/0.8  # Example value for the dirty sector (e.g., sigma_d = 2/3)
# --- END MODIFICATION ---
t = 24.0
d0 = 0.5
epsilon_c = 0.995
epsilon_d = 0.612971
p_c = 1.0  # Numeraire

# Productivity weights split by sector
phi_d = np.array([0.071745, 0.323251, (1 - 0.071745 - 0.323251)])  # Dirty sector households (3 types, sum = 1.0)
phi_c = np.array([0.297862, (1 - 0.297862)])                      # Clean sector households (2 types, sum = 1.0)

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
        # Production functions (CES) - MODIFIED to use r_c and r_d
        # Add small epsilon to avoid division by zero or log(0) if r is near 0
        # or if inputs are zero.
        f_c = (epsilon_c * ((t_c + 1e-9)**r_c) + (1 - epsilon_c) * ((z_c + 1e-9)**r_c))**(1/r_c)
        f_d = (epsilon_d * ((t_d + 1e-9)**r_d) + (1 - epsilon_d) * ((z_d + 1e-9)**r_d))**(1/r_d)

        # --- Household side ---
        # Construct wage vector based on sector-specific wages:
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

        # Calculate "full income net of committed expenditure" (Y_tilde)
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0

        # Prevent non-positive values in Y_tilde (which would break the log utility)
        if np.any(Y_tilde <= 0):
            # Return a large penalty if Y_tilde is invalid to guide the solver away
            return np.full(8, 1e6)

        # Household demands based on Y_tilde
        denom_shares = alpha + beta + gamma

        # Demand for dirty good (includes baseline d0)
        d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0

        # Demand for leisure (l_agents) -- price of leisure is the net wage rate
        price_leisure = phi * wage * (1 - tau_w)
        # Add small epsilon to price_leisure denominator to avoid division by zero
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        # Ensure leisure doesn't exceed total time or become negative
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
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d # Added epsilon to p_d denominator

        # Firm FOCs for the clean sector - MODIFIED to use r_c
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c))
        eq4 = w_c - MP_L_c # Note: p_c = 1 (numeraire)
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c))
        eq5 = tau_z - MP_Z_c # Note: p_c = 1 (numeraire)

        # Firm FOCs for the dirty sector - MODIFIED to use r_d
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d))
        eq6 = w_d - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d))
        eq7 = tau_z - MP_Z_d * p_d

        # Government budget constraint:
        total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    # Initial guess: [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]
    # Might need adjustment depending on the chosen r_c, r_d values
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    # Solve the system using the Levenberg-Marquardt method (robust for non-linear systems)
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing ---
    if not sol.success:
        print("Warning: Solver did not converge. Results may be inaccurate.")
        print("Solver message:", sol.message)
        # Return None or raise an error if convergence failure is critical
        # return None, None, False # Example

    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    # Recalculate production functions using solved values and correct r - MODIFIED
    # Use solved t_c, z_c, t_d, z_d here
    f_c = (epsilon_c * (t_c**r_c) + (1 - epsilon_c) * (z_c**r_c))**(1/r_c)
    f_d = (epsilon_d * (t_d**r_d) + (1 - epsilon_d) * (z_d**r_d))**(1/r_d)

    # Create wage vector (first n_d entries for dirty-sector; next n_c entries for clean-sector)
    wage_vector = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

    # Recalculate household variables using solved values and wage_vector
    Y_tilde = phi * wage_vector * (1 - tau_w) * t + l - p_d * d0
    denom_shares = alpha + beta + gamma

    # Avoid division by zero if p_c or p_d are somehow zero (though p_c=1 and p_d should be solved > 0)
    c_agents = (alpha / (p_c * denom_shares + 1e-12)) * Y_tilde
    d_agents = (beta / (p_d * denom_shares + 1e-12)) * Y_tilde + d0
    price_leisure = phi * wage_vector * (1 - tau_w)
    l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, t - 1e-9) # Ensure leisure is valid
    labor_supply_agents = t - l_agents

    # Aggregate results
    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d]) # Recalculate for consistency checks
    agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:]) # Recalculate for consistency checks
    agg_labor = np.sum(phi * labor_supply_agents) # Total effective labor supply

    # Calculate profits (should be near zero if FOCs hold and p_c=1)
    profit_c = p_c * f_c - w_c * t_c - tau_z * z_c
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d

    # Budget errors calculated for each household type
    budget_errors = np.zeros(n)
    for i in range(n):
        income_i = phi[i] * wage_vector[i] * (1 - tau_w[i]) * labor_supply_agents[i] + l
        expenditure_i = p_c * c_agents[i] + p_d * d_agents[i]
        budget_errors[i] = income_i - expenditure_i

    # Calculate utilities
    utilities = np.zeros(n)
    for i in range(n):
        # Check for positive consumption above baseline and positive leisure
        if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
             # Added check for d_agents[i] > d0 before taking log
            utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
        else:
            # Assign a very low utility if consumption/leisure constraints are violated
            utilities[i] = -np.inf

    # Get the final residuals from the solver's perspective
    residuals = system_eqns(sol.x) if sol.success else np.full(8, np.nan) # Recalculate or use NaN if failed

    # Assemble and return results including the wage vector
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d,
        "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
        "wage_vector": wage_vector,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents, "labor_supply_agents": labor_supply_agents,
        "agg_c": agg_c, "agg_d": agg_d,
        "agg_labor_c_supply": agg_labor_c, "agg_labor_d_supply": agg_labor_d, # Aggregate supply from households
        "agg_labor_c_demand": t_c, "agg_labor_d_demand": t_d, # Aggregate demand from firms
        "agg_labor_total_supply": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "residuals": residuals,
        "sol_object": sol, # Keep the original solution object for details
        "Y_tilde": Y_tilde
    }

    return sol, results, sol.success

# --- Example Usage ---
# Inputs (Ensure tau_w has n=5 elements: 3 for dirty-sector, 2 for clean-sector)
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_z = 0.1 # Make float for clarity
g = 5.0     # Make float for clarity

try:
    solution_vector, results, converged = solve(tau_w, tau_z, g)

    print(f"Model Setup: {n_d} dirty households, {n_c} clean households (Total: {n})")
    print(f"Parameters: r_c={r_c}, r_d={r_d}") # Display the r values used
    print("Solver status:", results["sol_object"].status)
    print("Solver message:", results["sol_object"].message)
    print("Convergence:", converged)

    if not converged:
        print("\nSolver did NOT converge. Results might be unreliable.")
        print("Final residuals:", results["residuals"])
    else:
        print("\nSolution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
        print(solution_vector)

        print("\nEquilibrium Prices & Transfers:")
        print(f"p_c = {p_c:.4f} (Numeraire)")
        print(f"p_d = {results['p_d']:.4f}")
        print(f"w_c = {results['w_c']:.4f}")
        print(f"w_d = {results['w_d']:.4f}")
        print(f"l (lump-sum transfer) = {results['l']:.4f}")

        print("\nProduction Summary:")
        print(f"Sector c: Labor Demand (t_c) = {results['t_c']:.4f}, Z Input (z_c) = {results['z_c']:.4f}, Output (f_c) = {results['f_c']:.4f}")
        print(f"Sector d: Labor Demand (t_d) = {results['t_d']:.4f}, Z Input (z_d) = {results['z_d']:.4f}, Output (f_d) = {results['f_d']:.4f}")

        print("\nLabor Market Summary:")
        print(f"Clean Sector: Supply = {results['agg_labor_c_supply']:.4f}, Demand = {results['agg_labor_c_demand']:.4f}, Diff = {results['agg_labor_c_supply']-results['agg_labor_c_demand']:.4e}")
        print(f"Dirty Sector: Supply = {results['agg_labor_d_supply']:.4f}, Demand = {results['agg_labor_d_demand']:.4f}, Diff = {results['agg_labor_d_supply']-results['agg_labor_d_demand']:.4e}")
        print(f"Total Eff. Labor Supply = {results['agg_labor_total_supply']:.4f}")


        print("\nWage Vector (ordered by households):")
        print(results["wage_vector"])

        print("\nHousehold Details (Consumption, Leisure, Labor Supply):")
        for i in range(n_d):
            print(f"  Household D{i+1} (phi={phi[i]:.3f}): c={results['c_agents'][i]:.4f}, d={results['d_agents'][i]:.4f}, leisure={results['l_agents'][i]:.4f}, labor={results['labor_supply_agents'][i]:.4f}")
        for i in range(n_c):
            idx = n_d + i
            print(f"  Household C{i+1} (phi={phi[idx]:.3f}): c={results['c_agents'][idx]:.4f}, d={results['d_agents'][idx]:.4f}, leisure={results['l_agents'][idx]:.4f}, labor={results['labor_supply_agents'][idx]:.4f}")

        print("\nHousehold Utilities:")
        for i in range(n_d):
            print(f"  Household D{i+1}: utility = {results['utilities'][i]:.4f}")
        for i in range(n_c):
            idx = n_d + i
            print(f"  Household C{i+1}: utility = {results['utilities'][idx]:.4f}")

        print("\nCheck: Firm Profits (should be near zero):")
        print(f"  Profit Sector c = {results['profit_c']:.4e}")
        print(f"  Profit Sector d = {results['profit_d']:.4e}")

        print("\nCheck: Household Budget Constraint Errors (should be near zero):")
        for i in range(n_d):
            print(f"  Household D{i+1}: error = {results['budget_errors'][i]:.4e}")
        for i in range(n_c):
            idx = n_d + i
            print(f"  Household C{i+1}: error = {results['budget_errors'][idx]:.4e}")

        print("\nCheck: Residuals of Equilibrium Equations (should be near zero):")
        eq_names = [
            "Eq1: Clean Labor Market", "Eq2: Dirty Labor Market", "Eq3: Dirty Good Market",
            "Eq4: Firm C FOC Labor", "Eq5: Firm C FOC Z", "Eq6: Firm D FOC Labor",
            "Eq7: Firm D FOC Z", "Eq8: Gov Budget Constraint"
            ]
        for name, res in zip(eq_names, results["residuals"]):
            print(f"  {name}: residual = {res:.4e}")

except Exception as e:
    print(f"\n--- An error occurred during solving or post-processing ---")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging