import numpy as np
from scipy.optimize import root
import copy

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
# r = -1.0 # Removed single r
t = 12.0
d0 = 0.5
epsilon_c = 0.995
epsilon_d = 0.92  # <<< Using value from the code in this request
p_c = 1.0

# --- Define separate elasticity parameters ---
r_c = -1.0  # Elasticity parameter for the clean sector (Example value)
r_d = -1.0  # Elasticity parameter for the dirty sector (Example value)
print(f"Using sector-specific elasticity: r_c = {r_c}, r_d = {r_d}")
# ---

# Productivity weights split by sector (NEW 4 Dirty + 1 Clean STRUCTURE)
# Clarification: phi_d sums to 1, phi_c sums to 1.
# These represent *normalized* productivity/efficiency units within each sector.
# Define new example weights for the 4 dirty + 1 clean households:
phi_d = np.array([0.1, 0.2, 0.3, 0.4])   # Example: 4 Dirty sector households (sums to 1.0)
phi_c = np.array([1.0])                 # Example: 1 Clean sector household (must sum to 1.0)

# The full phi vector used in income/leisure calculations represents these productivities.
# It will sum to 2.0, consistent with the previous interpretation.
phi = np.concatenate([phi_d, phi_c])
n = len(phi) # n remains 5
n_d = len(phi_d) # <<< UPDATED: n_d is now 4
n_c = len(phi_c) # <<< UPDATED: n_c is now 1

print(f"Model structure updated: n_d = {n_d}, n_c = {n_c}, n = {n}")
print(f"phi_d (norm, sum={np.sum(phi_d)}): {phi_d}")
print(f"phi_c (norm, sum={np.sum(phi_c)}): {phi_c}")
print(f"phi (full, sum={np.sum(phi)}):   {phi}")


def solve(tau_w, tau_z, g):

    # Ensure tau_w still matches the total number of households (n=5)
    if len(tau_w) != n:
        raise ValueError(f"Length of tau_w ({len(tau_w)}) must match number of households ({n})")
    print(f"\nApplying tau_w: {tau_w}")
    print(f" tau_w for Dirty Sector (first {n_d}): {tau_w[:n_d]}")
    print(f" tau_w for Clean Sector (last {n_c}):  {tau_w[n_d:]}")


    def system_eqns(y):
        # Unpack variables to solve for
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y

        # Ensure positivity for z inputs
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        # --- Firm side ---
        # Production functions (CES) using r_c and r_d
        f_c = (epsilon_c * ((t_c + 1e-9)**r_c) + (1 - epsilon_c) * ((z_c + 1e-9)**r_c))**(1/r_c)
        f_d = (epsilon_d * ((t_d + 1e-9)**r_d) + (1 - epsilon_d) * ((z_d + 1e-9)**r_d))**(1/r_d)

        # --- Household side ---
        # Construct wage vector based on sector-specific wages (uses new n_d=4, n_c=1)
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Now expects n_d=4, n_c=1

        # Calculate "full income net of committed expenditure" (Y_tilde)
        # Uses the full phi vector (summing to 2)
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0

        # Check for non-positive Y_tilde which prevents log utility calculation
        if np.any(Y_tilde <= 1e-9):
             return np.full(8, 1e6)

        # Calculate household demands based on Y_tilde
        denom_shares = alpha + beta + gamma
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_tilde + d0

        # Demand for leisure (l_agents)
        # Uses the full phi vector (summing to 2) for price of leisure
        price_leisure = phi * wage * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)

        # Labor supply by household type (t - leisure)
        labor_supply_agents = t - l_agents

        # --- Aggregation ---
        # Aggregate effective labor supply by sector (uses new n_d=4, n_c=1 for slicing)
        # Weighted by the normalized productivity vectors phi_d, phi_c (each sum to 1)
        agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d]) # Slice uses n_d=4
        agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:]) # Slice uses n_d=4 index start

        # Aggregate demand for dirty good (simple sum over all 5 agents)
        agg_d = np.sum(d_agents)

        # --- System of Equations ---
        # Eq1: Labor market clearing (clean sector)
        eq1 = t_c - agg_labor_c
        # Eq2: Labor market clearing (dirty sector)
        eq2 = t_d - agg_labor_d
        # Eq3: Goods market clearing (dirty good)
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d

        # Firm FOCs using r_c and r_d
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c)); eq4 = w_c - MP_L_c
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c)); eq5 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d)); eq6 = w_d - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d)); eq7 = tau_z - MP_Z_d * p_d

        # Government budget constraint (uses full phi vector summing to 2)
        total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    # Initial guess remains the same size
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing ---
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)

    # Recalculate variables based on solution using r_c and r_d
    f_c = (epsilon_c * (t_c**r_c) + (1 - epsilon_c) * (z_c**r_c))**(1/r_c) if sol.success else np.nan
    f_d = (epsilon_d * (t_d**r_d) + (1 - epsilon_d) * (z_d**r_d))**(1/r_d) if sol.success else np.nan

    # Uses new n_d=4, n_c=1
    wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])

    # Recalculate demands
    Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0 if sol.success else np.full(n, np.nan)
    denom_shares = alpha + beta + gamma
    c_agents = (alpha / (p_c * denom_shares)) * Y_tilde if sol.success else np.full(n, np.nan)
    d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_tilde + d0 if sol.success else np.full(n, np.nan)
    price_leisure = phi * wage * (1 - tau_w) if sol.success else np.full(n, np.nan)
    l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde if sol.success else np.full(n, np.nan)
    l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
    labor_supply_agents = t - l_agents

    # Recalculate aggregates
    agg_c = np.sum(c_agents); agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi * labor_supply_agents) # Uses full phi (sum=2)

    # Profits
    profit_c = p_c * f_c - w_c * t_c - tau_z * z_c if sol.success else np.nan
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d if sol.success else np.nan

    # Budget errors
    budget_errors = np.zeros(n) * np.nan
    if sol.success:
        for i in range(n):
            income_i = phi[i] * wage[i] * (1 - tau_w[i]) * labor_supply_agents[i] + l
            expenditure_i = p_c * c_agents[i] + p_d * d_agents[i]
            budget_errors[i] = income_i - expenditure_i

    # Utilities
    utilities = np.zeros(n) * np.nan
    if sol.success:
        for i in range(n):
            if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
                utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
            else: utilities[i] = -np.inf

    # Residuals
    residuals = system_eqns(sol.x)

    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents, "labor_supply_agents": labor_supply_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "residuals": residuals,
        "sol": sol,
        "phi": phi, "phi_d": phi_d, "phi_c": phi_c,
        "r_c": r_c, "r_d": r_d
    }
    return sol.x, results, sol.success

# Inputs (Example Usage)
# tau_w must still have n=5 elements
# Interpretation: tau_w[0..3] apply to dirty HHs, tau_w[4] applies to clean HH
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_z = 1.0
g = 5.0

# Solve model
try:
    solution, results, converged = solve(tau_w, tau_z, g)

    # Output
    print("\n" + "="*50)
    print("Model Run Results (4 Dirty + 1 Clean HH, Separate r)")
    print("="*50)
    print(f"(Using r_c={results['r_c']}, r_d={results['r_d']})")
    print("solution status:", results["sol"].status)
    print("solution message:", results["sol"].message)
    print("convergence:", converged)
    if not converged:
         print("WARNING: Solver did NOT converge. Results are likely unreliable.")

    print("\nsolution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
    print(np.round(solution, 5))

    print("\nproduction summary:")
    print(f"sector c: t_prod = {results.get('t_c', np.nan):.4f}, z_c = {results.get('z_c', np.nan):.4f}, f_c = {results.get('f_c', np.nan):.4f}")
    print(f"sector d: t_prod = {results.get('t_d', np.nan):.4f}, z_d = {results.get('z_d', np.nan):.4f}, f_d = {results.get('f_d', np.nan):.4f}")
    print(f"Wages: w_c = {results.get('w_c', np.nan):.4f}, w_d = {results.get('w_d', np.nan):.4f}")
    print(f"Dirty Good Price: p_d = {results.get('p_d', np.nan):.4f}")
    print(f"Lump-sum Transfer: l = {results.get('l', np.nan):.4f}")

    print("\nhousehold demands, leisure, and labor supply:")
    print(f"{'Household':<10} {'Sector':<8} {'Phi':<6} {'Cons (c)':<10} {'Cons (d)':<10} {'Leisure':<10} {'Labor':<10}")
    print("-" * 70)
    phi_full = results.get('phi', [np.nan]*n)
    for i in range(n):
         # Correctly identify sector based on n_d=4
         sector = "Dirty" if i < n_d else "Clean"
         c_val = results.get('c_agents', [np.nan]*n)[i]; d_val = results.get('d_agents', [np.nan]*n)[i]
         l_val = results.get('l_agents', [np.nan]*n)[i]; lab_val = results.get('labor_supply_agents', [np.nan]*n)[i]
         print(f"{i+1:<10} {sector:<8} {phi_full[i]:<6.3f} {c_val:<10.4f} {d_val:<10.4f} {l_val:<10.4f} {lab_val:<10.4f}")

    # (Keep Income Analysis section if desired)
    if converged:
        print("\n" + "="*30); print("      Income Analysis"); print("="*30)
        w_c = results['w_c']; w_d = results['w_d']; l = results['l']
        labor_supply_agents = results['labor_supply_agents']
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Uses n_d=4, n_c=1
        phi_full = results['phi']; net_labor_income = phi_full * wage * (1 - tau_w) * labor_supply_agents
        total_disposable_income = net_labor_income + l
        print("\nHousehold Disposable Income Breakdown:")
        print("-" * 80); print(f"{'Household':<10} {'Sector':<8} {'Phi':<10} {'Wage':<10} {'Net Labor Inc':<15} {'Transfer (l)':<15} {'Total Income':<15}"); print("-" * 80)
        for i in range(n):
            sector = "Dirty" if i < n_d else "Clean"; sector_wage = wage[i]
            print(f"{i+1:<10} {sector:<8} {phi_full[i]:<10.3f} {sector_wage:<10.4f} {net_labor_income[i]:<15.4f} {l:<15.4f} {total_disposable_income[i]:<15.4f}")
        print("-" * 80)
        total_economy_income = np.sum(total_disposable_income)
        if abs(total_economy_income) > 1e-9:
             income_shares = total_disposable_income / total_economy_income
             print(f"\nTotal Disposable Income in Economy: {total_economy_income:.4f}")
             print("\nIncome Distribution (Share of Total):")
             print("-" * 35); print(f"{'Household':<10} {'Share (%)':<15}"); print("-" * 35)
             for i in range(n): print(f"{i+1:<10} {income_shares[i]*100:<15.2f}%")
             print("-" * 35)
             max_income = np.max(total_disposable_income); min_income = np.min(total_disposable_income)
             print("\nIncome Inequality Measures:")
             if min_income > 1e-9: print(f"Max/Min Income Ratio: {max_income / min_income:.4f}")
             else: print("Max/Min Income Ratio: Minimum income is zero or near zero.")
             print(f"Income Range (Max - Min): {max_income - min_income:.4f}")
        else: print("\nTotal economy income is zero or near zero.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()