import numpy as np
from scipy.optimize import root
import copy # Kept for consistency, though less critical here

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0
d0 = 0.5
epsilon_c = 0.995
epsilon_d = 0.92
p_c = 1.0

# --- UPDATED phi for 10 households ---
# EXAMPLE: Splitting the original 5 shares approximately in half.
# ** REPLACE THIS WITH YOUR ACTUAL 10-HOUSEHOLD DISTRIBUTION **
phi_orig = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
phi = np.array([
    phi_orig[0]/2, phi_orig[0]/2,         # Split HH1
    phi_orig[1]/2, phi_orig[1]/2,         # Split HH2
    phi_orig[2]/2, phi_orig[2]/2,         # Split HH3
    phi_orig[3]/2, phi_orig[3]/2,         # Split HH4
    phi_orig[4]/2, phi_orig[4]/2          # Split HH5
])
# Ensure it still sums very close to 1 after splitting
phi = phi / np.sum(phi) # Normalize just in case of rounding errors
# ---

n = len(phi) # <<< UPDATED: n is now 10
print(f"Model updated for n = {n} households.")
print(f"Using phi (sum={np.sum(phi)}): \n{np.round(phi, 5)}")


def solve(tau_w, tau_z, g):

    # Ensure tau_w length matches the new n
    if len(tau_w) != n:
         raise ValueError(f"Length of tau_w ({len(tau_w)}) must match number of households ({n})")

    # --- Nested system_eqns function ---
    # (No changes needed inside, vector ops scale automatically)
    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        f_c = (epsilon_c * ((t_c + 1e-9)**r) + (1 - epsilon_c) * ((z_c + 1e-9)**r))**(1/r)
        f_d = (epsilon_d * ((t_d + 1e-9)**r) + (1 - epsilon_d) * ((z_d + 1e-9)**r))**(1/r)

        # These calculations now use phi and tau_w of length n=10
        Y_term = phi * w * (1 - tau_w) * t + l - p_d * d0
        if np.any(Y_term <= 1e-9): return np.full(7, 1e6)

        denom_shares = alpha + beta + gamma
        # d_agents and l_agents will be arrays of length n=10
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_term + d0
        price_leisure = phi * w * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_term
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
        labor_supply_agents = t - l_agents

        # Aggregations sum over n=10 households
        agg_labor = np.sum(phi * labor_supply_agents) # Weighted by phi
        agg_d = np.sum(d_agents) # Simple sum

        # Equations remain the same structure
        eq1 = t_c + t_d - agg_labor
        eq2 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r - 1)) * (f_c**(1 - r)); eq3 = w - MP_L_c
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r - 1)) * (f_c**(1 - r)); eq4 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r - 1)) * (f_d**(1 - r)); eq5 = w - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r - 1)) * (f_d**(1 - r)); eq6 = tau_z - MP_Z_d * p_d
        # Sum in gov budget constraint runs over n=10 households
        total_wage_tax_revenue = np.sum(tau_w * w * phi * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq7 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g) # n is now 10

        # Still 7 equations
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
    # --- End of nested function ---

    # Initial guess y0 still has length 7
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])

    # Solve the system (using 'lm' or try 'hybr')
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})

    # --- Post-processing ---
    t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    # Recalculate results (arrays like c_agents will be length n=10)
    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r) if sol.success else np.nan
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r) if sol.success else np.nan

    Y_term = phi * w * (1 - tau_w) * t + l - p_d * d0 if sol.success else np.full(n, np.nan)
    denom_shares = alpha + beta + gamma
    c_agents = (alpha / (p_c * denom_shares)) * Y_term if sol.success else np.full(n, np.nan)
    d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_term + d0 if sol.success else np.full(n, np.nan)
    price_leisure = phi * w * (1 - tau_w) if sol.success else np.full(n, np.nan)
    l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_term if sol.success else np.full(n, np.nan)
    l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
    labor_supply_agents = t - l_agents

    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi * labor_supply_agents)

    profit_c = f_c - w * t_c - tau_z * z_c if sol.success else np.nan
    profit_d = p_d * f_d - w * t_d - tau_z * z_d if sol.success else np.nan

    # Loops now run n=10 times
    budget_errors = np.zeros(n) * np.nan
    if sol.success:
        for i in range(n):
            income_i = phi[i] * w * (1 - tau_w[i]) * labor_supply_agents[i] + l
            expenditure_i = c_agents[i] + p_d * d_agents[i]
            budget_errors[i] = income_i - expenditure_i

    utilities = np.zeros(n) * np.nan
    if sol.success:
        for i in range(n):
            if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
                utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
            else:
                utilities[i] = -np.inf

    residuals = system_eqns(sol.x)

    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w": w, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, # length n=10
        "d_agents": d_agents, # length n=10
        "l_agents": l_agents, # length n=10
        "labor_supply_agents": labor_supply_agents, # length n=10
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors, # length n=10
        "utilities": utilities, # length n=10
        "residuals": residuals, # length 7
        "sol": sol
    }

    return sol.x, results, sol.success

# --- Input Section ---
# EXAMPLE tau_w for 10 households (pairing original values)
# ** REPLACE THIS WITH YOUR ACTUAL 10 TAX RATES **
tau_w_orig = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w = np.array([
    tau_w_orig[0], tau_w_orig[0],
    tau_w_orig[1], tau_w_orig[1],
    tau_w_orig[2], tau_w_orig[2],
    tau_w_orig[3], tau_w_orig[3],
    tau_w_orig[4], tau_w_orig[4]
])
# ---

tau_z = 1.0
g = 5.0

# Solve model
try:
    solution, results, converged = solve(tau_w, tau_z, g)

    # --- Output Section ---
    print("\n" + "="*50)
    print(f"Model Run Results (n={n} Households, Homogenous Market)")
    print("="*50)
    print("solution status:", results["sol"].status)
    print("solution message:", results["sol"].message)
    print("convergence:", converged)
    if not converged:
         print("WARNING: Solver did NOT converge.")

    print("\nsolution vector [t_c, t_d, log_z_c, log_z_d, w, p_d, l]:")
    print(np.round(solution, 5))

    print("\nproduction summary:")
    print(f"sector c: t_prod = {results.get('t_c', np.nan):.4f}, z_c = {results.get('z_c', np.nan):.4f}, f_c = {results.get('f_c', np.nan):.4f}")
    print(f"sector d: t_prod = {results.get('t_d', np.nan):.4f}, z_d = {results.get('z_d', np.nan):.4f}, f_d = {results.get('f_d', np.nan):.4f}")
    print(f"Wage: w = {results.get('w', np.nan):.4f}")
    print(f"Dirty Good Price: p_d = {results.get('p_d', np.nan):.4f}")
    print(f"Lump-sum Transfer: l = {results.get('l', np.nan):.4f}")


    # Print loops run n=10 times
    print("\nhousehold demands and leisure:")
    print(f"{'Household':<10} {'Phi':<8} {'Cons (c)':<10} {'Cons (d)':<10} {'Leisure':<10}")
    print("-" * 55)
    for i in range(n):
         c_val = results.get('c_agents', [np.nan]*n)[i]; d_val = results.get('d_agents', [np.nan]*n)[i]
         l_val = results.get('l_agents', [np.nan]*n)[i]
         print(f"{i+1:<10} {phi[i]:<8.5f} {c_val:<10.4f} {d_val:<10.4f} {l_val:<10.4f}")

    print("\nhousehold utilities:")
    utilities = results.get('utilities', [np.nan]*n)
    for i in range(n):
        print(f"household {i+1}: utility = {utilities[i]:.4f}")

    # Add Income Analysis if needed (will now show 10 households)
    if converged:
        print("\n" + "="*30); print("      Income Analysis"); print("="*30)
        w = results['w']; l = results['l']
        labor_supply_agents = results['labor_supply_agents']
        net_labor_income = phi * w * (1 - tau_w) * labor_supply_agents
        total_disposable_income = net_labor_income + l
        print("\nHousehold Disposable Income Breakdown:")
        print("-" * 70)
        print(f"{'Household':<10} {'Phi':<10} {'Net Labor Inc':<15} {'Transfer (l)':<15} {'Total Income':<15}")
        print("-" * 70)
        for i in range(n):
            print(f"{i+1:<10} {phi[i]:<10.4f} {net_labor_income[i]:<15.4f} {l:<15.4f} {total_disposable_income[i]:<15.4f}")
        print("-" * 70)
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