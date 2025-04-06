# File: analyze_d0_segmented_many_values.py

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import copy

# ==============================================================
# Model Definition: Segmented Labor Market (2 Dirty + 3 Clean HH)
# Based on the user's provided code, modified to accept d0_val
# ==============================================================

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0 # Using single r as per user's code context for this run
t = 24.0
# d0 is now passed to solve()
epsilon_c = 0.995
epsilon_d = 0.92 # Using value from user code context
p_c = 1.0

# Productivity weights split by sector (Original 2 Dirty + 3 Clean STRUCTURE)
phi_d = np.array([0.15*2, 0.35*2])   # Dirty sector households (sum = 1.0)
phi_c = np.array([0.1*2, 0.2*2, 0.2*2])  # Clean sector households (sum = 1.0)
phi = np.concatenate([phi_d, phi_c])  # total phi for all 5 households (sum = 2.0)
n = len(phi)
n_d = len(phi_d) # n_d is 2
n_c = len(phi_c) # n_c is 3

print(f"Model structure: n_d = {n_d}, n_c = {n_c}, n = {n}")

# --- solve function modified to accept d0_val ---
# (Same function definition as in the previous response)
def solve(tau_w, tau_z, g, d0_val, y0_guess=None):
    """
    Solves the segmented model (2d+3c) for a given d0_val.
    Uses single 'r' parameter as per the provided base code.
    """
    if len(tau_w) != n:
        raise ValueError(f"Length of tau_w ({len(tau_w)}) must match number of households ({n})")

    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y
        z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)
        t_c_eff = max(t_c, 1e-9); t_d_eff = max(t_d, 1e-9)
        z_c_eff = max(z_c, 1e-9); z_d_eff = max(z_d, 1e-9)
        p_d_eff = max(p_d, 1e-9); w_c_eff = max(w_c, 1e-9); w_d_eff = max(w_d, 1e-9)

        try:
            f_c = (epsilon_c * (t_c_eff**r) + (1 - epsilon_c) * (z_c_eff**r))**(1/r)
            f_d = (epsilon_d * (t_d_eff**r) + (1 - epsilon_d) * (z_d_eff**r))**(1/r)
            wage = np.concatenate([w_d_eff * np.ones(n_d), w_c_eff * np.ones(n_c)])
            Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d_eff * d0_val
            if np.any(Y_tilde <= 1e-9): return np.full(8, 1e6)
            denom_shares = alpha + beta + gamma
            d_agents = (beta / (p_d_eff * denom_shares)) * Y_tilde + d0_val
            price_leisure = phi * wage * (1 - tau_w)
            if np.any(price_leisure <= 1e-9): return np.full(8, 1e6)
            l_agents = (gamma / (denom_shares * price_leisure)) * Y_tilde
            l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
            labor_supply_agents = t - l_agents
            agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d])
            agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:])
            agg_d = np.sum(d_agents)
            MP_L_c = epsilon_c * (t_c_eff**(r - 1)) * (f_c**(1 - r));
            MP_Z_c = (1 - epsilon_c) * (z_c_eff**(r - 1)) * (f_c**(1 - r));
            MP_L_d = epsilon_d * (t_d_eff**(r - 1)) * (f_d**(1 - r));
            MP_Z_d = (1 - epsilon_d) * (z_d_eff**(r - 1)) * (f_d**(1 - r));
            eq1 = t_c - agg_labor_c; eq2 = t_d - agg_labor_d
            eq3 = (agg_d + 0.5 * g / p_d_eff) - f_d
            eq4 = w_c - MP_L_c; eq5 = tau_z - MP_Z_c
            eq6 = w_d - MP_L_d * p_d; eq7 = tau_z - MP_Z_d * p_d
            total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
            total_z_tax_revenue = tau_z * (z_c + z_d)
            eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)
            eqs = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])
            if np.any(np.isnan(eqs)) or np.any(np.isinf(eqs)): return np.full(8, 1e12)
            return eqs
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            return np.full(8, 1e12)

    if y0_guess is None:
        y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])
    else:
        if np.any(np.isnan(y0_guess)) or np.any(np.isinf(y0_guess)):
            y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])
        else: y0 = y0_guess

    try:
        sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-9, 'ftol': 1e-9, 'maxiter':1000})
    except Exception as e:
        print(f"ERROR: root finding failed with exception: {e}")
        from scipy.optimize.optimize import OptimizeResult
        sol = OptimizeResult(x=np.full(len(y0), np.nan), success=False, status=-1, message=f"Exception: {e}")

    if not sol.success:
        nan_results = {k: np.nan for k in ["t_c", "t_d", "z_c", "z_d", "w_c", "w_d", "p_d", "l", "f_c", "f_d", "agg_c", "agg_d", "agg_labor", "profit_c", "profit_d"]}
        nan_results.update({k: np.full(n, np.nan) for k in ["c_agents", "d_agents", "l_agents", "labor_supply_agents", "budget_errors", "utilities", "Y_tilde"]})
        nan_results["sol"] = sol; nan_results["residuals"] = np.full(8, np.nan)
        nan_results["d0_used"] = d0_val
        return np.full(len(y0), np.nan), nan_results, False

    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)

    try:
        f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0_val
        denom_shares = alpha + beta + gamma
        c_agents = (alpha / (p_c * denom_shares)) * Y_tilde
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_tilde + d0_val
        price_leisure = phi * wage * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
        labor_supply_agents = t - l_agents
        agg_c = np.sum(c_agents); agg_d = np.sum(d_agents)
        agg_labor = np.sum(phi * labor_supply_agents)
        profit_c = p_c * f_c - w_c * t_c - tau_z * z_c
        profit_d = p_d * f_d - w_d * t_d - tau_z * z_d
        budget_errors = np.zeros(n)
        for i in range(n):
            income_i = phi[i] * wage[i] * (1 - tau_w[i]) * labor_supply_agents[i] + l
            expenditure_i = p_c * c_agents[i] + p_d * d_agents[i]
            budget_errors[i] = income_i - expenditure_i
        utilities = np.zeros(n)
        for i in range(n):
            if c_agents[i] > 1e-9 and (d_agents[i] - d0_val) > 1e-9 and l_agents[i] > 1e-9:
                utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0_val) + gamma * np.log(l_agents[i])
            else: utilities[i] = -np.inf
        residuals = system_eqns(sol.x)
        results = {
            "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
            "f_c": f_c, "f_d": f_d, "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
            "labor_supply_agents": labor_supply_agents, "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
            "profit_c": profit_c, "profit_d": profit_d, "budget_errors": budget_errors,
            "utilities": utilities, "residuals": residuals, "sol": sol, "Y_tilde": Y_tilde,
            "d0_used": d0_val
        }
        return sol.x, results, sol.success
    except (ValueError, OverflowError, ZeroDivisionError, RuntimeWarning) as e:
        print(f"ERROR: Post-processing failed for tau_z={tau_z} with exception: {e}")
        nan_results = {k: np.nan for k in ["t_c", "t_d", "z_c", "z_d", "w_c", "w_d", "p_d", "l", "f_c", "f_d", "agg_c", "agg_d", "agg_labor", "profit_c", "profit_d"]}
        nan_results.update({k: np.full(n, np.nan) for k in ["c_agents", "d_agents", "l_agents", "labor_supply_agents", "budget_errors", "utilities", "Y_tilde"]})
        nan_results["sol"] = sol; nan_results["residuals"] = np.full(8, np.nan)
        nan_results["d0_used"] = d0_val
        return sol.x, nan_results, False
# --- End of solve function ---


# ==============================================================
# Analysis Section
# ==============================================================

print("\n" + "="*50)
print("Running analysis over tau_z for different d0 values")
print("(Segmented Model: 2 Dirty + 3 Clean HH)")
print("="*50)

# --- Define ranges / values ---
# <<< Use more d0 values with smaller intervals >>>
d0_values_to_test = np.round(np.arange(0.1, 2.01, 0.1), 2) # d0 from 0.1 to 2.0 in steps of 0.1
# ---
print(f"Testing d0 values: {d0_values_to_test}")
print(f"Number of d0 values: {len(d0_values_to_test)}")

tau_z_range = np.linspace(0.1, 2.5, 30)  # Range for tau_z (keep number of points moderate)

# Use consistent tau_w and g
tau_w_loop = np.array([-1.75, -0.5, 0.0, 0.25, 0.55])
g_loop = 5.0
default_y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

# --- Storage for results ---
all_results_list = []

# --- Simulation Loops ---
# (Outer loop for d0, Inner loop for tau_z - structure remains the same)
for d0_current in d0_values_to_test:
    print(f"\n--- Running for d0 = {d0_current:.2f} ---") # Format printout
    last_successful_y = None

    for tau_z_val in tau_z_range:
        print(f"  Solving for tau_z = {tau_z_val:.3f} ... ", end="")
        current_guess = last_successful_y if last_successful_y is not None else default_y0
        try:
            solution_vec, results_dict, converged_flag = solve(
                tau_w_loop, tau_z_val, g_loop, d0_val=d0_current, y0_guess=current_guess
            )
            store_dict = {
                'tau_z': tau_z_val, 'd0_run': d0_current, 'converged': converged_flag,
                't_d': results_dict.get('t_d', np.nan), 't_c': results_dict.get('t_c', np.nan),
                'w_d': results_dict.get('w_d', np.nan), 'w_c': results_dict.get('w_c', np.nan),
                'p_d': results_dict.get('p_d', np.nan)
            }
            all_results_list.append(store_dict)
            if converged_flag:
                print(f"Converged. t_d = {results_dict.get('t_d', np.nan):.4f}")
                last_successful_y = solution_vec
            else:
                sol_obj = results_dict.get('sol', None)
                status = sol_obj.status if sol_obj else 'N/A'; msg = sol_obj.message if sol_obj else 'N/A'
                print(f"FAILED. Solver status: {status}, Msg: {msg}")
                last_successful_y = None
        except Exception as e:
            print(f"ERROR during solve call for d0={d0_current}, tau_z={tau_z_val}: {e}")
            traceback.print_exc()
            error_results = {'tau_z': tau_z_val, 'converged': False, 'd0_run': d0_current, 't_d': np.nan}
            all_results_list.append(error_results)
            last_successful_y = None


# --- Process and Display Results ---
if all_results_list:
    df_all = pd.DataFrame(all_results_list)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    converged_results = df_all[df_all['converged']].copy()

    if converged_results.empty:
        print("\nNo converged points found across all runs. Cannot plot.")
    else:
        print(f"\nPlotting results for {len(converged_results)} converged points.")
        cmap = plt.get_cmap('viridis') # Use a colormap for many lines
        num_lines = len(d0_values_to_test)
        colors = cmap(np.linspace(0, 1, num_lines))

        # Group by d0 and plot each group
        for idx, (d0_val, group) in enumerate(converged_results.groupby('d0_run')):
            if not group.empty:
                ax.plot(group['tau_z'], group['t_d'], marker='.', # Smaller marker
                        linestyle='-', linewidth=1.5, # Slightly thicker line
                        markersize=4, label=f'd0 = {d0_val:.2f}', color=colors[idx])
            else:
                print(f"Warning: No converged points found for d0 = {d0_val}")

        ax.set_title(f'Segmented Model (2 Dirty + 3 Clean HH): Labor in Dirty Sector (t_d) vs. tau_z')
        ax.set_xlabel('tau_z (Tax on Input z)')
        ax.set_ylabel('Labor in Dirty Sector (t_d)')
        # Adjust legend position and size if needed for many items
        ax.legend(title="Baseline Cons. (d0)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.show()
else:
    print("\nNo results were generated from the loops.")