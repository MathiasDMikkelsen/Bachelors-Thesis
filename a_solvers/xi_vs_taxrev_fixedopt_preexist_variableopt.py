# plot_revenue_comparison_3scen.py (Corrected version 2)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
# Assuming running in same folder as solvers:
import outer_solver
import inner_solver as solver
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0 # Define theta locally

# Define the fixed tau_w rates for Scenario 2 (Fixed Pre-existing)
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

# Define the fixed tau_w rates for Scenario 3 (Fixed Optimal at xi=0.1)
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])

# Define the range of xi values to test (Using the denser grid from the comparison script)
xi_values = np.linspace(0.1, 1.0, 50)
# --- End Simulation Parameters ---

# --- Helper Function to optimize ONLY tau_z for FIXED tau_w ---
# (Same as used in both original scripts)
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged: return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities[utilities > -1e5]
            # Ensure theta is accessible (it's defined globally in this script)
            welfare = np.sum(valid_utilities) - 5*xi_val * (agg_polluting**theta)
            return -welfare
        except Exception as e: return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [0.5]
    try:
        result = minimize(swf_obj_fixed_w, initial_tau_z_guess, args=(G, xi, fixed_tau_w_arr),
                          method='SLSQP', bounds=tau_z_bounds, options={'disp': False, 'ftol': 1e-7})
        if result.success: return result.x[0], -result.fun
        else: return None, None
    except Exception as e: return None, None
# --- End Helper Function ---


# Lists to store results for all 3 scenarios
rev_opt_w_env = []      # Scenario 1: Variable tau_w
rev_opt_w_inc = []
rev_opt_w_tot = []
rev_fix_preex_env = []  # Scenario 2: Fixed Pre-existing tau_w
rev_fix_preex_inc = []
rev_fix_preex_tot = []
rev_fix_opt01_env = []  # Scenario 3: Fixed Optimal (xi=0.1) tau_w
rev_fix_opt01_inc = []
rev_fix_opt01_tot = []
valid_xi_values_processed = [] # Stores xi values for which results were obtained

print(f"Running revenue comparison for 3 scenarios...")
print(f"  1. Variable tau_w")
print(f"  2. Fixed Pre-existing tau_w = {fixed_tau_w_preexisting}")
print(f"  3. Fixed Optimal (xi=0.1) tau_w = {fixed_tau_w_optimal_xi01}")
print("-" * 30)

# --- Combined Simulation Loop ---
for xi_val in xi_values:
    # print(f"  Processing xi = {xi_val:.4f}...") # Optional print
    valid_xi_values_processed.append(xi_val)

    # Initialize results for this xi value to NaN
    current_rev_opt_w_env = np.nan; current_rev_opt_w_inc = np.nan; current_rev_opt_w_tot = np.nan
    current_rev_fix_preex_env = np.nan; current_rev_fix_preex_inc = np.nan; current_rev_fix_preex_tot = np.nan
    current_rev_fix_opt01_env = np.nan; current_rev_fix_opt01_inc = np.nan; current_rev_fix_opt01_tot = np.nan

    # --- Scenario 1: Variable tau_w ---
    try:
        opt_tau_w, opt_tau_z, _ = outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            _, results, converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
            if converged:
                current_rev_opt_w_env = opt_tau_z * (results['z_c'] + results['z_d'])
                hours_worked_i = T - results['l_agents']
                hours_worked_i = np.maximum(hours_worked_i, 0)
                gross_labor_income_i = phi * results['w'] * hours_worked_i
                current_rev_opt_w_inc = np.sum(opt_tau_w * gross_labor_income_i)
                current_rev_opt_w_tot = current_rev_opt_w_env + current_rev_opt_w_inc
    except Exception as e: print(f"      Error during Scenario 1 (Var tau_w) for xi = {xi_val:.4f}: {e}")
    rev_opt_w_env.append(current_rev_opt_w_env)
    rev_opt_w_inc.append(current_rev_opt_w_inc)
    rev_opt_w_tot.append(current_rev_opt_w_tot)

    # --- Scenario 2: Fixed Pre-existing tau_w ---
    try:
        # Step 1: Find optimal tau_z for the fixed pre-existing tau_w
        opt_tau_z_fix_preex, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)

        if opt_tau_z_fix_preex is not None:
            # Step 2: Re-solve inner solver with fixed pre-existing tau_w and optimal tau_z
            _, results, converged = solver.solve(fixed_tau_w_preexisting, opt_tau_z_fix_preex, G_value)
            if converged:
                # Step 3: Calculate Revenues
                current_rev_fix_preex_env = opt_tau_z_fix_preex * (results['z_c'] + results['z_d'])
                hours_worked_i = T - results['l_agents']
                hours_worked_i = np.maximum(hours_worked_i, 0)
                gross_labor_income_i = phi * results['w'] * hours_worked_i
                # *** Use the FIXED pre-existing tau_w for income revenue calculation ***
                current_rev_fix_preex_inc = np.sum(fixed_tau_w_preexisting * gross_labor_income_i)
                current_rev_fix_preex_tot = current_rev_fix_preex_env + current_rev_fix_preex_inc
            # else: print(f"      Warning: Inner solver failed for optimal tau_z (Fixed Preex) at xi = {xi_val:.4f}")
        # else: print(f"      Warning: Outer Optimization failed for Fixed Preex tau_w at xi = {xi_val:.4f}")
    except Exception as e: print(f"      Error during Scenario 2 (Fixed Preex) for xi = {xi_val:.4f}: {e}")
    rev_fix_preex_env.append(current_rev_fix_preex_env)
    rev_fix_preex_inc.append(current_rev_fix_preex_inc)
    rev_fix_preex_tot.append(current_rev_fix_preex_tot)

    # --- Scenario 3: Fixed Optimal (xi=0.1) tau_w ---
    try:
        opt_tau_z_fix_opt01, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
        if opt_tau_z_fix_opt01 is not None:
            _, results, converged = solver.solve(fixed_tau_w_optimal_xi01, opt_tau_z_fix_opt01, G_value)
            if converged:
                current_rev_fix_opt01_env = opt_tau_z_fix_opt01 * (results['z_c'] + results['z_d'])
                hours_worked_i = T - results['l_agents']
                hours_worked_i = np.maximum(hours_worked_i, 0)
                gross_labor_income_i = phi * results['w'] * hours_worked_i
                current_rev_fix_opt01_inc = np.sum(fixed_tau_w_optimal_xi01 * gross_labor_income_i)
                current_rev_fix_opt01_tot = current_rev_fix_opt01_env + current_rev_fix_opt01_inc
            # else: print(f"      Warning: Inner solver failed for optimal tau_z (Fixed Opt01) at xi = {xi_val:.4f}")
        # else: print(f"      Warning: Outer Optimization failed for Fixed Opt01 tau_w at xi = {xi_val:.4f}")
    except Exception as e: print(f"      Error during Scenario 3 (Fixed Opt01) for xi = {xi_val:.4f}: {e}")
    rev_fix_opt01_env.append(current_rev_fix_opt01_env)
    rev_fix_opt01_inc.append(current_rev_fix_opt01_inc)
    rev_fix_opt01_tot.append(current_rev_fix_opt01_tot)


print("-" * 30)
print("Simulations complete.")

# --- Plotting ---

# Convert lists to numpy arrays
rev_opt_w_env = np.array(rev_opt_w_env); rev_opt_w_inc = np.array(rev_opt_w_inc); rev_opt_w_tot = np.array(rev_opt_w_tot)
rev_fix_preex_env = np.array(rev_fix_preex_env); rev_fix_preex_inc = np.array(rev_fix_preex_inc); rev_fix_preex_tot = np.array(rev_fix_preex_tot)
rev_fix_opt01_env = np.array(rev_fix_opt01_env); rev_fix_opt01_inc = np.array(rev_fix_opt01_inc); rev_fix_opt01_tot = np.array(rev_fix_opt01_tot)
valid_xi_values_processed = np.array(valid_xi_values_processed)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7.5)) # Slightly taller figure to accommodate legend

# Define colors, styles, labels for ALL scenarios
# Scenario 1: Variable tau_w
color_opt_tot = 'tab:blue';   ls_opt_tot = '-'; lw_opt_tot = 2.5; label_opt_tot = 'Total Rev. (Var $\\tau_w$)'
color_opt_env = 'tab:green';  ls_opt_env = '--'; lw_opt_env = 1.5; label_opt_env = 'Env. Rev. (Var $\\tau_w$)'
color_opt_inc = 'tab:red';    ls_opt_inc = ':';  lw_opt_inc = 1.5; label_opt_inc = 'Inc. Rev. (Var $\\tau_w$)'
# Scenario 2: Fixed Pre-existing tau_w
color_fix_pre_tot = 'black';    ls_fix_pre_tot = '-';  lw_fix_pre_tot = 1.0; label_fix_pre_tot = 'Total Rev. (Fixed Preex $\\tau_w$)'
color_fix_pre_env = 'dimgray';  ls_fix_pre_env = '--'; lw_fix_pre_env = 1.0; label_fix_pre_env = 'Env. Rev. (Fixed Preex $\\tau_w$)'
color_fix_pre_inc = 'lightgray';ls_fix_pre_inc = ':';  lw_fix_pre_inc = 1.5; label_fix_pre_inc = 'Inc. Rev. (Fixed Preex $\\tau_w$)'
# Scenario 3: Fixed Optimal (xi=0.1) tau_w
color_fix_opt_tot = 'tab:purple'; ls_fix_opt_tot = '-'; lw_fix_opt_tot = 1.0; label_fix_opt_tot = 'Total Rev. (Fixed Opt $\\tau_w$)'
color_fix_opt_env = 'tab:orange'; ls_fix_opt_env = '--'; lw_fix_opt_env = 1.0; label_fix_opt_env = 'Env. Rev. (Fixed Opt $\\tau_w$)'
color_fix_opt_inc = 'tab:brown';  ls_fix_opt_inc = ':';  lw_fix_opt_inc = 1.0; label_fix_inc = 'Inc. Rev. (Fixed Opt $\\tau_w$)' # Corrected variable name here
# Government Spending
color_g = 'grey'; label_g = f'G={G_value}'

# Plot Lines - Filter NaNs for each scenario
valid_opt_env_idx = ~np.isnan(rev_opt_w_env); valid_opt_inc_idx = ~np.isnan(rev_opt_w_inc); valid_opt_tot_idx = ~np.isnan(rev_opt_w_tot)
valid_fix_preex_env_idx = ~np.isnan(rev_fix_preex_env); valid_fix_preex_inc_idx = ~np.isnan(rev_fix_preex_inc); valid_fix_preex_tot_idx = ~np.isnan(rev_fix_preex_tot)
valid_fix_opt_env_idx = ~np.isnan(rev_fix_opt01_env); valid_fix_opt_inc_idx = ~np.isnan(rev_fix_opt01_inc); valid_fix_opt_tot_idx = ~np.isnan(rev_fix_opt01_tot)

# Plot Scenario 1
if np.any(valid_opt_env_idx): ax.plot(valid_xi_values_processed[valid_opt_env_idx], rev_opt_w_env[valid_opt_env_idx], linestyle=ls_opt_env, color=color_opt_env, label=label_opt_env, linewidth=lw_opt_env)
if np.any(valid_opt_inc_idx): ax.plot(valid_xi_values_processed[valid_opt_inc_idx], rev_opt_w_inc[valid_opt_inc_idx], linestyle=ls_opt_inc, color=color_opt_inc, label=label_opt_inc, linewidth=lw_opt_inc)
if np.any(valid_opt_tot_idx): ax.plot(valid_xi_values_processed[valid_opt_tot_idx], rev_opt_w_tot[valid_opt_tot_idx], linestyle=ls_opt_tot, color=color_opt_tot, label=label_opt_tot, linewidth=lw_opt_tot)
# Plot Scenario 2
if np.any(valid_fix_preex_env_idx): ax.plot(valid_xi_values_processed[valid_fix_preex_env_idx], rev_fix_preex_env[valid_fix_preex_env_idx], linestyle=ls_fix_pre_env, color=color_fix_pre_env, label=label_fix_pre_env, linewidth=lw_fix_pre_env)
if np.any(valid_fix_preex_inc_idx): ax.plot(valid_xi_values_processed[valid_fix_preex_inc_idx], rev_fix_preex_inc[valid_fix_preex_inc_idx], linestyle=ls_fix_pre_inc, color=color_fix_pre_inc, label=label_fix_pre_inc, linewidth=lw_fix_pre_inc)
if np.any(valid_fix_preex_tot_idx): ax.plot(valid_xi_values_processed[valid_fix_preex_tot_idx], rev_fix_preex_tot[valid_fix_preex_tot_idx], linestyle=ls_fix_pre_tot, color=color_fix_pre_tot, label=label_fix_pre_tot, linewidth=lw_fix_pre_tot)

# Plot Scenario 3
if np.any(valid_fix_opt_env_idx):
    ax.plot(valid_xi_values_processed[valid_fix_opt_env_idx],
            rev_fix_opt01_env[valid_fix_opt_env_idx],
            linestyle=ls_fix_opt_env, color=color_fix_opt_env, label=label_fix_opt_env, linewidth=lw_fix_opt_env)
if np.any(valid_fix_opt_inc_idx):
     # *** Corrected this line ***
     ax.plot(valid_xi_values_processed[valid_fix_opt_inc_idx],
             rev_fix_opt01_inc[valid_fix_opt_inc_idx],
             linestyle=ls_fix_opt_inc, color=color_fix_opt_inc, label=label_fix_inc, linewidth=lw_fix_opt_inc)
if np.any(valid_fix_opt_tot_idx):
    ax.plot(valid_xi_values_processed[valid_fix_opt_tot_idx],
            rev_fix_opt01_tot[valid_fix_opt_tot_idx],
            linestyle=ls_fix_opt_tot, color=color_fix_opt_tot, label=label_fix_opt_tot, linewidth=lw_fix_opt_tot)


# Plot Government Spending Line
ax.axhline(y=G_value, color=color_g, linestyle='--', linewidth=1.5, label=label_g)

# Add labels
ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel('Tax Revenue', fontsize=14)
ax.set_title('Tax Revenue Comparison across Scenarios', fontsize=16)

# Add legend below plot (Adjust ncol and bbox_to_anchor as needed)
# ncol=5 fits 10 items neatly in 2 rows
handles, labels = ax.get_legend_handles_labels()
if handles: # Check if there are any lines plotted to create a legend for
    ax.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), # Adjust vertical position (-0.15)
              ncol=5, fontsize='small')

# Adjust layout to make room for legend below plot
plt.subplots_adjust(bottom=0.25) # Increase bottom margin if needed

# Save the figure
output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# New filename for the combined plot
output_filename = "revenue_comparison_3scenarios_legend_below.pdf"
output_path = os.path.join(output_dir, output_filename)
# Use bbox_inches='tight' to ensure the legend is included in the saved file
plt.savefig(output_path, bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

plt.show()