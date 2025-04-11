# plot_revenue_comparison_2scen.py (Legend below plot in 2 rows)

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

# Define the fixed tau_w rates for the fixed-tau_w scenario
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])

# Define the range of xi values to test
xi_values = np.linspace(0.1, 1.0, 50)
# --- End Simulation Parameters ---

# --- Helper Function to optimize ONLY tau_z for FIXED tau_w ---
# (Same as before)
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged: return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities[utilities > -1e5]
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


# Lists to store results
rev_opt_w_env = []
rev_opt_w_inc = []
rev_opt_w_tot = []
rev_fix_opt01_env = []
rev_fix_opt01_inc = []
rev_fix_opt01_tot = []
valid_xi_values_processed = []

print(f"Running revenue comparison for 2 scenarios (Variable vs. Fixed Optimal tau_w)...")
print("-" * 30)

# --- Combined Simulation Loop ---
# (Same loop as before to calculate all results)
for xi_val in xi_values:
    # print(f"  Processing xi = {xi_val:.4f}...") # Optional print
    valid_xi_values_processed.append(xi_val)
    current_rev_opt_w_env = np.nan; current_rev_opt_w_inc = np.nan; current_rev_opt_w_tot = np.nan
    current_rev_fix_opt01_env = np.nan; current_rev_fix_opt01_inc = np.nan; current_rev_fix_opt01_tot = np.nan

    # Scenario 1
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
    except Exception as e: print(f"    Error during Scenario 1 for xi = {xi_val:.4f}: {e}")
    rev_opt_w_env.append(current_rev_opt_w_env)
    rev_opt_w_inc.append(current_rev_opt_w_inc)
    rev_opt_w_tot.append(current_rev_opt_w_tot)

    # Scenario 3
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
    except Exception as e: print(f"    Error during Scenario 3 for xi = {xi_val:.4f}: {e}")
    rev_fix_opt01_env.append(current_rev_fix_opt01_env)
    rev_fix_opt01_inc.append(current_rev_fix_opt01_inc)
    rev_fix_opt01_tot.append(current_rev_fix_opt01_tot)

print("-" * 30)
print("Simulations complete.")

# --- Plotting ---

# Convert lists to numpy arrays
rev_opt_w_env = np.array(rev_opt_w_env); rev_opt_w_inc = np.array(rev_opt_w_inc); rev_opt_w_tot = np.array(rev_opt_w_tot)
rev_fix_opt01_env = np.array(rev_fix_opt01_env); rev_fix_opt01_inc = np.array(rev_fix_opt01_inc); rev_fix_opt01_tot = np.array(rev_fix_opt01_tot)
valid_xi_values_processed = np.array(valid_xi_values_processed)

# Create the plot - *** Reverted to original figsize ***
fig, ax = plt.subplots(figsize=(10, 7.0))

# Define colors, styles, labels
color_opt_tot = 'tab:blue';   ls_opt_tot = '-'; lw_opt_tot = 2; label_opt_tot = 'Total Rev. (Var $\\tau_w$)'
color_opt_env = 'tab:green';  ls_opt_env = '--'; label_opt_env = 'Env. Rev. (Var $\\tau_w$)'
color_opt_inc = 'tab:red';    ls_opt_inc = ':'; label_opt_inc = 'Inc. Rev. (Var $\\tau_w$)'
color_fix_tot = 'tab:purple'; ls_fix_tot = '-'; lw_fix_tot = 1; label_fix_tot = 'Total Rev. (Fixed $\\tau_w$ Opt)'
color_fix_env = 'tab:orange'; ls_fix_env = '--'; label_fix_env = 'Env. Rev. (Fixed $\\tau_w$ Opt)'
color_fix_inc = 'tab:brown';  ls_fix_inc = ':'; label_fix_inc = 'Inc. Rev. (Fixed $\\tau_w$ Opt)'
color_g = 'grey'; label_g = f'G={G_value}'

# Plot Lines - Filter NaNs for each
valid_opt_env_idx = ~np.isnan(rev_opt_w_env); valid_opt_inc_idx = ~np.isnan(rev_opt_w_inc); valid_opt_tot_idx = ~np.isnan(rev_opt_w_tot)
valid_fix_env_idx = ~np.isnan(rev_fix_opt01_env); valid_fix_inc_idx = ~np.isnan(rev_fix_opt01_inc); valid_fix_tot_idx = ~np.isnan(rev_fix_opt01_tot)

if np.any(valid_opt_env_idx): ax.plot(valid_xi_values_processed[valid_opt_env_idx], rev_opt_w_env[valid_opt_env_idx], linestyle=ls_opt_env, color=color_opt_env, label=label_opt_env)
if np.any(valid_opt_inc_idx): ax.plot(valid_xi_values_processed[valid_opt_inc_idx], rev_opt_w_inc[valid_opt_inc_idx], linestyle=ls_opt_inc, color=color_opt_inc, label=label_opt_inc)
if np.any(valid_opt_tot_idx): ax.plot(valid_xi_values_processed[valid_opt_tot_idx], rev_opt_w_tot[valid_opt_tot_idx], linestyle=ls_opt_tot, color=color_opt_tot, label=label_opt_tot, linewidth=lw_opt_tot)
if np.any(valid_fix_env_idx): ax.plot(valid_xi_values_processed[valid_fix_env_idx], rev_fix_opt01_env[valid_fix_env_idx], linestyle=ls_fix_env, color=color_fix_env, label=label_fix_env)
if np.any(valid_fix_inc_idx): ax.plot(valid_xi_values_processed[valid_fix_inc_idx], rev_fix_opt01_inc[valid_fix_inc_idx], linestyle=ls_fix_inc, color=color_fix_inc, label=label_fix_inc)
if np.any(valid_fix_tot_idx): ax.plot(valid_xi_values_processed[valid_fix_tot_idx], rev_fix_opt01_tot[valid_fix_tot_idx], linestyle=ls_fix_tot, color=color_fix_tot, label=label_fix_tot, linewidth=lw_fix_tot)
ax.axhline(y=G_value, color=color_g, linestyle='--', linewidth=1.5, label=label_g)

# Add labels
ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel('Tax Revenue', fontsize=14)

# Add legend *** Positioned below plot in 4 columns ***
# Check if at least one line has valid data
if np.any([valid_opt_env_idx, valid_opt_inc_idx, valid_opt_tot_idx, valid_fix_env_idx, valid_fix_inc_idx, valid_fix_tot_idx]):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), # Anchor below axes (y<0), centered horizontally (x=0.5)
              ncol=4, fontsize='small') # Arrange in 4 columns (fits 7 items in 2 rows)

# Adjust layout to make room for legend below plot
plt.subplots_adjust(bottom=0.25) # Increase bottom margin (adjust value as needed)

# Save the figure
output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = "revenue_comparison_var_vs_fixedopt01_legend_below.pdf" # New filename
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches='tight' to include legend
print(f"\nPlot saved to {output_path}")

plt.show()