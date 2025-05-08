# plot_revenue_difference_2scen.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
# Assuming running in same folder as solvers:
import b_outer_solver
import a_inner_solver as solver
from a_inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0 # Define theta locally

# Define the fixed tau_w rates for the fixed-tau_w scenario
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])

# Define the range of xi values to test
xi_values = np.linspace(0.1, 1.0, 25) # Using 25 points as in the provided script
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


# Lists to store results (same as before)
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

    # Scenario 1: Variable tau_w
    try:
        opt_tau_w, opt_tau_z, _ = b_outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            _, results, converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
            if converged:
                current_rev_opt_w_env = opt_tau_z * (results['z_c'] + results['z_d'])
                hours_worked_i = T - results['l_agents']
                hours_worked_i = np.maximum(hours_worked_i, 0)
                gross_labor_income_i = phi * results['w'] * hours_worked_i
                current_rev_opt_w_inc = np.sum(opt_tau_w * gross_labor_income_i)
                current_rev_opt_w_tot = current_rev_opt_w_env + current_rev_opt_w_inc
    except Exception as e: print(f"      Error during Scenario 1 for xi = {xi_val:.4f}: {e}")
    rev_opt_w_env.append(current_rev_opt_w_env)
    rev_opt_w_inc.append(current_rev_opt_w_inc)
    rev_opt_w_tot.append(current_rev_opt_w_tot)

    # Scenario 2: Fixed Optimal (xi=0.1) tau_w
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
    except Exception as e: print(f"      Error during Scenario 2 for xi = {xi_val:.4f}: {e}")
    rev_fix_opt01_env.append(current_rev_fix_opt01_env)
    rev_fix_opt01_inc.append(current_rev_fix_opt01_inc)
    rev_fix_opt01_tot.append(current_rev_fix_opt01_tot)

print("-" * 30)
print("Simulations complete.")

# --- Calculate Differences ---

# Convert lists to numpy arrays first
rev_opt_w_env = np.array(rev_opt_w_env); rev_opt_w_inc = np.array(rev_opt_w_inc); rev_opt_w_tot = np.array(rev_opt_w_tot)
rev_fix_opt01_env = np.array(rev_fix_opt01_env); rev_fix_opt01_inc = np.array(rev_fix_opt01_inc); rev_fix_opt01_tot = np.array(rev_fix_opt01_tot)
valid_xi_values_processed = np.array(valid_xi_values_processed)

# Calculate the difference (Variable - Fixed)
# np.subtract handles potential NaNs gracefully (result is NaN if either operand is NaN)
diff_rev_tot = np.subtract(rev_opt_w_tot, rev_fix_opt01_tot)
diff_rev_env = np.subtract(rev_opt_w_env, rev_fix_opt01_env)
diff_rev_inc = np.subtract(rev_opt_w_inc, rev_fix_opt01_inc)


# --- Plotting Differences ---

fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted figsize

# Define colors, styles, labels for the DIFFERENCE plots
color_diff_tot = 'tab:blue';  ls_diff_tot = '-';  lw_diff_tot = 2.0; label_diff_tot = 'Total Rev. Diff.'
color_diff_env = 'tab:green'; ls_diff_env = '--'; lw_diff_env = 1.5; label_diff_env = 'Env. Rev. Diff.'
color_diff_inc = 'tab:red';   ls_diff_inc = ':';  lw_diff_inc = 1.5; label_diff_inc = 'Inc. Rev. Diff.'

# Plot Lines - Filter NaNs for each difference array
valid_diff_tot_idx = ~np.isnan(diff_rev_tot)
valid_diff_env_idx = ~np.isnan(diff_rev_env)
valid_diff_inc_idx = ~np.isnan(diff_rev_inc)

if np.any(valid_diff_tot_idx):
    ax.plot(valid_xi_values_processed[valid_diff_tot_idx],
            diff_rev_tot[valid_diff_tot_idx],
            linestyle=ls_diff_tot, color=color_diff_tot, label=label_diff_tot, linewidth=lw_diff_tot)
if np.any(valid_diff_env_idx):
    ax.plot(valid_xi_values_processed[valid_diff_env_idx],
            diff_rev_env[valid_diff_env_idx],
            linestyle=ls_diff_env, color=color_diff_env, label=label_diff_env, linewidth=lw_diff_env)
if np.any(valid_diff_inc_idx):
    ax.plot(valid_xi_values_processed[valid_diff_inc_idx],
            diff_rev_inc[valid_diff_inc_idx],
            linestyle=ls_diff_inc, color=color_diff_inc, label=label_diff_inc, linewidth=lw_diff_inc)

# Add a horizontal line at y=0 for reference
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, label='Zero Difference')

# Add labels and title
ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel('Revenue Difference (Variable - Fixed Opt)', fontsize=12) # Updated Y-axis label
ax.set_title('Difference in Tax Revenues between Scenarios', fontsize=16) # Updated Title

# Add legend (adjust position as needed, e.g., 'best')
ax.legend(loc='best')

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# New filename for the difference plot
output_filename = "revenue_difference_var_vs_fixedopt01.pdf"
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path) # No need for bbox_inches='tight' if legend is inside plot
print(f"\nDifference plot saved to {output_path}")

plt.show()