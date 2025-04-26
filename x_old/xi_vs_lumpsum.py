# plot_lumpsum_vs_xi.py

import numpy as np
import matplotlib.pyplot as plt
import os
# Assuming script is in the same folder as solvers:
from scipy.optimize import minimize
import outer_solver
import inner_solver as solver
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params

# --- Simulation Parameters ---
G_value = 5.0
# Define theta locally (as it's not in inner_solver)
theta = 1.0

# Define the fixed tau_w sets for scenario 2 and 3
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])

# Define the range and number of xi values to test
xi_values = np.linspace(0.04, 1.0, 25) # Using value from your previous code
# --- End Simulation Parameters ---

# --- Helper Function to optimize ONLY tau_z for FIXED tau_w ---
# (Same as used in plot_tau_z_comparison.py and plot_pollution_vs_xi.py)
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
lumpsum_optimal_w = []
lumpsum_fixed_pre = []
lumpsum_fixed_opt01 = []
valid_xi_values_processed = []

print(f"Running optimization and lumpsum calculation for {len(xi_values)} values of xi...")
print("-" * 30)

# --- Combined Simulation Loop for all Scenarios ---
for xi_val in xi_values:
    print(f"  Processing xi = {xi_val:.4f}...")
    valid_xi_values_processed.append(xi_val)
    l_opt_w = np.nan # Default values for this xi
    l_fix_pre = np.nan
    l_fix_opt01 = np.nan

    # --- Scenario 1: Variable tau_w ---
    try:
        opt_tau_w, opt_tau_z, _ = outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            try:
                _, results, converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
                if converged: l_opt_w = results['l'] # <-- Extract lumpsum
            except Exception as e: print(f"    Scen 1: Inner solve error: {e}")
    except Exception as e: print(f"    Scen 1: Outer solve error: {e}")
    lumpsum_optimal_w.append(l_opt_w)

    # --- Scenario 2: Fixed tau_w (Pre-existing) ---
    try:
        opt_tau_z_fix_pre, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
        if opt_tau_z_fix_pre is not None:
            try:
                _, results, converged = solver.solve(fixed_tau_w_preexisting, opt_tau_z_fix_pre, G_value)
                if converged: l_fix_pre = results['l'] # <-- Extract lumpsum
            except Exception as e: print(f"    Scen 2: Inner solve error: {e}")
    except Exception as e: print(f"    Scen 2: Outer solve error: {e}")
    lumpsum_fixed_pre.append(l_fix_pre)

    # --- Scenario 3: Fixed tau_w (Optimal at xi=0.1) ---
    try:
        opt_tau_z_fix_opt01, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
        if opt_tau_z_fix_opt01 is not None:
            try:
                _, results, converged = solver.solve(fixed_tau_w_optimal_xi01, opt_tau_z_fix_opt01, G_value)
                if converged: l_fix_opt01 = results['l'] # <-- Extract lumpsum
            except Exception as e: print(f"    Scen 3: Inner solve error: {e}")
    except Exception as e: print(f"    Scen 3: Outer solve error: {e}")
    lumpsum_fixed_opt01.append(l_fix_opt01)

    # print(f"    Results l: OptW={l_opt_w:.4f}, FixPre={l_fix_pre:.4f}, FixOpt01={l_fix_opt01:.4f}") # Optional

print("-" * 30)
print("Simulations complete.")


# --- Plotting (Styled like CV plot example) ---

# Convert lists to numpy arrays
lumpsum_optimal_w = np.array(lumpsum_optimal_w)
lumpsum_fixed_pre = np.array(lumpsum_fixed_pre)
lumpsum_fixed_opt01 = np.array(lumpsum_fixed_opt01)
valid_xi_values_processed = np.array(valid_xi_values_processed)

# Create the plot with specified size
plt.figure(figsize=(5, 3.5))

# Plot Lines - Filter NaNs for each line individually
valid_opt_indices = ~np.isnan(lumpsum_optimal_w)
valid_fix_pre_indices = ~np.isnan(lumpsum_fixed_pre)
valid_fix_opt01_indices = ~np.isnan(lumpsum_fixed_opt01)

# Use same colors/styles as tau_z comparison plot for consistency
color_opt = 'tab:blue'
color_fix_pre = 'tab:red' # Changed from orange in original example
color_fix_opt01 = 'tab:purple' # Changed from green in original example

ls_opt = '-'
ls_fix_pre = '--'
ls_fix_opt01 = ':'

# Use same labels as tau_z comparison plot
label_opt = 'Variable $\\tau_w$'
label_fix_pre = 'Fixed $\\tau_w$ (Pre-existing)'
label_fix_opt01 = 'Fixed $\\tau_w$ (Optimal at $\\xi=0.1$)'


if np.any(valid_opt_indices):
    plt.plot(valid_xi_values_processed[valid_opt_indices],
             lumpsum_optimal_w[valid_opt_indices],
             linestyle=ls_opt, color=color_opt, label=label_opt)

if np.any(valid_fix_pre_indices):
    plt.plot(valid_xi_values_processed[valid_fix_pre_indices],
             lumpsum_fixed_pre[valid_fix_pre_indices],
             linestyle=ls_fix_pre, color=color_fix_pre, label=label_fix_pre)

if np.any(valid_fix_opt01_indices):
    plt.plot(valid_xi_values_processed[valid_fix_opt01_indices],
             lumpsum_fixed_opt01[valid_fix_opt01_indices],
             linestyle=ls_fix_opt01, color=color_fix_opt01, label=label_fix_opt01)

# Add labels with specified font size
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Lump-sum Transfer ($l$)', fontsize=14) # Updated label

# Add legend
# Check if at least one line has valid data
if np.any(valid_opt_indices) or np.any(valid_fix_pre_indices) or np.any(valid_fix_opt01_indices):
    plt.legend(loc='best')

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs" # Subdirectory relative to script location
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = "lumpsum_comparison_plot.pdf" # New filename
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()