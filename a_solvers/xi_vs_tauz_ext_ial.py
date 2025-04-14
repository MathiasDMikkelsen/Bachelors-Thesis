# plot_tau_z_comparison.py (Adding third scenario with optimal fixed tau_w)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import outer_solver_ext as outer_solver # Imports the outer solver with maximize_welfare(G, xi)
import inner_solver_ext as solver # Import inner solver directly too
# Removed theta from import, defined locally below
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params
import os # For saving figure

# --- Simulation Parameters ---
G_value = 5.0
# Ensure theta is defined locally
theta = 1.0

# Define the fixed tau_w sets for scenario 2 and 3
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.08858208, -0.04377549,  0.22144972,  0.39697164,  0.64084534])

# Define the range and number of xi values to test
# Using the xi_values from your previous code version
xi_values = np.linspace(0.01, 4.0, 10)
# --- End Simulation Parameters ---

# --- Function to optimize ONLY tau_z for FIXED tau_w (Unchanged) ---
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    """
    Optimizes social welfare by choosing only tau_z, given fixed G, xi, and tau_w.
    """
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged:
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities[utilities > -1e5]
            welfare = np.sum(valid_utilities) - 5*xi_val * (agg_polluting**theta)
            return -welfare
        except Exception as e:
            return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [0.5]
    try:
        result = minimize(swf_obj_fixed_w,
                          initial_tau_z_guess,
                          args=(G, xi, fixed_tau_w_arr),
                          method='SLSQP',
                          bounds=tau_z_bounds,
                          options={'disp': False, 'ftol': 1e-7})
        if result.success:
            return result.x[0], -result.fun
        else:
            return None, None
    except Exception as e:
        return None, None

# Lists to store results for all 3 scenarios
tau_z_optimal_w_results = []
tau_z_fixed_preexisting_results = [] # Renamed list
tau_z_fixed_optimal_xi01_results = [] # <-- Added list for scenario 3
valid_xi_optimal_w = []
valid_xi_fixed_preexisting = [] # Renamed list
valid_xi_fixed_optimal_xi01 = [] # <-- Added list for scenario 3

# --- Scenario 1: Optimal tau_w and tau_z ---
print("Running Scenario 1: Variable tau_w...")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing all for xi = {xi_val:.4f}...") # Optional print
    try:
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            tau_z_optimal_w_results.append(opt_tau_z)
            valid_xi_optimal_w.append(xi_val)
        else:
            tau_z_optimal_w_results.append(np.nan)
            valid_xi_optimal_w.append(xi_val)
    except Exception as e:
        print(f"    Error during Scenario 1 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_optimal_w_results.append(np.nan)
        valid_xi_optimal_w.append(xi_val)
print("Scenario 1 finished.")

# --- Scenario 2: Fixed tau_w (Pre-existing), Optimal tau_z ---
print("\nRunning Scenario 2: Fixed tau_w (Pre-existing)...")
print(f"(Using fixed tau_w = {fixed_tau_w_preexisting})")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Optional print
    try:
        # Call with pre-existing fixed tau_w
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
        if opt_tau_z is not None:
            tau_z_fixed_preexisting_results.append(opt_tau_z)
            valid_xi_fixed_preexisting.append(xi_val)
        else:
            tau_z_fixed_preexisting_results.append(np.nan)
            valid_xi_fixed_preexisting.append(xi_val)
    except Exception as e:
        print(f"    Error during Scenario 2 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_preexisting_results.append(np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
print("Scenario 2 finished.")

# --- Scenario 3: Fixed tau_w (Optimal at xi=0.1), Optimal tau_z ---
print("\nRunning Scenario 3: Fixed tau_w (Optimal at xi=0.1)...")
print(f"(Using fixed tau_w = {fixed_tau_w_optimal_xi01})")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Optional print
    try:
        # Call with optimal fixed tau_w
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
        if opt_tau_z is not None:
            tau_z_fixed_optimal_xi01_results.append(opt_tau_z)
            valid_xi_fixed_optimal_xi01.append(xi_val)
        else:
            tau_z_fixed_optimal_xi01_results.append(np.nan)
            valid_xi_fixed_optimal_xi01.append(xi_val)
    except Exception as e:
        print(f"    Error during Scenario 3 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_optimal_xi01_results.append(np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)
print("Scenario 3 finished.")

print("-" * 30)
print("Simulations complete.")


# --- Plotting (3 lines, updated labels) ---

# Convert lists to numpy arrays
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w = np.array(valid_xi_optimal_w)
tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results) # Renamed
valid_xi_fixed_preexisting = np.array(valid_xi_fixed_preexisting) # Renamed
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results) # Added
valid_xi_fixed_optimal_xi01 = np.array(valid_xi_fixed_optimal_xi01) # Added

# Create the plot
plt.figure(figsize=(5, 3.5))

# Plot Lines - Filter NaNs and update labels
valid_opt_indices = ~np.isnan(tau_z_optimal_w_results)
valid_fixed_pre_indices = ~np.isnan(tau_z_fixed_preexisting_results)
valid_fixed_opt01_indices = ~np.isnan(tau_z_fixed_optimal_xi01_results)

if np.any(valid_opt_indices):
    plt.plot(valid_xi_optimal_w[valid_opt_indices],
             tau_z_optimal_w_results[valid_opt_indices],
             linestyle='-', label='Variable $\\tau_w$') # Updated label

if np.any(valid_fixed_pre_indices):
    plt.plot(valid_xi_fixed_preexisting[valid_fixed_pre_indices],
             tau_z_fixed_preexisting_results[valid_fixed_pre_indices],
             linestyle='--', label='Fixed $\\tau_w$ (Pre-existing)') # Updated label

if np.any(valid_fixed_opt01_indices):
    plt.plot(valid_xi_fixed_optimal_xi01[valid_fixed_opt01_indices],
             tau_z_fixed_optimal_xi01_results[valid_fixed_opt01_indices],
             linestyle=':', label='Fixed $\\tau_w$ (Optimal at $\\xi=0.1$)') # Added line and label

# Add labels
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\tau_z$', fontsize=14)

# Add legend
# Check if at least one line has valid data
if np.any(valid_opt_indices) or np.any(valid_fixed_pre_indices) or np.any(valid_fixed_opt01_indices):
    plt.legend(loc='best') # Changed loc back to 'best' for potentially 3 lines

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs" # Save in sub-folder relative to script execution
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tau_z_comparison_3_scenarios.pdf") # New filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()