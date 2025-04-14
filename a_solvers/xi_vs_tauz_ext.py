# plot_tau_z_comparison_2hh.py (Modified for 2-household solvers)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import os # For saving figure

# --- MODIFIED: Import correct solvers and parameters ---
try:
    import outer_solver_ext as outer_solver # Imports the 2-hh outer solver
    import inner_solver_ext as solver      # Imports the 2-hh inner solver
    # Import necessary params from the 2-hh inner solver
    from inner_solver_ext import n, alpha, beta, gamma, d0, t as T, p_c
except ImportError:
    print("Error: Could not import 'inner_solver_ext' or 'outer_solver_ext'.")
    print("Ensure these files exist, are correctly named, and accessible.")
    exit()
# -------------------------------------------------------

# --- Assertions for 2-household model ---
assert n == 2, "This script requires the imported solvers to have n=2 households."
# ----------------------------------------

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0 # Pollution exponent in SWF

# Define the range and number of xi values to test
xi_values = np.linspace(0.1, 3.0, 50) # Start from xi=0.0 for broader range, 11 points

# --- Determine Fixed tau_w arrays for Scenarios 2 & 3 ---

# Scenario 2: Define a simple fixed tau_w for the 2 households
fixed_tau_w_scenario2 = np.array([0.0, 0.0]) # Example: [tau_w_d, tau_w_c]
print(f"Using fixed tau_w for Scenario 2: {fixed_tau_w_scenario2}")

# Scenario 3: Find the optimal tau_w at xi=0.1 to use as fixed values
print("\nFinding optimal tau_w at xi=0.1 for Scenario 3...")
xi_for_fixed_opt = 0.1
opt_tau_w_at_xi01, _, _ = outer_solver.maximize_welfare(G_value, xi_for_fixed_opt)

fixed_tau_w_scenario3 = None
if opt_tau_w_at_xi01 is not None:
    fixed_tau_w_scenario3 = opt_tau_w_at_xi01
    print(f"Determined fixed tau_w for Scenario 3 (Optimal @ xi={xi_for_fixed_opt}): {fixed_tau_w_scenario3}")
else:
    print(f"Warning: Could not determine optimal tau_w at xi={xi_for_fixed_opt}. Scenario 3 will be skipped.")
    # Scenario 3 loop will check if fixed_tau_w_scenario3 is None

print("-" * 30)
# --- End Parameter Setup ---


# --- Function to optimize ONLY tau_z for FIXED tau_w ---
# This function remains largely the same, but now expects fixed_tau_w_arr of length 2
# and calls the 2-hh inner solver.
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    """
    Optimizes social welfare by choosing only tau_z, given fixed G, xi, and tau_w (len 2).
    """
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            # Calls the imported 2-hh inner solver
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged:
                return 1e10

            utilities = results['utilities'] # Should be length 2
            if np.any(np.isinf(utilities)) or np.any(np.isnan(utilities)):
                 return 1e10 # Penalize invalid states

            agg_polluting = results['z_c'] + results['z_d']
            # Using SWF from outer_solver
            welfare = np.sum(utilities) - 5*xi_val * (agg_polluting**theta)
            return -welfare
        except Exception as e:
            # Catch potential errors during inner solve
            # print(f"Warning: Inner solve failed in fixed_w obj: {e}") # Optional debug
            return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [1.0] # Start guess for tau_z
    try:
        result = minimize(swf_obj_fixed_w,
                          initial_tau_z_guess,
                          args=(G, xi, fixed_tau_w_arr),
                          method='SLSQP', # Changed to SLSQP, often better for bounds
                          bounds=tau_z_bounds,
                          options={'disp': False, 'ftol': 1e-7}) # Added tolerance
        if result.success:
            return result.x[0], -result.fun
        else:
            # print(f"Warning: Fixed_w optimization failed for xi={xi:.2f}. Message: {result.message}") # Optional
            return None, None
    except Exception as e:
        # print(f"Error during fixed_w optimization for xi={xi:.2f}: {e}") # Optional
        return None, None

# Lists to store results for all 3 scenarios
tau_z_optimal_w_results = []
tau_z_fixed_scenario2_results = [] # Renamed list
tau_z_fixed_scenario3_results = [] # Renamed list
valid_xi_optimal_w = []
valid_xi_fixed_scenario2 = [] # Renamed list
valid_xi_fixed_scenario3 = [] # Renamed list

# --- Scenario 1: Optimal tau_w and tau_z ---
print("\nRunning Scenario 1: Variable tau_w...")
print("-" * 30)
for xi_val in xi_values:
    print(f"  Optimizing all for xi = {xi_val:.4f}...") # Show progress
    try:
        # Calls the imported 2-hh outer solver
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            tau_z_optimal_w_results.append(opt_tau_z)
            valid_xi_optimal_w.append(xi_val)
        else:
            print(f"    Optimization failed for xi = {xi_val:.4f}")
            tau_z_optimal_w_results.append(np.nan) # Store NaN on failure
            valid_xi_optimal_w.append(xi_val)
    except Exception as e:
        print(f"      Error during Scenario 1 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_optimal_w_results.append(np.nan)
        valid_xi_optimal_w.append(xi_val)
print("Scenario 1 finished.")

# --- Scenario 2: Fixed tau_w (Set A), Optimal tau_z ---
print("\nRunning Scenario 2: Fixed tau_w (Set A)...")
print(f"(Using fixed tau_w = {fixed_tau_w_scenario2})")
print("-" * 30)
for xi_val in xi_values:
    print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Show progress
    try:
        # Pass the length-2 fixed tau_w array
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_scenario2)
        if opt_tau_z is not None:
            tau_z_fixed_scenario2_results.append(opt_tau_z)
            valid_xi_fixed_scenario2.append(xi_val)
        else:
            print(f"    Optimization failed for xi = {xi_val:.4f}")
            tau_z_fixed_scenario2_results.append(np.nan)
            valid_xi_fixed_scenario2.append(xi_val)
    except Exception as e:
        print(f"      Error during Scenario 2 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_scenario2_results.append(np.nan)
        valid_xi_fixed_scenario2.append(xi_val)
print("Scenario 2 finished.")

# --- Scenario 3: Fixed tau_w (Set B: Optimal at xi=0.1), Optimal tau_z ---
print("\nRunning Scenario 3: Fixed tau_w (Set B: Optimal @ xi=0.1)...")
if fixed_tau_w_scenario3 is not None:
    print(f"(Using fixed tau_w = {fixed_tau_w_scenario3})")
    print("-" * 30)
    for xi_val in xi_values:
        print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Show progress
        try:
            # Pass the length-2 optimal fixed tau_w array determined earlier
            opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_scenario3)
            if opt_tau_z is not None:
                tau_z_fixed_scenario3_results.append(opt_tau_z)
                valid_xi_fixed_scenario3.append(xi_val)
            else:
                print(f"    Optimization failed for xi = {xi_val:.4f}")
                tau_z_fixed_scenario3_results.append(np.nan)
                valid_xi_fixed_scenario3.append(xi_val)
        except Exception as e:
            print(f"      Error during Scenario 3 optimization for xi = {xi_val:.4f}: {e}")
            tau_z_fixed_scenario3_results.append(np.nan)
            valid_xi_fixed_scenario3.append(xi_val)
    print("Scenario 3 finished.")
else:
    print("Scenario 3 skipped (could not determine fixed optimal tau_w).")
    # Fill with NaNs to prevent plotting errors if needed, match length of xi_values
    tau_z_fixed_scenario3_results = [np.nan] * len(xi_values)
    valid_xi_fixed_scenario3 = list(xi_values)


print("-" * 30)
print("Simulations complete.")


# --- Plotting ---

plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
plt.figure(figsize=(6, 4)) # Adjusted size

# Convert lists to numpy arrays for easier NaN handling
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w = np.array(valid_xi_optimal_w)
tau_z_fixed_scenario2_results = np.array(tau_z_fixed_scenario2_results)
valid_xi_fixed_scenario2 = np.array(valid_xi_fixed_scenario2)
tau_z_fixed_scenario3_results = np.array(tau_z_fixed_scenario3_results)
valid_xi_fixed_scenario3 = np.array(valid_xi_fixed_scenario3)

# Filter NaNs for plotting
valid_opt_indices = ~np.isnan(tau_z_optimal_w_results)
valid_fixed_scen2_indices = ~np.isnan(tau_z_fixed_scenario2_results)
valid_fixed_scen3_indices = ~np.isnan(tau_z_fixed_scenario3_results)

# Plot lines with markers
if np.any(valid_opt_indices):
    plt.plot(valid_xi_optimal_w[valid_opt_indices],
             tau_z_optimal_w_results[valid_opt_indices],
             linestyle='-', markersize=4, label='Variable $\\tau_w$')

if np.any(valid_fixed_scen2_indices):
    plt.plot(valid_xi_fixed_scenario2[valid_fixed_scen2_indices],
             tau_z_fixed_scenario2_results[valid_fixed_scen2_indices],
             linestyle='--', markersize=4, label=f'Fixed $\\tau_w$ (Set A: {fixed_tau_w_scenario2})') # Updated label

if fixed_tau_w_scenario3 is not None and np.any(valid_fixed_scen3_indices):
    label_scen3 = f'Fixed $\\tau_w$ (Set B: Opt @ $\\xi$={xi_for_fixed_opt})'
    plt.plot(valid_xi_fixed_scenario3[valid_fixed_scen3_indices],
             tau_z_fixed_scenario3_results[valid_fixed_scen3_indices],
             linestyle=':', markersize=4, label=label_scen3)

# Add labels and title
plt.xlabel(r'Pollution Aversion ($\xi$)', fontsize=12)
plt.ylabel(r'Optimal Environmental Tax ($\tau_z$)', fontsize=12)
plt.title(r'Optimal $\tau_z$ vs. Pollution Aversion ($\xi$)', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add legend
if np.any(valid_opt_indices) or np.any(valid_fixed_scen2_indices) or (fixed_tau_w_scenario3 is not None and np.any(valid_fixed_scen3_indices)):
    plt.legend(loc='best', fontsize=10)

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs_2hh" # New folder name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tau_z_vs_xi_comparison_2hh.pdf") # New filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()