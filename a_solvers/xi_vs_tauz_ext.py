# plot_tau_z_comparison.py (Adapted for Heterogeneous Pollution Solvers)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os # For saving figure
import warnings # To potentially ignore specific warnings during optimization

# --- MODIFIED: Import correct baseline solvers and parameters ---
try:
    # Import the baseline outer solver (n=5, IC constraints, HeteroPoll)
    import outer_solver as outer_solver
    # Import the baseline inner solver (n=5)
    import inner_solver as solver
    # Import necessary params from the baseline inner solver
    from inner_solver import n, alpha, beta, gamma, d0, phi, t as T, p_c # Ensure all needed params are imported
except ImportError as e:
    print(f"Error: Could not import '{e.name}'.")
    print("Ensure 'inner_solver.py' and 'outer_solver.py' (with HeteroPoll updates) exist and are accessible.")
    exit()
except AttributeError as e:
     print(f"Fatal Error: A required parameter might be missing from 'inner_solver.py': {e}")
     exit()

# --- Assert n=5 ---
assert n == 5, "This script requires the imported baseline solvers to have n=5 households."

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0 # Pollution exponent in SWF (ensure this is consistent with outer_solver)

# Define the range and number of xi values to test
xi_values = np.linspace(0.1, 1.0, 10) # 10 points from 0.1 to 1.0

# Define heterogeneous pollution sensitivity parameters (kappa)
# MUST match the kappa vector defined inside outer_solver.maximize_welfare
kappa = np.array([1.5, 1.25, 1.0, 0.75, 0.5]) # Example for n=5
assert len(kappa) == n, "Length of kappa vector must match number of households (n=5)"
sum_kappa = np.sum(kappa)

# --- Determine Fixed tau_w arrays for Scenarios 2 & 3 ---

# Scenario 2: Pre-existing tau_w
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
print(f"Using fixed tau_w for Scenario 2 (Pre-existing): {fixed_tau_w_preexisting}")

# Scenario 3: Calculate optimal tau_w at xi=0.1 using the updated outer_solver
print("\nFinding optimal tau_w at xi=0.1 for Scenario 3...")
xi_for_fixed_opt = 0.1
# Call the imported (updated) outer_solver
opt_tau_w_at_xi01, _, _ = outer_solver.maximize_welfare(G_value, xi_for_fixed_opt)

fixed_tau_w_optimal_xi01 = None
if opt_tau_w_at_xi01 is not None:
    fixed_tau_w_optimal_xi01 = opt_tau_w_at_xi01
    print(f"Determined fixed tau_w for Scenario 3 (Optimal @ xi={xi_for_fixed_opt}): {fixed_tau_w_optimal_xi01}")
else:
    print(f"Warning: Could not determine optimal tau_w at xi={xi_for_fixed_opt}. Scenario 3 will be skipped.")
    # Scenario 3 loop will check if fixed_tau_w_optimal_xi01 is None

print("-" * 30)
# --- End Parameter Setup ---


# --- Function to optimize ONLY tau_z for FIXED tau_w ---
# MODIFIED: Updated SWF calculation to use kappa
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    """
    Optimizes social welfare by choosing only tau_z, given fixed G, xi, and tau_w.
    Uses heterogeneous pollution effects (kappa).
    """
    # kappa and sum_kappa defined globally in the script scope now
    assert len(fixed_tau_w_arr) == n, "fixed_tau_w_arr must have length n"

    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        """Objective function for fixed tau_w optimization."""
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar

        if tau_z <= 0: return 1e12

        try:
            # Call the imported baseline inner solver
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)

            if not converged or results is None: return 1e10

            utilities = results.get('utilities', None) # log(u_tilde_i)
            z_c = results.get('z_c', None)
            z_d = results.get('z_d', None)

            if utilities is None or z_c is None or z_d is None: return 1e11
            if np.any(np.isinf(utilities)) or np.any(np.isnan(utilities)): return 1e9 # Penalize invalid utilities

            agg_polluting = z_c + z_d
            if agg_polluting < 0: return 1e8

            # --- MODIFIED SWF CALCULATION (using kappa) ---
            blue_welfare_sum = np.sum(utilities)
            # Use sum_kappa defined in the outer script scope
            green_disutility_total = sum_kappa * xi_val * (agg_polluting ** theta)
            welfare = blue_welfare_sum - green_disutility_total
            # --- END MODIFIED SWF ---

            return -welfare # Minimize negative welfare

        except Exception as e:
            # print(f"Warning: Inner solve failed in fixed_w obj: {e}") # Optional debug
            return 1e10 # Penalize any exception

    # --- Optimization setup (bounds, guess, call) ---
    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [1.0] # Adjusted guess slightly
    try:
        # Suppress RuntimeWarnings temporarily if necessary (e.g., overflow)
        # warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = minimize(
            swf_obj_fixed_w,
            initial_tau_z_guess,
            args=(G, xi, fixed_tau_w_arr),
            method='SLSQP', # SLSQP is generally good with bounds
            bounds=tau_z_bounds,
            options={'disp': False, 'ftol': 1e-7} # Use solver defaults or fine-tune
        )
        # warnings.filterwarnings('default', category=RuntimeWarning) # Restore warnings

        if result.success:
            return result.x[0], -result.fun # Return optimal tau_z and max welfare
        else:
            # print(f"Warning: Fixed_w optimization failed for xi={xi:.4f}. Message: {result.message}") # Optional
            return None, None
    except Exception as e:
        # print(f"Error during fixed_w optimization for xi={xi:.4f}: {e}") # Optional
        return None, None

# Lists to store results for all 3 scenarios
tau_z_optimal_w_results = []
tau_z_fixed_preexisting_results = []
tau_z_fixed_optimal_xi01_results = []
valid_xi_optimal_w = []
valid_xi_fixed_preexisting = []
valid_xi_fixed_optimal_xi01 = []

# --- Run Scenario Simulations ---

# Scenario 1: Optimal tau_w and tau_z (using updated outer_solver with HeteroPoll+IC)
print("\nRunning Scenario 1: Variable tau_w (HeteroPoll + IC Constraints)...")
print("-" * 30)
for xi_val in xi_values:
    print(f"  Optimizing all for xi = {xi_val:.4f}...") # Show progress
    try:
        # Call the imported (updated) outer_solver
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            tau_z_optimal_w_results.append(opt_tau_z)
            valid_xi_optimal_w.append(xi_val)
        else:
            # Optimization failed
            print(f"    -> Optimization failed for xi = {xi_val:.4f} (Scenario 1)")
            tau_z_optimal_w_results.append(np.nan) # Store NaN on failure
            valid_xi_optimal_w.append(xi_val)
    except Exception as e:
        print(f"      Error during Scenario 1 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_optimal_w_results.append(np.nan)
        valid_xi_optimal_w.append(xi_val)
print("Scenario 1 finished.")

# Scenario 2: Fixed tau_w (Pre-existing), Optimal tau_z (using updated fixed_w func)
print("\nRunning Scenario 2: Fixed tau_w (Pre-existing, HeteroPoll)...")
print(f"(Using fixed tau_w = {fixed_tau_w_preexisting})")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Optional progress
    try:
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
        if opt_tau_z is not None:
            tau_z_fixed_preexisting_results.append(opt_tau_z)
            valid_xi_fixed_preexisting.append(xi_val)
        else:
            # print(f"    -> Optimization failed for xi = {xi_val:.4f} (Scenario 2)") # Optional
            tau_z_fixed_preexisting_results.append(np.nan)
            valid_xi_fixed_preexisting.append(xi_val)
    except Exception as e:
        print(f"      Error during Scenario 2 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_preexisting_results.append(np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
print("Scenario 2 finished.")

# Scenario 3: Fixed tau_w (Optimal at xi=0.1), Optimal tau_z (using updated fixed_w func)
print("\nRunning Scenario 3: Fixed tau_w (Optimal at xi=0.1, HeteroPoll)...")
if fixed_tau_w_optimal_xi01 is not None:
    print(f"(Using fixed tau_w = {fixed_tau_w_optimal_xi01})")
    print("-" * 30)
    for xi_val in xi_values:
        # print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Optional progress
        try:
            opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
            if opt_tau_z is not None:
                tau_z_fixed_optimal_xi01_results.append(opt_tau_z)
                valid_xi_fixed_optimal_xi01.append(xi_val)
            else:
                # print(f"    -> Optimization failed for xi = {xi_val:.4f} (Scenario 3)") # Optional
                tau_z_fixed_optimal_xi01_results.append(np.nan)
                valid_xi_fixed_optimal_xi01.append(xi_val)
        except Exception as e:
            print(f"      Error during Scenario 3 optimization for xi = {xi_val:.4f}: {e}")
            tau_z_fixed_optimal_xi01_results.append(np.nan)
            valid_xi_fixed_optimal_xi01.append(xi_val)
    print("Scenario 3 finished.")
else:
    print("Scenario 3 skipped (could not determine fixed optimal tau_w).")
    # Fill with NaNs if skipped
    tau_z_fixed_optimal_xi01_results = [np.nan] * len(xi_values)
    valid_xi_fixed_optimal_xi01 = list(xi_values)

print("-" * 30)
print("Simulations complete.")


# --- Plotting (Handles NaNs) ---

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6)) # Adjusted size slightly

# Convert lists to numpy arrays for easier NaN handling
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w = np.array(valid_xi_optimal_w)
tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results)
valid_xi_fixed_preexisting = np.array(valid_xi_fixed_preexisting)
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)
valid_xi_fixed_optimal_xi01 = np.array(valid_xi_fixed_optimal_xi01)

# Filter NaNs for plotting cleanly
valid_opt_indices = ~np.isnan(tau_z_optimal_w_results)
valid_fixed_pre_indices = ~np.isnan(tau_z_fixed_preexisting_results)
valid_fixed_opt01_indices = ~np.isnan(tau_z_fixed_optimal_xi01_results)

# Plot lines only where data is valid
if np.any(valid_opt_indices):
    plt.plot(valid_xi_optimal_w[valid_opt_indices],
             tau_z_optimal_w_results[valid_opt_indices],
             linestyle='-', marker='o', markersize=4, label='Variable $\\tau_w$ (IC + HeteroPoll)') # Updated label

if np.any(valid_fixed_pre_indices):
    plt.plot(valid_xi_fixed_preexisting[valid_fixed_pre_indices],
             tau_z_fixed_preexisting_results[valid_fixed_pre_indices],
             linestyle='--', marker='s', markersize=4, label='Fixed $\\tau_w$ (Pre-existing, HeteroPoll)') # Updated label

if fixed_tau_w_optimal_xi01 is not None and np.any(valid_fixed_opt01_indices):
    plt.plot(valid_xi_fixed_optimal_xi01[valid_fixed_opt01_indices],
             tau_z_fixed_optimal_xi01_results[valid_fixed_opt01_indices],
             linestyle=':', marker='^', markersize=4, label=f'Fixed $\\tau_w$ (Opt @ $\\xi$=0.1, HeteroPoll)') # Updated label

# Add labels and title
plt.xlabel(r'Pollution Aversion ($\xi$)', fontsize=12)
plt.ylabel(r'Optimal Environmental Tax ($\tau_z$)', fontsize=12)
plt.title(r'Optimal $\tau_z$ vs. Pollution Aversion ($\xi$) - Heterogeneous Effects', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True) # Ensure grid is on

# Add legend only if there's something to label
if np.any(valid_opt_indices) or np.any(valid_fixed_pre_indices) or (fixed_tau_w_optimal_xi01 is not None and np.any(valid_fixed_opt01_indices)):
    plt.legend(loc='best', fontsize=10)

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs_hetero" # Changed folder name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tau_z_comparison_hetero_poll.pdf") # Changed filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()