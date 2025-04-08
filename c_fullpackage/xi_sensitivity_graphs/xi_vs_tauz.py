# plot_tau_z_comparison.py (Fixed theta import)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import outer_solver # Imports the outer solver with maximize_welfare(G, xi)
import inner_solver as solver # Import inner solver directly too
# *** Removed theta from this import line ***
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params
import os # For saving figure

# --- Simulation Parameters ---
G_value = 5.0
# Use the specified fixed tau_w rates
fixed_tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
# *** Added local definition for theta ***
theta = 1.0

# Define the range and number of xi values to test
xi_values = np.linspace(0, 1.0, 25) # Example: 0 to 0.4 in 21 steps
# --- End Simulation Parameters ---

# --- Function to optimize ONLY tau_z for FIXED tau_w ---
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
            # Uses local theta now
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

# Lists to store results
tau_z_optimal_w_results = []
tau_z_fixed_w_results = []
valid_xi_optimal_w = []
valid_xi_fixed_w = []

# --- Scenario 1: Optimal tau_w and tau_z ---
print("Running Scenario 1: Optimal tau_w and tau_z...")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing all for xi = {xi_val:.4f}...") # Reduced print frequency
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

# --- Scenario 2: Fixed tau_w, Optimal tau_z ---
print("\nRunning Scenario 2: Fixed tau_w, Optimal tau_z...")
print(f"(Using fixed tau_w = {fixed_tau_w})")
print("-" * 30)
for xi_val in xi_values:
    # print(f"  Optimizing tau_z for xi = {xi_val:.4f}...") # Reduced print frequency
    try:
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w)
        if opt_tau_z is not None:
            tau_z_fixed_w_results.append(opt_tau_z)
            valid_xi_fixed_w.append(xi_val)
        else:
            tau_z_fixed_w_results.append(np.nan)
            valid_xi_fixed_w.append(xi_val)
    except Exception as e:
        print(f"    Error during Scenario 2 optimization for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_w_results.append(np.nan)
        valid_xi_fixed_w.append(xi_val)
print("Scenario 2 finished.")

print("-" * 30)
print("Simulations complete.")


# --- Plotting (Styled like CV plot example) ---

# Convert lists to numpy arrays
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w = np.array(valid_xi_optimal_w)
tau_z_fixed_w_results = np.array(tau_z_fixed_w_results)
valid_xi_fixed_w = np.array(valid_xi_fixed_w)

# Create the plot with specified size
plt.figure(figsize=(5, 3.5)) # Set figure size from example

# Plot Line 1 (Optimal tau_w) - Filter NaNs
valid_opt_indices = ~np.isnan(tau_z_optimal_w_results)
if np.any(valid_opt_indices):
    plt.plot(valid_xi_optimal_w[valid_opt_indices],
             tau_z_optimal_w_results[valid_opt_indices],
             linestyle='-', label='Optimal $\\tau_w$')

# Plot Line 2 (Fixed tau_w) - Filter NaNs
valid_fixed_indices = ~np.isnan(tau_z_fixed_w_results)
if np.any(valid_fixed_indices):
    plt.plot(valid_xi_fixed_w[valid_fixed_indices],
             tau_z_fixed_w_results[valid_fixed_indices],
             linestyle='--', label='Fixed $\\tau_w$ (US Calib.)')

# Add labels with specified font size
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\tau_z$', fontsize=14)

# Add legend
if np.any(valid_opt_indices) or np.any(valid_fixed_indices):
    plt.legend()

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tau_z_comparison_plot_styled.pdf")
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()