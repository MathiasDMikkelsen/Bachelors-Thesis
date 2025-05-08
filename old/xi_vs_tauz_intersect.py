# plot_tau_z_comparison.py (Finds intersection tau_z of fixed schedules)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Assuming running in same folder as solvers:
import b_outer_solver
import a_inner_solver as solver
from a_inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params
import os

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0 # Define theta locally

# Define the fixed tau_w sets
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])

# Define the range and number of xi values to test
xi_values = np.linspace(0.001, 1.0, 50) # Keep slightly higher resolution
# --- End Simulation Parameters ---

# --- Helper Function to optimize ONLY tau_z for FIXED tau_w ---
# (Same function as before)
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
    tau_z_bounds = [(1e-6, 100.0)]; initial_tau_z_guess = [0.5]
    try:
        result = minimize(swf_obj_fixed_w, initial_tau_z_guess, args=(G, xi, fixed_tau_w_arr),
                          method='SLSQP', bounds=tau_z_bounds, options={'disp': False, 'ftol': 1e-7})
        if result.success: return result.x[0], -result.fun
        else: return None, None
    except Exception as e: return None, None
# --- End Helper Function ---

# Lists to store results
tau_z_optimal_w_results = []
tau_z_fixed_preexisting_results = []
tau_z_fixed_optimal_xi01_results = []
valid_xi_processed = []

# --- Simulation Loops ---
print("Running simulations...")
for xi_val in xi_values:
    # print(f"  Processing xi = {xi_val:.4f}...") # Optional print
    valid_xi_processed.append(xi_val)
    # Run Scenario 1
    opt_tau_z_scen1 = np.nan
    try:
        _, opt_tau_z, _ = b_outer_solver.maximize_welfare(G_value, xi_val)
        if opt_tau_z is not None: opt_tau_z_scen1 = opt_tau_z
    except Exception as e: print(f"    Error Scen 1, xi={xi_val:.4f}: {e}")
    tau_z_optimal_w_results.append(opt_tau_z_scen1)
    # Run Scenario 2
    opt_tau_z_scen2 = np.nan
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
        if opt_tau_z is not None: opt_tau_z_scen2 = opt_tau_z
    except Exception as e: print(f"    Error Scen 2, xi={xi_val:.4f}: {e}")
    tau_z_fixed_preexisting_results.append(opt_tau_z_scen2)
    # Run Scenario 3
    opt_tau_z_scen3 = np.nan
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
        if opt_tau_z is not None: opt_tau_z_scen3 = opt_tau_z
    except Exception as e: print(f"    Error Scen 3, xi={xi_val:.4f}: {e}")
    tau_z_fixed_optimal_xi01_results.append(opt_tau_z_scen3)
print("Simulations complete.")

# --- Find Intersection ---
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results)
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)
valid_xi_processed = np.array(valid_xi_processed)

# Filter for valid comparison points
valid_comparison_indices = ~np.isnan(tau_z_fixed_preexisting_results) & ~np.isnan(tau_z_fixed_optimal_xi01_results)
xi_compare = valid_xi_processed[valid_comparison_indices]
tau_z_pre_compare = tau_z_fixed_preexisting_results[valid_comparison_indices]
tau_z_opt01_compare = tau_z_fixed_optimal_xi01_results[valid_comparison_indices]

intersection_xi = np.nan
intersection_tau_z = np.nan # <-- Variable to store intersection tau_z

if len(xi_compare) > 1:
    diff = tau_z_pre_compare - tau_z_opt01_compare
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        k = sign_changes[0]
        xi_k, xi_k1 = xi_compare[k], xi_compare[k+1]
        d_k, d_k1 = diff[k], diff[k+1]

        if abs(d_k1 - d_k) > 1e-9:
             # Find xi at intersection using interpolation
             intersection_xi = xi_k - d_k * (xi_k1 - xi_k) / (d_k1 - d_k)

             # --- Calculate tau_z at intersection_xi ---
             # We can use either fixed schedule; let's use pre-existing
             print(f"\nCalculating tau_z at intersection xi = {intersection_xi:.4f} using fixed pre-existing tau_w...")
             opt_tau_z_at_intersect, _ = maximize_welfare_fixed_w(G_value, intersection_xi, fixed_tau_w_preexisting)

             if opt_tau_z_at_intersect is not None:
                 intersection_tau_z = opt_tau_z_at_intersect
                 print(f"--> Estimated intersection tau_z = {intersection_tau_z:.4f}")
             else:
                 print(f"--> Failed to calculate tau_z at intersection xi.")
             # --- End Calculate tau_z ---

        print("-" * 30)
        print(f"Intersection of Fixed Pre-existing and Fixed Optimal tau_z schedules")
        print(f"Occurs between xi = {xi_k:.4f} and xi = {xi_k1:.4f}")
        if not np.isnan(intersection_xi): print(f"Estimated intersection xi = {intersection_xi:.4f}")
        if not np.isnan(intersection_tau_z): print(f"Estimated intersection tau_z = {intersection_tau_z:.4f}")
        print("-" * 30)
    else: print("No sign change detected in the difference between the two fixed tau_z schedules.")
else: print("Not enough valid comparison points to find intersection.")


# --- Plotting ---
valid_xi_optimal_w = valid_xi_processed

plt.figure(figsize=(5, 3.5))

# Plot Lines - Filter NaNs
valid_opt_indices = ~np.isnan(tau_z_optimal_w_results)
valid_fixed_pre_indices = ~np.isnan(tau_z_fixed_preexisting_results)
valid_fixed_opt01_indices = ~np.isnan(tau_z_fixed_optimal_xi01_results)

if np.any(valid_opt_indices): plt.plot(valid_xi_optimal_w[valid_opt_indices], tau_z_optimal_w_results[valid_opt_indices], linestyle='-', label='Variable $\\tau_w$')
if np.any(valid_fixed_pre_indices): plt.plot(valid_xi_optimal_w[valid_fixed_pre_indices], tau_z_fixed_preexisting_results[valid_fixed_pre_indices], linestyle='--', label='Fixed $\\tau_w$ (Pre-existing)')
if np.any(valid_fixed_opt01_indices): plt.plot(valid_xi_optimal_w[valid_fixed_opt01_indices], tau_z_fixed_optimal_xi01_results[valid_fixed_opt01_indices], linestyle=':', label='Fixed $\\tau_w$ (Optimal at $\\xi=0.1$)')

# Add vertical line at intersection xi and horizontal line at intersection tau_z
if not np.isnan(intersection_xi) and not np.isnan(intersection_tau_z):
    plt.axvline(x=intersection_xi, color='grey', linestyle='-.', linewidth=1, label=f'Intersect ($\\xi\\approx${intersection_xi:.2f})')
    plt.axhline(y=intersection_tau_z, color='grey', linestyle='-.', linewidth=1, label=f'Intersect ($\\tau_z\\approx${intersection_tau_z:.2f})')
    # Alternatively, add text annotation:
    # plt.text(intersection_xi*1.01, intersection_tau_z*1.01, f'($\\xi\\approx${intersection_xi:.2f}, $\\tau_z\\approx${intersection_tau_z:.2f})', color='grey')


# Add labels
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\tau_z$', fontsize=14)

# Add legend
if np.any(valid_opt_indices) or np.any(valid_fixed_pre_indices) or np.any(valid_fixed_opt01_indices):
    plt.legend(loc='best', fontsize='small')

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs"; os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "tau_z_comparison_3_scenarios_intersect_value.pdf") # New filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()