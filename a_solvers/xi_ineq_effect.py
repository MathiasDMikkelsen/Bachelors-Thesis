import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import new_inner_solver as solver  # inner_solver.solve now accepts phi as an argument
import os

# --- Simulation Parameters ---
G_value = 0.0          # Government consumption = 0
theta = 1.0

# Use old calibration for φ always
phi_old = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])

# Define two tax system scenarios:
fixed_tau_w_initial = np.zeros(len(phi_old))                # Initial Tax System: all zero
fixed_tau_w_unequal = np.array([0.015, 0.072, 0.115, 0.156, 0.24])  # Unequal Tax System

# Define the range of ξ values to test
xi_values = np.linspace(0.1, 0.5, 25)
# --- End Simulation Parameters ---

# --- Function to optimize ONLY τ_z for FIXED τ_w ---
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr, phi_arr):
    """
    Optimizes social welfare by choosing only τ_z given fixed τ_w, G, ξ and φ.
    Social welfare = sum(utilities) - 5 * ξ * (agg_polluting**theta)
    """
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr, phi_arr_val):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val, phi_arr_val)
            if not converged:
                print(f"[DEBUG] τ_z = {tau_z:.4f}: Solver did not converge, returning high penalty.")
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            welfare = np.sum(utilities) - 5 * xi_val * (agg_polluting**theta)
            print(f"[DEBUG] τ_z = {tau_z:.4f} | Σ utilities = {np.sum(utilities):.4f} | "
                  f"agg_polluting = {agg_polluting:.4f} | welfare = {welfare:.4f}")
            return -welfare
        except Exception as e:
            print(f"[DEBUG] Exception for τ_z = {tau_z:.4f}: {e}")
            return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [0.5]
    try:
        result = minimize(swf_obj_fixed_w,
                          initial_tau_z_guess,
                          args=(G, xi, fixed_tau_w_arr, phi_arr),
                          method='SLSQP',
                          bounds=tau_z_bounds,
                          options={'disp': False, 'ftol': 1e-7, 'maxiter': 200})
        if result.success:
            print(f"[INFO] ξ = {xi:.4f}: Optimal τ_z found: {result.x[0]:.4f} with welfare = {-result.fun:.4f}")
            return result.x[0], -result.fun
        else:
            print(f"[INFO] ξ = {xi:.4f}: Optimization failed, message: {result.message}")
            return None, None
    except Exception as e:
        print(f"[INFO] Exception during optimization for ξ = {xi:.4f}: {e}")
        return None, None

# --- Lists to store results for each tax system scenario ---
tau_z_initial_results = []
valid_xi_initial = []

tau_z_unequal_results = []
valid_xi_unequal = []

# --- Scenario 1: Initial Tax System (zero τ_w) ---
print("Running Scenario 1: Initial Tax System (uniform zero τ_w)...")
for xi_val in xi_values:
    opt_tau_z, welfare = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_initial, phi_old)
    if opt_tau_z is not None:
        tau_z_initial_results.append(opt_tau_z)
        valid_xi_initial.append(xi_val)
    else:
        tau_z_initial_results.append(np.nan)
        valid_xi_initial.append(xi_val)
print("Scenario 1 finished.\n")

# --- Scenario 2: Unequal Tax System ---
print("Running Scenario 2: Unequal Tax System...")
for xi_val in xi_values:
    opt_tau_z, welfare = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_unequal, phi_old)
    if opt_tau_z is not None:
        tau_z_unequal_results.append(opt_tau_z)
        valid_xi_unequal.append(xi_val)
    else:
        tau_z_unequal_results.append(np.nan)
        valid_xi_unequal.append(xi_val)
print("Scenario 2 finished.\n")
print("Simulations complete.\n")

# --- Plotting ---
tau_z_initial_results = np.array(tau_z_initial_results)
valid_xi_initial = np.array(valid_xi_initial)
tau_z_unequal_results = np.array(tau_z_unequal_results)
valid_xi_unequal = np.array(valid_xi_unequal)

plt.figure(figsize=(5, 3.5))
initial_valid = ~np.isnan(tau_z_initial_results)
unequal_valid = ~np.isnan(tau_z_unequal_results)
if np.any(initial_valid):
    plt.plot(valid_xi_initial[initial_valid],
             tau_z_initial_results[initial_valid],
             linestyle='-', label="Initial Tax System")
if np.any(unequal_valid):
    plt.plot(valid_xi_unequal[unequal_valid],
             tau_z_unequal_results[unequal_valid],
             linestyle='--', label="Unequal Tax System")

plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\tau_z$', fontsize=14)
plt.title(r'Optimal $\tau_z$ vs. $\xi$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tau_z_comparison_taxsystems.pdf")
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")
plt.show()