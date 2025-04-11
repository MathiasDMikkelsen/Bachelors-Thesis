# plot_revenue_vs_xi_fixed_pre.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
# Import only inner_solver, outer_solver is not needed here
import inner_solver as solver
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T # Import necessary params

# --- Simulation Parameters ---
G_value = 5.0
# Define theta locally
theta = 1.0

# Define the fixed pre-existing tau_w rates
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

# Define the range and number of xi values to test
xi_values = np.linspace(0.1, 1.0, 25) # Using value from your previous code
# --- End Simulation Parameters ---

# --- Helper Function to optimize ONLY tau_z for FIXED tau_w ---
# (Same as used before, ensures theta is defined/available)
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged: return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities[utilities > -1e5]
            welfare = np.sum(valid_utilities) - 5*xi_val * (agg_polluting**theta) # Uses local theta
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
revenue_env_tax_results = []
revenue_inc_tax_results = []
revenue_total_tax_results = []
valid_xi_values = []

print(f"Running fixed tau_w optimization and revenue calculation for {len(xi_values)} values of xi...")
print(f"(Using fixed tau_w = {fixed_tau_w_preexisting})")
print("-" * 30)

# --- Simulation Loop ---
for xi_val in xi_values:
    # print(f"  Processing xi = {xi_val:.4f}...") # Optional print
    rev_z = np.nan
    rev_w = np.nan
    rev_total = np.nan
    try:
        # Step 1: Find optimal tau_z for the fixed tau_w
        opt_tau_z, max_welfare_val = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)

        if opt_tau_z is not None:
            # Step 2: Re-solve inner solver with fixed tau_w and optimal tau_z
            try:
                 inner_solution, inner_results, inner_converged = solver.solve(fixed_tau_w_preexisting, opt_tau_z, G_value)
                 if inner_converged:
                     # Step 3: Calculate Revenues
                     rev_z = opt_tau_z * (inner_results['z_c'] + inner_results['z_d'])
                     hours_worked_i = T - inner_results['l_agents']
                     hours_worked_i = np.maximum(hours_worked_i, 0)
                     gross_labor_income_i = phi * inner_results['w'] * hours_worked_i
                     # *** Use the FIXED tau_w for income revenue calculation ***
                     rev_w = np.sum(fixed_tau_w_preexisting * gross_labor_income_i)
                     rev_total = rev_z + rev_w
                 # else: print(f"      Warning: Inner solver failed for optimal tau_z at xi = {xi_val:.4f}")
            except Exception as inner_e:
                 print(f"      Error during inner solve for xi = {xi_val:.4f}: {inner_e}")

            # Append results (even if inner solve failed)
            revenue_env_tax_results.append(rev_z)
            revenue_inc_tax_results.append(rev_w)
            revenue_total_tax_results.append(rev_total)
            valid_xi_values.append(xi_val)
        else:
            # print(f"    Outer Optimization (fixed tau_w) failed for xi = {xi_val:.4f}") # Optional print
            revenue_env_tax_results.append(np.nan)
            revenue_inc_tax_results.append(np.nan)
            revenue_total_tax_results.append(np.nan)
            valid_xi_values.append(xi_val)
    except Exception as e:
        print(f"    Error during outer optimization run for xi = {xi_val:.4f}: {e}")
        revenue_env_tax_results.append(np.nan)
        revenue_inc_tax_results.append(np.nan)
        revenue_total_tax_results.append(np.nan)
        valid_xi_values.append(xi_val)

print("-" * 30)
print("Simulations complete.")


# --- Plotting ---

# Convert lists to numpy arrays
revenue_env_tax_results = np.array(revenue_env_tax_results)
revenue_inc_tax_results = np.array(revenue_inc_tax_results)
revenue_total_tax_results = np.array(revenue_total_tax_results)
valid_xi_values = np.array(valid_xi_values)

# Create the plot
plt.figure(figsize=(5, 3.5))

# Plot Lines - Filter NaNs
valid_env_indices = ~np.isnan(revenue_env_tax_results)
valid_inc_indices = ~np.isnan(revenue_inc_tax_results)
valid_tot_indices = ~np.isnan(revenue_total_tax_results)

color_total = 'tab:blue'
color_env = 'tab:green'
color_inc = 'tab:red'
color_g = 'grey'

# Using the same labels as before, maybe add note about fixed tau_w if needed
label_env = 'Env. Tax Rev.'
label_inc = 'Inc. Tax Rev.'
label_total = 'Total Tax Rev.'
label_g = f'G={G_value}'

if np.any(valid_env_indices):
    plt.plot(valid_xi_values[valid_env_indices],
             revenue_env_tax_results[valid_env_indices],
             linestyle='--', color=color_env, label=label_env)
if np.any(valid_inc_indices):
    plt.plot(valid_xi_values[valid_inc_indices],
             revenue_inc_tax_results[valid_inc_indices],
             linestyle=':', color=color_inc, label=label_inc)
if np.any(valid_tot_indices):
    plt.plot(valid_xi_values[valid_tot_indices],
             revenue_total_tax_results[valid_tot_indices],
             linestyle='-', color=color_total, label=label_total, linewidth=2)

# Add Horizontal Line for Government Spending
plt.axhline(y=G_value, color=color_g, linestyle='--', linewidth=1.5, label=label_g)

# Add labels
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Tax Revenue', fontsize=14)

# Add legend
if np.any(valid_env_indices) or np.any(valid_inc_indices) or np.any(valid_tot_indices):
    plt.legend(loc='upper right') # Keep location as requested before

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs" # Subdirectory relative to script location
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = "revenue_fixed_preexisting_plot.pdf" # New filename
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()