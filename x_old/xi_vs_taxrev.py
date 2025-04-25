# plot_revenue_vs_xi.py (Legend bottom right)

import numpy as np
import matplotlib.pyplot as plt
import os
import outer_solver # Imports the outer solver with maximize_welfare(G, xi)
import inner_solver as solver # Import inner solver directly too
# *** Removed theta from import, defined locally below ***
from inner_solver import n, phi, t as T # Import necessary params from inner_solver

# --- Simulation Parameters ---
G_value = 5.0
# *** Ensure theta is defined locally ***
theta = 1.0
# Set xi_values as requested
xi_values = np.linspace(0.1, 1.0, 25) # Using the value from the previous working version
# --- End Simulation Parameters ---

# Lists to store results
revenue_env_tax_results = []
revenue_inc_tax_results = []
revenue_total_tax_results = []
valid_xi_values = []

print(f"Running optimization and revenue calculation for {len(xi_values)} values of xi...")
print("-" * 30)

# Loop through each xi value
for xi_val in xi_values:
    print(f"  Processing xi = {xi_val:.4f}...")
    rev_z = np.nan
    rev_w = np.nan
    rev_total = np.nan
    try:
        # Step 1: Find optimal taxes
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)

        if opt_tau_w is not None and opt_tau_z is not None:
            # Step 2: Re-solve inner solver
            try:
                 inner_solution, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
                 if inner_converged:
                     # Step 3: Calculate Revenues
                     rev_z = opt_tau_z * (inner_results['z_c'] + inner_results['z_d'])
                     hours_worked_i = T - inner_results['l_agents']
                     hours_worked_i = np.maximum(hours_worked_i, 0)
                     gross_labor_income_i = phi * inner_results['w'] * hours_worked_i
                     rev_w = np.sum(opt_tau_w * gross_labor_income_i)
                     rev_total = rev_z + rev_w
                 else:
                     print(f"      Warning: Inner solver failed for optimal taxes at xi = {xi_val:.4f}")
            except Exception as inner_e:
                 print(f"      Error during inner solve for xi = {xi_val:.4f}: {inner_e}")

            # Append results
            revenue_env_tax_results.append(rev_z)
            revenue_inc_tax_results.append(rev_w)
            revenue_total_tax_results.append(rev_total)
            valid_xi_values.append(xi_val)
        else:
            print(f"    Outer Optimization failed for xi = {xi_val:.4f}")
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
plt.figure(figsize=(7, 5))

# Plot Lines - Filter NaNs
valid_env_indices = ~np.isnan(revenue_env_tax_results)
valid_inc_indices = ~np.isnan(revenue_inc_tax_results)
valid_tot_indices = ~np.isnan(revenue_total_tax_results)

color_total = 'tab:blue'
color_env = 'tab:green'
color_inc = 'tab:red'

if np.any(valid_env_indices):
    plt.plot(valid_xi_values[valid_env_indices],
             revenue_env_tax_results[valid_env_indices],
             linestyle='--', color=color_env, label='Env. tax revenue')

if np.any(valid_inc_indices):
    plt.plot(valid_xi_values[valid_inc_indices],
             revenue_inc_tax_results[valid_inc_indices],
             linestyle=':', color=color_inc, label='Income tax revenue')

if np.any(valid_tot_indices):
    plt.plot(valid_xi_values[valid_tot_indices],
             revenue_total_tax_results[valid_tot_indices],
             linestyle='-', color=color_total, label='Total revenue', linewidth=2)

# Add labels
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Revenue', fontsize=14)

# Add legend *** Changed location ***
if np.any(valid_env_indices) or np.any(valid_inc_indices) or np.any(valid_tot_indices):
    plt.legend(loc='lower right') # <-- Set location to bottom right

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "c_opttax"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "f_xi_taxrev.pdf")
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()