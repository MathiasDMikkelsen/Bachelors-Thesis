# plot_revenue_vs_xi.py (with total revenue and new colors)

import numpy as np
import matplotlib.pyplot as plt
import os
import outer_solver # Imports the outer solver with maximize_welfare(G, xi)
import inner_solver as solver # Import inner solver directly too
from inner_solver import n, phi, t as T, theta # Import necessary params from inner_solver

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0
# Set xi_values as requested previously
xi_values = np.linspace(0, 1.0, 25)
# --- End Simulation Parameters ---

# Lists to store results
revenue_env_tax_results = []
revenue_inc_tax_results = []
revenue_total_tax_results = [] # <-- Added list for total revenue
valid_xi_values = []

print(f"Running optimization and revenue calculation for {len(xi_values)} values of xi...")
print("-" * 30)

# Loop through each xi value
for xi_val in xi_values:
    print(f"  Processing xi = {xi_val:.4f}...")
    rev_z = np.nan # Default values
    rev_w = np.nan
    rev_total = np.nan
    try:
        # --- Step 1: Find optimal taxes using outer solver ---
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)

        if opt_tau_w is not None and opt_tau_z is not None:
            # print(f"    Outer Success: tau_z = {opt_tau_z:.4f}") # Optional print

            # --- Step 2: Re-solve inner solver with optimal taxes to get equilibrium values ---
            try:
                 inner_solution, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value)

                 if inner_converged:
                     # --- Step 3: Calculate Revenues ---
                     rev_z = opt_tau_z * (inner_results['z_c'] + inner_results['z_d'])
                     hours_worked_i = T - inner_results['l_agents']
                     hours_worked_i = np.maximum(hours_worked_i, 0)
                     gross_labor_income_i = phi * inner_results['w'] * hours_worked_i
                     rev_w = np.sum(opt_tau_w * gross_labor_income_i)
                     rev_total = rev_z + rev_w # <-- Calculate total revenue
                     # print(f"      Inner Success: Rev_Z={rev_z:.4f}, Rev_W={rev_w:.4f}, Rev_Total={rev_total:.4f}") # Optional

                 else:
                     print(f"      Warning: Inner solver failed for optimal taxes at xi = {xi_val:.4f}")

            except Exception as inner_e:
                 print(f"      Error during inner solve for xi = {xi_val:.4f}: {inner_e}")

            # Append results for this xi value (even if inner failed, append nans)
            revenue_env_tax_results.append(rev_z)
            revenue_inc_tax_results.append(rev_w)
            revenue_total_tax_results.append(rev_total) # <-- Append total revenue
            valid_xi_values.append(xi_val)

        else:
            print(f"    Outer Optimization failed for xi = {xi_val:.4f}")
            revenue_env_tax_results.append(np.nan)
            revenue_inc_tax_results.append(np.nan)
            revenue_total_tax_results.append(np.nan) # <-- Append nan for total
            valid_xi_values.append(xi_val)

    except Exception as e:
        print(f"    Error during outer optimization run for xi = {xi_val:.4f}: {e}")
        revenue_env_tax_results.append(np.nan)
        revenue_inc_tax_results.append(np.nan)
        revenue_total_tax_results.append(np.nan) # <-- Append nan for total
        valid_xi_values.append(xi_val)

print("-" * 30)
print("Simulations complete.")


# --- Plotting (Styled like CV plot example, 3 lines, new colors) ---

# Convert lists to numpy arrays
revenue_env_tax_results = np.array(revenue_env_tax_results)
revenue_inc_tax_results = np.array(revenue_inc_tax_results)
revenue_total_tax_results = np.array(revenue_total_tax_results) # <-- Convert total
valid_xi_values = np.array(valid_xi_values)

# Create the plot with specified size
plt.figure(figsize=(5, 3.5)) # Set figure size from example

# Plot Lines - Filter NaNs for each line individually
valid_env_indices = ~np.isnan(revenue_env_tax_results)
valid_inc_indices = ~np.isnan(revenue_inc_tax_results)
valid_tot_indices = ~np.isnan(revenue_total_tax_results)

# Determine common valid xi values if plotting points where *all* data is valid
# common_valid_indices = valid_env_indices & valid_inc_indices & valid_tot_indices
# For now, plot each line where its own data is valid

# Color choices inspired by image (using common names)
color_total = 'tab:blue'  # Blue for total
color_env = 'tab:green' # Green for environmental
color_inc = 'tab:red'   # Red for income

if np.any(valid_env_indices):
    plt.plot(valid_xi_values[valid_env_indices],
             revenue_env_tax_results[valid_env_indices],
             linestyle='--', color=color_env, label='Env. Tax Revenue') # Dashed

if np.any(valid_inc_indices):
    plt.plot(valid_xi_values[valid_inc_indices],
             revenue_inc_tax_results[valid_inc_indices],
             linestyle=':', color=color_inc, label='Income Tax Revenue') # Dotted

if np.any(valid_tot_indices):
    plt.plot(valid_xi_values[valid_tot_indices],
             revenue_total_tax_results[valid_tot_indices],
             linestyle='-', color=color_total, label='Total Tax Revenue', linewidth=2) # Solid, thicker

# Add labels with specified font size
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)

# Add legend
# Check if at least one line has valid data to plot
if np.any(valid_env_indices) or np.any(valid_inc_indices) or np.any(valid_tot_indices):
    plt.legend()

# Apply tight layout
plt.tight_layout()

# Save the figure
output_dir = "xi_sensitivity_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "revenue_comparison_plot_total.pdf") # New filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()