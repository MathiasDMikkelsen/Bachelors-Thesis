# c_fullpackage/plot_xi_sensitivity.py
# (Uses path logic from working example - REQUIRES __init__.py in a_solvers)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys # Import sys
import math

# --- Path Setup (Identical to your working example) ---
# Get the directory of the current script (c_fullpackage)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which contains a_solvers and c_fullpackage)
project_root = os.path.dirname(script_dir)
# Add the parent directory to the Python path to find a_solvers
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Updated Imports (Using logic from working example) ---
from a_solvers import outer_solver # Import from a_solvers folder
from a_solvers import inner_solver as solver # Import from a_solvers folder
from a_solvers.inner_solver import n # Import n from inner_solver within a_solvers
# --- End Updated Imports ---

# --- Simulation Parameters ---
G_value = 5.0
# Define the range and number of xi values to test (using value from your last code)
xi_values = np.linspace(0.001, 1.0, 25)
# Define theta locally if needed by outer_solver's objective function logic
# theta = 1.0
# --- End Simulation Parameters ---

# Lists to store results
optimal_tau_w_results = []
lumpsum_results = []
valid_xi_values = []

print(f"Running optimization for {len(xi_values)} values of xi...")
print("-" * 30)

# Loop through each xi value
for xi_val in xi_values:
    print(f"  Running for xi = {xi_val:.4f}...")
    l_val = np.nan
    try:
        # Step 1: Find optimal taxes using outer solver
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)

        if opt_tau_w is not None and opt_tau_z is not None:
            # print(f"    Outer Success: tau_z = {opt_tau_z:.4f}, SWF = {max_welfare_val:.4f}") # Optional

            # Step 2: Re-solve inner solver with optimal taxes to get 'l'
            try:
                 inner_solution, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
                 if inner_converged:
                     l_val = inner_results['l']
                     # print(f"      Inner Success: l = {l_val:.4f}") # Optional
                 else:
                     print(f"      Warning: Inner solver failed (optimal taxes) for xi = {xi_val:.4f}")
            except Exception as inner_e:
                 print(f"      Error during inner solve for xi = {xi_val:.4f}: {inner_e}")

            # Step 3: Store results
            optimal_tau_w_results.append(opt_tau_w)
            lumpsum_results.append(l_val)
            valid_xi_values.append(xi_val)
        else:
            print(f"    Outer Optimization failed for xi = {xi_val:.4f}")
            optimal_tau_w_results.append([np.nan]*n)
            lumpsum_results.append(np.nan)
            valid_xi_values.append(xi_val)

    except Exception as e:
        print(f"    Error during outer optimization run for xi = {xi_val:.4f}: {e}")
        optimal_tau_w_results.append([np.nan]*n)
        lumpsum_results.append(np.nan)
        valid_xi_values.append(xi_val)

print("-" * 30)
print("Optimization runs complete.")

# --- Plotting ---

# Filter results where lumpsum is valid
lumpsum_results = np.array(lumpsum_results)
valid_indices = ~np.isnan(lumpsum_results)

if np.sum(valid_indices) > 0:
    optimal_tau_w_results = np.array(optimal_tau_w_results)[valid_indices]
    lumpsum_results = lumpsum_results[valid_indices]
    valid_xi_values = np.array(valid_xi_values)[valid_indices]

    # Calculate max range for tau_w
    max_range_tau_w = 0
    if optimal_tau_w_results.size > 0:
        valid_tau_w_for_range = optimal_tau_w_results[~np.isnan(optimal_tau_w_results).any(axis=1)]
        if valid_tau_w_for_range.size > 0:
             ranges = [np.ptp(valid_tau_w_for_range[:, i]) for i in range(n)]
             max_range_tau_w = np.max(ranges)
             margin_range = max_range_tau_w * 0.10
             if margin_range < 1e-6: margin_range = 0.2
             max_range_tau_w += margin_range
        else: max_range_tau_w = 0

    # Create figure and axes
    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.5, 4.4))

    # Plot tau_w
    for i in range(n):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col]
        current_tau_w_data = optimal_tau_w_results[:, i]
        if not np.isnan(current_tau_w_data).all():
            ax.plot(valid_xi_values, current_tau_w_data, linestyle='-')
            ax.set_ylabel(rf'$\tau_w^{{{i+1}}}$', fontsize=12, rotation=90)
            if max_range_tau_w > 1e-9:
                data_min, data_max = np.nanmin(current_tau_w_data), np.nanmax(current_tau_w_data)
                midpoint_i = (data_max + data_min) / 2
                ax.set_ylim(midpoint_i - max_range_tau_w / 2, midpoint_i + max_range_tau_w / 2)
            if row == n_rows - 1: ax.set_xlabel(r'$\xi$', fontsize=12)
        else: ax.set_visible(False)

    # Plot Lumpsum 'l'
    if n < n_rows * n_cols:
        row, col = n // n_cols, n % n_cols
        ax = axs[row, col]
        ax.plot(valid_xi_values, lumpsum_results, linestyle='-', color='tab:green')
        ax.set_ylabel(r'$l$', fontsize=12, rotation=90)
        if row == n_rows - 1: ax.set_xlabel(r'$\xi$', fontsize=12)

    # Hide unused subplots
    for i in range(n + 1, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if row < axs.shape[0] and col < axs.shape[1]: fig.delaxes(axs[row, col])

    # Output Path within c_fullpackage
    output_filename = "xi_sensitivity_grid_plots_lumpsum.pdf"
    output_path = os.path.join(script_dir, output_filename) # Use script_dir

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlots saved to {output_path}")
    plt.show()
else:
    print("\nNo successful optimization runs with valid lumpsum values were completed.")