# plot_xi_sensitivity_g_comp.py
# (Compares G=5 vs G=0 for tau_w and l vs xi in 2x3 grid)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys # Keep sys for potential path issues if needed later
import math

# --- Path Setup (Assuming script is in same folder as solvers for now) ---
# If needed, uncomment and adjust:
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Imports (Assuming script is in same folder as solvers) ---
import b_outer_solver
import a_inner_solver as solver
from a_inner_solver import n, phi, t as T # Import n, phi, T from inner_solver
# --- End Imports ---

# --- Simulation Parameters ---
G_value_base = 5.0
G_value_zero = 0.0
# Define theta locally if not imported (needed for outer_solver's objective)
theta = 1.0
# Define the range and number of xi values to test
xi_values = np.linspace(0.04, 1.0, 25) # From your previous version
# --- End Simulation Parameters ---

# Lists to store results for BOTH G values
optimal_tau_w_results_g_base = []
lumpsum_results_g_base = []
optimal_tau_w_results_g0 = [] # <-- New lists for G=0
lumpsum_results_g0 = [] # <-- New lists for G=0
valid_xi_values = [] # Store xi if *either* simulation succeeds

print(f"Running optimization for G={G_value_base} and G={G_value_zero} for {len(xi_values)} values of xi...")
print("-" * 30)

# Loop through each xi value
for xi_val in xi_values:
    print(f"  Running for xi = {xi_val:.4f}...")
    # Results for G=5 (Base)
    opt_tau_w_g_base = [np.nan] * n
    l_val_g_base = np.nan
    # Results for G=0
    opt_tau_w_g0 = [np.nan] * n
    l_val_g0 = np.nan
    outer_success_g_base = False
    outer_success_g0 = False

    # --- Run for G = 5.0 ---
    try:
        opt_tau_w, opt_tau_z, _ = b_outer_solver.maximize_welfare(G_value_base, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            outer_success_g_base = True
            opt_tau_w_g_base = opt_tau_w # Store temp
            try:
                 _, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value_base)
                 if inner_converged: l_val_g_base = inner_results['l']
            except Exception as inner_e: print(f"      Inner solve error (G={G_value_base}): {inner_e}")
    except Exception as e: print(f"      Outer solve error (G={G_value_base}): {e}")

    # --- Run for G = 0.0 ---
    try:
        opt_tau_w, opt_tau_z, _ = b_outer_solver.maximize_welfare(G_value_zero, xi_val)
        if opt_tau_w is not None and opt_tau_z is not None:
            outer_success_g0 = True
            opt_tau_w_g0 = opt_tau_w # Store temp
            try:
                 _, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value_zero)
                 if inner_converged: l_val_g0 = inner_results['l']
            except Exception as inner_e: print(f"      Inner solve error (G={G_value_zero}): {inner_e}")
    except Exception as e: print(f"      Outer solve error (G={G_value_zero}): {e}")

    # Store results for this xi if at least one scenario worked
    if outer_success_g_base or outer_success_g0:
        valid_xi_values.append(xi_val)
        optimal_tau_w_results_g_base.append(opt_tau_w_g_base)
        lumpsum_results_g_base.append(l_val_g_base)
        optimal_tau_w_results_g0.append(opt_tau_w_g0)
        lumpsum_results_g0.append(l_val_g0)
        print(f"    Stored: l(G=5)={l_val_g_base:.4f}, l(G=0)={l_val_g0:.4f}")
    else:
        print(f"    Both scenarios failed for xi = {xi_val:.4f}")


print("-" * 30)
print("Optimization runs complete.")

# --- Plotting ---

# Convert results to numpy arrays
optimal_tau_w_results_g_base = np.array(optimal_tau_w_results_g_base)
lumpsum_results_g_base = np.array(lumpsum_results_g_base)
optimal_tau_w_results_g0 = np.array(optimal_tau_w_results_g0)
lumpsum_results_g0 = np.array(lumpsum_results_g0)
valid_xi_values = np.array(valid_xi_values)

if len(valid_xi_values) > 0:

    # --- Calculate max range across BOTH scenarios for tau_w plots ---
    max_range_tau_w = 0
    # Combine valid results from both scenarios
    combined_tau_w = np.concatenate((optimal_tau_w_results_g_base[~np.isnan(optimal_tau_w_results_g_base).any(axis=1)],
                                     optimal_tau_w_results_g0[~np.isnan(optimal_tau_w_results_g0).any(axis=1)]), axis=0)

    if combined_tau_w.size > 0:
        ranges = [np.ptp(combined_tau_w[:, i]) for i in range(n)]
        max_range_tau_w = np.max(ranges)
        margin_range = max_range_tau_w * 0.10
        if margin_range < 1e-6: margin_range = 0.2
        max_range_tau_w += margin_range
    # --- End Calculate max range ---

    # Create figure and axes array for a 2x3 grid
    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.5, 4.4)) # Adjusted height from before

    # Plot 1 to n: Plot each tau_w for both G values
    for i in range(n):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col]

        # Data for this agent type, for both scenarios
        tau_w_g_base = optimal_tau_w_results_g_base[:, i]
        tau_w_g0 = optimal_tau_w_results_g0[:, i]

        # Filter NaNs for each line
        valid_g_base_idx = ~np.isnan(tau_w_g_base)
        valid_g0_idx = ~np.isnan(tau_w_g0)

        # Plot lines
        if np.any(valid_g_base_idx):
             ax.plot(valid_xi_values[valid_g_base_idx], tau_w_g_base[valid_g_base_idx],
                     linestyle='-', color='tab:blue', label=f'G={G_value_base}')
        if np.any(valid_g0_idx):
             ax.plot(valid_xi_values[valid_g0_idx], tau_w_g0[valid_g0_idx],
                     linestyle='--', color='tab:red', label=f'G={G_value_zero}')

        ax.set_ylabel(rf'$\tau_w^{{{i+1}}}$', fontsize=12, rotation=90)

        # Apply shared interval based on combined data range for this plot
        if max_range_tau_w > 1e-9:
             # Combine data for midpoint calculation for this subplot only
             combined_data_i = np.concatenate((tau_w_g_base[valid_g_base_idx], tau_w_g0[valid_g0_idx]))
             if combined_data_i.size > 0:
                  midpoint_i = (np.max(combined_data_i) + np.min(combined_data_i)) / 2
                  ax.set_ylim(midpoint_i - max_range_tau_w / 2, midpoint_i + max_range_tau_w / 2)

        if row == n_rows - 1: ax.set_xlabel(r'$\xi$', fontsize=12)
        # Add legend to each tau_w subplot
        if np.any(valid_g_base_idx) or np.any(valid_g0_idx):
             ax.legend(fontsize='small')


    # Plot n+1: Plot Lumpsum 'l' for both G values
    if n < n_rows * n_cols:
        row, col = n // n_cols, n % n_cols
        ax = axs[row, col]

        l_g_base = lumpsum_results_g_base
        l_g0 = lumpsum_results_g0
        valid_l_g_base_idx = ~np.isnan(l_g_base)
        valid_l_g0_idx = ~np.isnan(l_g0)

        if np.any(valid_l_g_base_idx):
             ax.plot(valid_xi_values[valid_l_g_base_idx], l_g_base[valid_l_g_base_idx],
                     linestyle='-', color='tab:blue', label=f'G={G_value_base}')
        if np.any(valid_l_g0_idx):
             ax.plot(valid_xi_values[valid_l_g0_idx], l_g0[valid_l_g0_idx],
                     linestyle='--', color='tab:red', label=f'G={G_value_zero}')

        ax.set_ylabel(r'$l$', fontsize=12, rotation=90)
        # Let this plot auto-scale its y-axis

        if row == n_rows - 1: ax.set_xlabel(r'$\xi$', fontsize=12)
        # Add legend to lumpsum plot
        if np.any(valid_l_g_base_idx) or np.any(valid_l_g0_idx):
             ax.legend(fontsize='small')


    # Hide any unused subplots
    for i in range(n + 1, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if row < axs.shape[0] and col < axs.shape[1]: fig.delaxes(axs[row, col])

    # Ensure the directory for saving exists
    output_dir = "xi_sensitivity_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = "xi_sensitivity_grid_g_comparison.pdf" # New filename
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlots saved to {output_path}")
    plt.show()
else:
    print("\nNo successful optimization runs were completed for any scenario.")