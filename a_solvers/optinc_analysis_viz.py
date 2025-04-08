# plot_xi_sensitivity.py (2x3 grid, lumpsum instead of tau_z)

import numpy as np
import matplotlib.pyplot as plt
import outer_solver # Imports the modified outer_solver.py
import inner_solver as solver # Import inner_solver directly to call solve()
from inner_solver import n # Import n (number of agent types)
import os # Import os for saving figure
import math # To calculate grid dimensions if needed later

# --- Simulation Parameters ---
G_value = 5.0
# Define the range and number of xi values to test
xi_values = np.linspace(0, 0.4, 21) # Example: 0 to 0.4 in 21 steps
# --- End Simulation Parameters ---

# Lists to store results
optimal_tau_w_results = []
# optimal_tau_z_results = [] # No longer need to store tau_z for plotting
lumpsum_results = [] # <-- New list to store lumpsum 'l'
valid_xi_values = []

print(f"Running optimization for {len(xi_values)} values of xi...")
print("-" * 30)

# Loop through each xi value
for xi_val in xi_values:
    print(f"  Running for xi = {xi_val:.4f}...")
    l_val = np.nan # Default l value if optimization or solve fails
    try:
        # --- Step 1: Find optimal taxes using outer solver ---
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)

        if opt_tau_w is not None and opt_tau_z is not None:
            print(f"    Outer Success: tau_z = {opt_tau_z:.4f}, SWF = {max_welfare_val:.4f}")

            # --- Step 2: Re-solve inner solver with optimal taxes to get 'l' ---
            try:
                 # Call inner solver with the optimal taxes found
                 inner_solution, inner_results, inner_converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
                 if inner_converged:
                     l_val = inner_results['l'] # Extract the lumpsum value
                     print(f"      Inner Success: l = {l_val:.4f}")
                 else:
                     print(f"      Warning: Inner solver failed to converge with optimal taxes for xi = {xi_val:.4f}")
                     # Keep l_val as np.nan

            except Exception as inner_e:
                 print(f"      Error during inner solve for xi = {xi_val:.4f}: {inner_e}")
                 # Keep l_val as np.nan

            # --- Step 3: Store results ---
            optimal_tau_w_results.append(opt_tau_w)
            lumpsum_results.append(l_val) # Append l_val (could be nan)
            valid_xi_values.append(xi_val) # Still use xi if outer solver succeeded

        else:
            print(f"    Outer Optimization failed for xi = {xi_val:.4f}")
            # Append placeholders if outer opt failed, or adjust logic as needed
            # optimal_tau_w_results.append([np.nan]*n) # Option: append nan arrays
            # lumpsum_results.append(np.nan)
            # valid_xi_values.append(xi_val) # Or maybe don't append xi here

    except Exception as e:
        print(f"    Error during outer optimization run for xi = {xi_val:.4f}: {e}")
        # Append placeholders if outer opt failed
        # optimal_tau_w_results.append([np.nan]*n)
        # lumpsum_results.append(np.nan)
        # valid_xi_values.append(xi_val)


print("-" * 30)
print("Optimization runs complete.")

# --- Plotting ---

# Filter out potential failures where outer opt succeeded but inner failed (lumpsum is nan)
# Or handle NaNs during plotting if preferred
valid_indices = ~np.isnan(lumpsum_results) # Indices where lumpsum is not NaN

if np.sum(valid_indices) > 0: # Check if there are any valid points to plot
    # Convert results to numpy arrays and filter invalid points
    optimal_tau_w_results = np.array(optimal_tau_w_results)[valid_indices]
    lumpsum_results = np.array(lumpsum_results)[valid_indices]
    valid_xi_values = np.array(valid_xi_values)[valid_indices]


    # --- Calculate max range and apply consistent interval for tau_w ---
    max_range_tau_w = 0
    if optimal_tau_w_results.size > 0:
        ranges = [np.ptp(optimal_tau_w_results[:, i]) for i in range(n)]
        max_range_tau_w = np.max(ranges)
        margin_range = max_range_tau_w * 0.10
        if margin_range < 1e-6:
             margin_range = 0.2
        max_range_tau_w += margin_range
    # --- End Calculate max range ---

    # Create figure and axes array for a 2x3 grid
    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.5, 4.34)) # Adjusted height

    # Plot 1 to n: Plot each tau_w with consistent y-axis interval size
    for i in range(n):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        current_tau_w_data = optimal_tau_w_results[:, i]
        ax.plot(valid_xi_values, current_tau_w_data, linestyle='-')
        ax.set_ylabel(rf'$\tau_w^{{{i+1}}}$', fontsize=12, rotation=90)

        if max_range_tau_w > 1e-9:
            midpoint_i = (np.max(current_tau_w_data) + np.min(current_tau_w_data)) / 2
            ax.set_ylim(midpoint_i - max_range_tau_w / 2, midpoint_i + max_range_tau_w / 2)

        if row == n_rows - 1:
             ax.set_xlabel(r'$\xi$', fontsize=12)


    # Plot n+1: Plot Lumpsum 'l' in the next available grid position
    if n < n_rows * n_cols:
        row = n // n_cols
        col = n % n_cols
        ax = axs[row, col]
        # --- Changed to plot lumpsum_results ---
        ax.plot(valid_xi_values, lumpsum_results, linestyle='-', color='tab:green') # Changed color
        # --- Changed y-axis label ---
        ax.set_ylabel(r'$l$', fontsize=12, rotation=90)
        # Let this plot auto-scale its y-axis (no set_ylim)
        if row == n_rows - 1:
            ax.set_xlabel(r'$\xi$', fontsize=12)

    # Hide any unused subplots
    for i in range(n + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row, col])


    # Ensure the directory for saving exists
    output_dir = "xi_sensitivity_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "xi_sensitivity_grid_plots_lumpsum.pdf") # New filename

    plt.tight_layout()
    plt.savefig(output_path) # Save the figure
    print(f"Plots saved to {output_path}")
    plt.show()
else:
    print("\nNo successful optimization runs with valid lumpsum values were completed. Cannot generate plots.")