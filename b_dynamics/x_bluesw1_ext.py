# plot_agg_utility_change_2hh.py (Modified for 2-household solver)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import make_interp_spline
import warnings

# --- MODIFIED: Import correct 2-hh inner solver and params ---
try:
    # Set up project root so we can import the solver
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    solver_path = os.path.join(project_root, 'a_solvers')
    if solver_path not in sys.path:
        sys.path.insert(0, solver_path)
    import inner_solver_ext as solver # Use alias solver
    # Import necessary params from the 2-hh inner solver
    from inner_solver_ext import n, alpha, beta, gamma, d0, t as T
    assert n == 2, "Loaded inner solver is not for n=2 households."
except (ImportError, ModuleNotFoundError):
    print("Error: Could not import 'inner_solver_ext'.")
    print("Ensure 'inner_solver_ext.py' exists in the 'a_solvers' directory relative to this script's parent directory.")
    sys.exit(1)
except FileNotFoundError:
     print("Warning: Could not automatically determine project root. Assuming solver is importable.")
     import inner_solver_ext as solver
     from inner_solver_ext import n
     assert n == 2, "Loaded inner solver is not for n=2 households."
# -----------------------------------------------------------

# --- MODIFIED: Define tau_w policies of length 2 ---
tau_w_policy_a = np.array([0.10, 0.15]) # Example Policy A: [tau_w_d, tau_w_c]
tau_w_policy_b = np.array([0.05, 0.20]) # Example Policy B: [tau_w_d, tau_w_c]
# ---------------------------------------------------
g = 5.0
tau_z_values = np.linspace(0.1, 20.0, 50)
tau_z_baseline = 0.1 # Use the first value as baseline

# --- Baseline Equilibrium for the Two Tax Systems (using tau_z = tau_z_baseline) ---
print(f"Calculating baseline utilities at tau_z = {tau_z_baseline:.2f}...")

# Policy A Baseline
try:
    # --- MODIFIED: Use new solver and policy A ---
    sol_base_a, res_base_a, conv_base_a = solver.solve(tau_w_policy_a, tau_z_baseline, g)
    if not conv_base_a:
        warnings.warn(f"Baseline model did not converge for policy A at tau_z = {tau_z_baseline}")
        baseline_util_sys_a = np.full(n, np.nan) # Use NaN array of size n=2
    else:
        baseline_util_sys_a = res_base_a["utilities"] # Array of shape (2,)
except Exception as e:
    warnings.warn(f"Error solving baseline for policy A: {e}")
    baseline_util_sys_a = np.full(n, np.nan)

# Policy B Baseline
try:
    # --- MODIFIED: Use new solver and policy B ---
    sol_base_b, res_base_b, conv_base_b = solver.solve(tau_w_policy_b, tau_z_baseline, g)
    if not conv_base_b:
        warnings.warn(f"Baseline model did not converge for policy B at tau_z = {tau_z_baseline}")
        baseline_util_sys_b = np.full(n, np.nan) # Use NaN array of size n=2
    else:
        baseline_util_sys_b = res_base_b["utilities"] # Array of shape (2,)
except Exception as e:
    warnings.warn(f"Error solving baseline for policy B: {e}")
    baseline_util_sys_b = np.full(n, np.nan)

# Check if baseline calculations failed completely
if np.all(np.isnan(baseline_util_sys_a)) or np.all(np.isnan(baseline_util_sys_b)):
    print("Error: Cannot proceed without valid baseline utilities.")
    exit()

print("Baseline calculations complete.")

# --- Solve for New Equilibria across tau_z_values ---
print("Solving across tau_z range...")
# --- MODIFIED: Initialize arrays for n=2 households ---
utility_new_sys_a = np.zeros((n, len(tau_z_values)))
utility_new_sys_b = np.zeros((n, len(tau_z_values)))
# ----------------------------------------------------

for j, tau_z in enumerate(tau_z_values):
    print(f"  Processing tau_z = {tau_z:.2f} ({j+1}/{len(tau_z_values)})")
    # Solve for Policy A
    try:
        # --- MODIFIED: Use new solver and policy A ---
        sol_a, res_a, conv_a = solver.solve(tau_w_policy_a, tau_z, g)
        if not conv_a:
            print(f"    Warning: Model did not converge for policy A at tau_z = {tau_z:.2f}; using NaN.")
            utility_new_sys_a[:, j] = np.nan # Store NaN for both agents
        else:
            utility_new_sys_a[:, j] = res_a["utilities"] # Store utilities [u_d, u_c]
    except Exception as e:
        print(f"    Error solving for policy A at tau_z={tau_z:.2f}: {e}")
        utility_new_sys_a[:, j] = np.nan

    # Solve for Policy B
    try:
        # --- MODIFIED: Use new solver and policy B ---
        sol_b, res_b, conv_b = solver.solve(tau_w_policy_b, tau_z, g)
        if not conv_b:
            print(f"    Warning: Model did not converge for policy B at tau_z = {tau_z:.2f}; using NaN.")
            utility_new_sys_b[:, j] = np.nan # Store NaN for both agents
        else:
            utility_new_sys_b[:, j] = res_b["utilities"] # Store utilities [u_d, u_c]
    except Exception as e:
        print(f"    Error solving for policy B at tau_z={tau_z:.2f}: {e}")
        utility_new_sys_b[:, j] = np.nan


print("Solver runs complete.")

# --- Compute the Change in Utility Relative to Baseline ---
# Reshape baseline utilities to allow broadcasting (shape becomes (2, 1))
# Use np.nanmean or handle NaNs if baseline failed for one system
baseline_util_sys_a_col = baseline_util_sys_a.reshape(-1, 1)
baseline_util_sys_b_col = baseline_util_sys_b.reshape(-1, 1)

# Calculate differences (result shape (2, len(tau_z_values)))
diff_util_sys_a = utility_new_sys_a - baseline_util_sys_a_col
diff_util_sys_b = utility_new_sys_b - baseline_util_sys_b_col

# Sum the differences over the 2 households (axis=0). Use nansum.
aggregate_diff_sys_a = np.nansum(diff_util_sys_a, axis=0)
aggregate_diff_sys_b = np.nansum(diff_util_sys_b, axis=0)

# --- Smoothing and Derivatives (Handle potential NaNs) ---
# Filter out NaNs before creating splines
valid_idx_a = ~np.isnan(aggregate_diff_sys_a)
valid_idx_b = ~np.isnan(aggregate_diff_sys_b)

tau_z_smooth = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
agg_diff_sys_a_smooth = np.full_like(tau_z_smooth, np.nan)
agg_diff_sys_b_smooth = np.full_like(tau_z_smooth, np.nan)
deriv_sys1 = np.full_like(tau_z_smooth, np.nan)
deriv_sys2 = np.full_like(tau_z_smooth, np.nan)

if np.sum(valid_idx_a) > 3: # Need enough points for spline
    spline_sys_a = make_interp_spline(tau_z_values[valid_idx_a], aggregate_diff_sys_a[valid_idx_a], k=3) # k=3 for cubic spline
    agg_diff_sys_a_smooth = spline_sys_a(tau_z_smooth)
    deriv_sys1 = spline_sys_a.derivative()(tau_z_smooth)
else:
    print("Warning: Not enough valid data points for Policy A spline.")

if np.sum(valid_idx_b) > 3:
    spline_sys_b = make_interp_spline(tau_z_values[valid_idx_b], aggregate_diff_sys_b[valid_idx_b], k=3)
    agg_diff_sys_b_smooth = spline_sys_b(tau_z_smooth)
    deriv_sys2 = spline_sys_b.derivative()(tau_z_smooth)
else:
    print("Warning: Not enough valid data points for Policy B spline.")

# --- Find where slopes are equal ---
tau_z_equal_slope = None
# Only calculate if both derivatives are valid
if not (np.all(np.isnan(deriv_sys1)) or np.all(np.isnan(deriv_sys2))):
    diff_deriv = deriv_sys1 - deriv_sys2
    # Find sign changes (more robust)
    signs = np.sign(diff_deriv)
    # Check where sign goes from + to - or - to + (ignores zero crossing exactly at a point)
    crossings = np.where(np.diff(signs))[0]

    if len(crossings) > 0:
        # Take the first crossing index
        idx = crossings[0]
        # Linear interpolation for a more precise crossing point
        t0, t1 = tau_z_smooth[idx], tau_z_smooth[idx+1]
        d0, d1 = diff_deriv[idx], diff_deriv[idx+1]
        if (d1 - d0) != 0: # Avoid division by zero if derivative difference is flat
             tau_z_equal_slope = t0 - d0 * (t1 - t0) / (d1 - d0)
             # Ensure the intersection point is within the interpolated segment
             if not (min(t0, t1) <= tau_z_equal_slope <= max(t0, t1)):
                 tau_z_equal_slope = None # Interpolation failed or outside range
else:
    print("Warning: Cannot calculate slope difference due to NaN derivatives.")


# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))

# Plot smoothed aggregate utility changes
plt.plot(tau_z_smooth, agg_diff_sys_a_smooth, color='tab:blue', linewidth=2,
         label=f'Policy A ($\\tau_w$={tau_w_policy_a})') # Updated Label
plt.plot(tau_z_smooth, agg_diff_sys_b_smooth, color='tab:red', linewidth=2,
         label=f'Policy B ($\\tau_w$={tau_w_policy_b})') # Updated Label

# Add horizontal line at 0
plt.axhline(0, color='grey', linestyle='--', linewidth=1)

# Mark where slopes are equal
if tau_z_equal_slope is not None:
    # Find corresponding utility change values at the intersection point using splines
    util_change_at_intersect_a = np.nan
    util_change_at_intersect_b = np.nan
    if np.sum(valid_idx_a) > 3: util_change_at_intersect_a = spline_sys_a(tau_z_equal_slope)
    if np.sum(valid_idx_b) > 3: util_change_at_intersect_b = spline_sys_b(tau_z_equal_slope)
    # Plot marker near the average height of the two curves at intersection
    plot_height = np.nanmean([util_change_at_intersect_a, util_change_at_intersect_b])

    plt.axvline(tau_z_equal_slope, color='dimgrey', linestyle=':', linewidth=1.5)
    if not np.isnan(plot_height):
        plt.plot(tau_z_equal_slope, plot_height, marker='o', color='dimgrey',
                 markersize=8, label=f'Equal Slope @ $\\tau_z \\approx$ {tau_z_equal_slope:.2f}')
        # Adjusted text position slightly
        plt.text(tau_z_equal_slope, plot_height, # Removed offset for simplicity
                 f'{tau_z_equal_slope:.2f}', color='dimgrey', fontsize=11,
                 ha='center', va='bottom', backgroundcolor='white', alpha=0.7) # Added background

plt.xlabel(r'Environmental Tax ($\tau_z$)', fontsize=12)
plt.ylabel('Aggregate Change in Log Utility (vs. $\\tau_z$=0.1)', fontsize=12)
plt.title('Aggregate Utility Change Comparison (2 Households)', fontsize=14)
plt.legend(fontsize=11, loc='best')
plt.grid(True, linestyle=':')
plt.tight_layout()

# Save the figure
output_dir = "b_dynamics_2hh" # Use same directory as other 2hh plots
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "x_agg_util_change_2hh.pdf") # New filename
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")

plt.show()

if tau_z_equal_slope is not None:
    print(f'\nSlopes are equal at approximately tau_z = {tau_z_equal_slope:.3f}')
else:
    print('\nNo crossing in slopes was found in the evaluated range or insufficient data.')