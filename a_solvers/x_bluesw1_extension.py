# plot_utility_change_abatement.py (Using Abatement + Direct HH Redist Solver)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import make_interp_spline

# --- MODIFIED: Ensure correct path and import LATEST inner_solver ---
# Set up project root so we can import the solver
# Adjust path logic if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import the latest inner_solver.py (with abatement + direct HH redist)
    # Assuming it's named inner_solver_ext.py based on user's last code
    from inner_solver_ext import solve, phi, t, n
except ImportError:
    print("Error: Could not import 'inner_solver_ext'.")
    print("Ensure the solver file exists and is accessible.")
    exit()
except AttributeError as e:
    print(f"Fatal Error: A required parameter might be missing from 'inner_solver_ext.py': {e}")
    exit()
# --- End Import Modification ---

# --- Assert n=5 ---
assert n == 5, "This script expects n=5 based on the imported inner_solver."

# Define parameters and grid for tau_z
# Using tau_w values from user's previous plotting script version
tau_w_init = np.array([0.015, 0.072, 0.115, 0.156, 0.24]) # Pre-existing
tau_w_opt  = np.array([-1.08858208, -0.04377549,  0.22144972,  0.39697164,  0.64084534]) # Optimal (as used before)
g = 5.0
tau_z_values = np.linspace(0.4, 4.0, 25) # Adjusted range from user code
tau_z_baseline = 0.4 # Use the start of the range as baseline

# --- Baseline Equilibrium for the Two Tax Systems ---
print("Calculating baseline equilibrium...")
# Tax System 1 Baseline
sol_base1, res_base1, conv_base1 = solve(tau_w_init, tau_z_baseline, g)
if not conv_base1 or res_base1 is None:
    print("FATAL: Baseline model failed for tax system 1. Cannot proceed.")
    exit()
baseline_util_sys1 = res_base1["utilities"] # log(u_tilde_i)

# Tax System 2 Baseline
sol_base2, res_base2, conv_base2 = solve(tau_w_opt, tau_z_baseline, g)
if not conv_base2 or res_base2 is None:
    print("FATAL: Baseline model failed for tax system 2. Cannot proceed.")
    exit()
baseline_util_sys2 = res_base2["utilities"] # log(u_tilde_i)
print("Baseline calculations complete.")

# --- Solve for New Equilibria across tau_z_values ---
print("Running simulations across tau_z range...")
utility_new_sys1 = np.zeros((n, len(tau_z_values)))
utility_new_sys2 = np.zeros((n, len(tau_z_values)))
convergence_issues_sys1 = False
convergence_issues_sys2 = False

for j, tau_z in enumerate(tau_z_values):
    # Solve for Tax System 1
    sol1, res1, conv1 = solve(tau_w_init, tau_z, g)
    if not conv1 or res1 is None:
        print(f"Warning: Model did not converge for tax system 1 at tau_z = {tau_z:.2f}; storing NaNs.")
        utility_new_sys1[:, j] = np.nan
        convergence_issues_sys1 = True
    else:
        utility_new_sys1[:, j] = res1["utilities"]

    # Solve for Tax System 2
    sol2, res2, conv2 = solve(tau_w_opt, tau_z, g)
    if not conv2 or res2 is None:
        print(f"Warning: Model did not converge for tax system 2 at tau_z = {tau_z:.2f}; storing NaNs.")
        utility_new_sys2[:, j] = np.nan
        convergence_issues_sys2 = True
    else:
        utility_new_sys2[:, j] = res2["utilities"]
print("Simulations finished.")

# --- Compute the Change in Utility Relative to Baseline ---
# Ensure baseline utility is broadcast correctly for subtraction
diff_util_sys1 = utility_new_sys1 - baseline_util_sys1[:, np.newaxis]
diff_util_sys2 = utility_new_sys2 - baseline_util_sys2[:, np.newaxis]

# Sum the differences over households (axis=0). Use nanmean to ignore NaNs if needed, but sum is likely intended.
# If summing, NaN propagates. We'll filter NaNs before spline fitting instead.
aggregate_diff_sys1 = np.sum(diff_util_sys1, axis=0)
aggregate_diff_sys2 = np.sum(diff_util_sys2, axis=0)

# --- Smoothing and Derivatives (Filter NaNs before fitting) ---
valid_idx_sys1 = ~np.isnan(aggregate_diff_sys1)
valid_idx_sys2 = ~np.isnan(aggregate_diff_sys2)

tau_z_smooth = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
agg_diff_sys1_smooth = np.full_like(tau_z_smooth, np.nan)
agg_diff_sys2_smooth = np.full_like(tau_z_smooth, np.nan)
deriv_sys1 = np.full_like(tau_z_smooth, np.nan)
deriv_sys2 = np.full_like(tau_z_smooth, np.nan)

if np.sum(valid_idx_sys1) > 3: # Need enough points for cubic spline
    spline_sys1 = make_interp_spline(tau_z_values[valid_idx_sys1], aggregate_diff_sys1[valid_idx_sys1])
    agg_diff_sys1_smooth = spline_sys1(tau_z_smooth)
    deriv_sys1 = spline_sys1.derivative()(tau_z_smooth)
else:
    print("Warning: Not enough valid data points for spline fitting (System 1).")

if np.sum(valid_idx_sys2) > 3: # Need enough points for cubic spline
    spline_sys2 = make_interp_spline(tau_z_values[valid_idx_sys2], aggregate_diff_sys2[valid_idx_sys2])
    agg_diff_sys2_smooth = spline_sys2(tau_z_smooth)
    deriv_sys2 = spline_sys2.derivative()(tau_z_smooth)
else:
    print("Warning: Not enough valid data points for spline fitting (System 2).")


# --- Find the tau_z where the slopes are equal ---
tau_z_equal_slope = None
# Ensure derivatives are valid numbers before calculating difference
valid_deriv_idx = ~np.isnan(deriv_sys1) & ~np.isnan(deriv_sys2)
if np.any(valid_deriv_idx):
    diff_deriv = deriv_sys1[valid_deriv_idx] - deriv_sys2[valid_deriv_idx]
    tau_z_smooth_valid = tau_z_smooth[valid_deriv_idx]
    # Find sign changes in the difference of derivatives
    sign_changes = np.where(np.diff(np.sign(diff_deriv)))[0]

    if len(sign_changes) > 0:
        idx = sign_changes[0] # Take the first crossing
        # Linear interpolation to find the crossing point
        t0, t1 = tau_z_smooth_valid[idx], tau_z_smooth_valid[idx+1]
        d0, d1 = diff_deriv[idx], diff_deriv[idx+1]
        if (d1 - d0) != 0: # Avoid division by zero
            tau_z_equal_slope = t0 - d0 * (t1 - t0) / (d1 - d0)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))

# Plot only valid data
plt.plot(tau_z_smooth[~np.isnan(agg_diff_sys1_smooth)], agg_diff_sys1_smooth[~np.isnan(agg_diff_sys1_smooth)],
         color='tab:blue', linewidth=2, label='Pre-existing $\\tau_w$') # Simpler label
plt.plot(tau_z_smooth[~np.isnan(agg_diff_sys2_smooth)], agg_diff_sys2_smooth[~np.isnan(agg_diff_sys2_smooth)],
         color='tab:red', linewidth=2, label='Optimal Baseline $\\tau_w$') # Simpler label

plt.axhline(0, color='grey', linestyle='--', linewidth=1)

# Mark the equal slope point
if tau_z_equal_slope is not None:
    # Ensure the slope equality point is within the plotted range and utility is valid
    if tau_z_equal_slope >= tau_z_smooth.min() and tau_z_equal_slope <= tau_z_smooth.max():
         y_intersect1 = spline_sys1(tau_z_equal_slope) if np.sum(valid_idx_sys1) > 3 else np.nan
         if not np.isnan(y_intersect1):
             plt.axvline(tau_z_equal_slope, color='grey', linestyle=':', linewidth=1.5)
             plt.plot(tau_z_equal_slope, y_intersect1, marker='o', color='grey',
                      markersize=8, label=f'Equal Slope @ $\\tau_z \\approx$ {tau_z_equal_slope:.2f}')
             # Removed text label on plot as it can overlap with axvline label

plt.xlabel(r'Environmental Tax ($\tau_z$)', fontsize=12)
plt.ylabel('Aggregate Change in Log Utility vs Baseline', fontsize=12)
# UPDATED Title
plt.title('Aggregate Log Utility Change (Abatement Model)', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Save figure
output_dir = "aggregate_dynamics_abatement" # Use same directory as other plot
if not os.path.exists(output_dir): os.makedirs(output_dir)
# UPDATED Filename
output_path = os.path.join(output_dir, "utility_change_slope_abatement.pdf")
plt.savefig(output_path)
print(f"\nUtility change plot saved to {output_path}")
plt.show()

if tau_z_equal_slope is not None:
    print(f'\nThe slopes of the two curves are equal at approximately tau_z = {tau_z_equal_slope:.2f}')
else:
    print('\nNo crossing in slopes was found in the evaluated range or insufficient data.')