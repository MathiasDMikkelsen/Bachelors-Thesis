import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import make_interp_spline

# Set up project root so we can import the solver
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a_solvers.inner_solver import solve, phi, t, n, tau_w

# Define parameters and grid for tau_z
tau_w_init = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_opt  = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
g = 5.0
tau_z_values = np.linspace(0.1, 20.0, 50)
tau_z_baseline = 0.1

# --- Baseline Equilibrium for the Two Tax Systems (using tau_z = 0.1) ---
# Tax System 1 Baseline
sol_base1, res_base1, conv_base1 = solve(tau_w_init, tau_z_baseline, g)
if not conv_base1:
    print("Baseline model did not converge for tax system 1 at tau_z = 0.1")
baseline_util_sys1 = res_base1["utilities"]  # assumed to be an array of shape (n,)

# Tax System 2 Baseline
sol_base2, res_base2, conv_base2 = solve(tau_w_opt, tau_z_baseline, g)
if not conv_base2:
    print("Baseline model did not converge for tax system 2 at tau_z = 0.1")
baseline_util_sys2 = res_base2["utilities"]

# --- Solve for New Equilibria across tau_z_values ---
# We'll store the utility (already in log form) for each household and each tau_z.
utility_new_sys1 = np.zeros((n, len(tau_z_values)))
utility_new_sys2 = np.zeros((n, len(tau_z_values)))

for j, tau_z in enumerate(tau_z_values):
    # Solve for Tax System 1
    sol1, res1, conv1 = solve(tau_w_init, tau_z, g)
    if not conv1:
        print(f"Model did not converge for tax system 1 at tau_z = {tau_z:.2f}; using baseline utility.")
        utility_new_sys1[:, j] = baseline_util_sys1
    else:
        utility_new_sys1[:, j] = res1["utilities"]
    
    # Solve for Tax System 2
    sol2, res2, conv2 = solve(tau_w_opt, tau_z, g)
    if not conv2:
        print(f"Model did not converge for tax system 2 at tau_z = {tau_z:.2f}; using baseline utility.")
        utility_new_sys2[:, j] = baseline_util_sys2
    else:
        utility_new_sys2[:, j] = res2["utilities"]

# --- Compute the Change in Utility Relative to Baseline ---
diff_util_sys1 = utility_new_sys1 - baseline_util_sys1.reshape(-1, 1)
diff_util_sys2 = utility_new_sys2 - baseline_util_sys2.reshape(-1, 1)

# Sum the differences over households.
aggregate_diff_sys1 = np.sum(diff_util_sys1, axis=0)
aggregate_diff_sys2 = np.sum(diff_util_sys2, axis=0)

# --- Smoothing the curves ---
tau_z_smooth = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
spline_sys1 = make_interp_spline(tau_z_values, aggregate_diff_sys1)
spline_sys2 = make_interp_spline(tau_z_values, aggregate_diff_sys2)

agg_diff_sys1_smooth = spline_sys1(tau_z_smooth)
agg_diff_sys2_smooth = spline_sys2(tau_z_smooth)

# --- Calculate the derivatives (slopes) of the two curves ---
deriv_sys1 = spline_sys1.derivative()(tau_z_smooth)
deriv_sys2 = spline_sys2.derivative()(tau_z_smooth)

# --- Find the tau_z where the slopes are equal (deriv_sys1 - deriv_sys2 == 0) ---
tau_z_equal_slope = None
diff_deriv = deriv_sys1 - deriv_sys2

for i in range(len(tau_z_smooth) - 1):
    if diff_deriv[i] * diff_deriv[i+1] < 0:  # sign change detected
        # Linear interpolation between tau_z_smooth[i] and tau_z_smooth[i+1]
        t0, t1 = tau_z_smooth[i], tau_z_smooth[i+1]
        d0, d1 = diff_deriv[i], diff_deriv[i+1]
        tau_z_equal_slope = t0 - d0 * (t1 - t0) / (d1 - d0)
        break

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.plot(tau_z_smooth, agg_diff_sys1_smooth, color='tab:blue', linewidth=2,
         label='Tax System 1')
plt.plot(tau_z_smooth, agg_diff_sys2_smooth, color='tab:red', linewidth=2,
         label='Tax System 2')

# Add horizontal line at 0 for reference.
plt.axhline(0, color='grey', linestyle='--', linewidth=1)

# Mark the tax level where the slopes are equal.
if tau_z_equal_slope is not None:
    plt.axvline(tau_z_equal_slope, color='grey', linestyle='--', linewidth=1)
    plt.plot(tau_z_equal_slope, spline_sys1(tau_z_equal_slope), marker='o', color='grey',
             markersize=8, label=f'Equal Slope @ τ_z ≈ {tau_z_equal_slope:.2f}')
    plt.text(tau_z_equal_slope, spline_sys1(tau_z_equal_slope) + 0.02,
             f'{tau_z_equal_slope:.2f}', color='grey', fontsize=12,
             ha='center', va='bottom')

plt.xlabel(r'$\tau_z$', fontsize=12)
plt.ylabel('Aggregate Change in Log Utility', fontsize=12)
plt.title('Aggregate Change and Slope Equality Across Households', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("b_dynamics/x_bluesw1.pdf")
plt.show()

if tau_z_equal_slope is not None:
    print(f'The slopes of the two curves are equal at approximately τ_z = {tau_z_equal_slope:.2f}')
else:
    print('No crossing in slopes was found in the evaluated range.')