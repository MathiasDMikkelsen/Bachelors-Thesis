import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import a_inner_solver as solver

# --- Model Parameters ---
G = 0.0            # Government consumption
theta = 1.0        # Parameter used in the welfare formula

# Use fixed τ_w (unequal tax system or zeros – here we use zeros as in ineq_averse.py)
fixed_tau_w = np.zeros(5)

# --- Define Parameter Grids ---
# We'll vary xi (e.g., tax aggressiveness parameter) and elasticity simultaneously.
xi_values = np.linspace(0.05, 0.2, 20)          # 20 values for xi
elasticity_values = np.linspace(0.2, 0.9, 20)     # 20 values for elasticity

# Prepare a 2D array to store optimal τ_z for each (xi, elasticity) combination.
# Rows correspond to xi, and columns to elasticity.
optimal_tau_z_grid = np.full((len(xi_values), len(elasticity_values)), np.nan)

# --- Objective Function: Optimize τ_z with fixed τ_w ---
def objective_tau_z(tau_z_arr, G, xi, elasticity):
    # Extract scalar τ_z from the one-element array
    tau_z = tau_z_arr[0]
    try:
        # Solve the inner model with fixed τ_w, current τ_z, and G.
        sol, results, converged = solver.solve(fixed_tau_w, tau_z, G)
        if not converged:
            return 1e10  # High penalty if the solver does not converge.
    
        utilities = results["utilities"]
        agg_polluting = results["z_c"] + results["z_d"]
        # Compute welfare as defined:
        # welfare = sum(utilities transformed by elasticity) - 5*xi*(agg_polluting**theta)
        # Here we follow the given transformation:
        transformed_term = (1 - elasticity)**(-1) * np.exp(utilities)**(1 - elasticity)
        welfare = np.sum(transformed_term) - 5 * xi * (agg_polluting ** theta)
        # Return the negative welfare for minimization.
        return -welfare
    except Exception as e:
        print(f"Error for τ_z = {tau_z:.4f} with xi = {xi:.2f} and elasticity = {elasticity:.2f}: {e}")
        return 1e10

# --- Loop over the grid to compute optimal τ_z ---
for i, xi_val in enumerate(xi_values):
    for j, elas in enumerate(elasticity_values):
        # Use a standard initial guess for τ_z.
        res = minimize(objective_tau_z,
                       x0=[0.5],
                       args=(G, xi_val, elas),
                       bounds=[(1e-6, 100.0)],
                       method="SLSQP",
                       options={'disp': False, 'ftol': 1e-7, 'maxiter': 200})
        if res.success:
            optimal_tau_z_grid[i, j] = res.x[0]
        else:
            print(f"Optimization failed for xi = {xi_val:.2f}, elasticity = {elas:.2f}")
            optimal_tau_z_grid[i, j] = np.nan

# --- Create a 3D Surface Plot ---
X, Y = np.meshgrid(elasticity_values, xi_values)  # X: elasticity, Y: xi

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, optimal_tau_z_grid, cmap="viridis", edgecolor="none")
ax.set_xlabel("Elasticity", fontsize=12)
ax.set_ylabel("xi", fontsize=12)
ax.set_zlabel(r"Optimal $\tau_z$", fontsize=12)
ax.set_title(r"Optimal $\tau_z$ vs. Elasticity and xi", fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=5)

# --- Save and Show the Figure ---
output_dir = "tax_opt_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "optimal_tau_z_3d.pdf")
plt.savefig(output_path)
print(f"3D surface figure saved to {output_path}")
plt.show()