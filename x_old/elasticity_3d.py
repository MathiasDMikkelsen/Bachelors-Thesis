import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Import the joint optimization routine from outer_klenert.py.
# It is assumed that outer_klenert.py defines:
#    def maximize_welfare(G, xi, elasticity)
from outer_klenert import maximize_welfare

# --- Model Parameters ---
G = 0.0              # Government consumption
theta = 1.0          # Parameter used in the welfare formula (defined in outer_klenert as well)
# (Other parameters like α, β, γ, etc. remain inside inner_solver.)

# --- Define Parameter Grids ---
# We will vary xi and elasticity simultaneously.
xi_values = np.linspace(0.1, 1.0, 5)         # 20 values for xi
elasticity_values = np.linspace(0.2, 0.6, 5)    # 20 values for elasticity

# Prepare 2D arrays to store optimal τ_z and the τ_w vectors for each (xi, elasticity) combination.
optimal_tau_z_grid = np.full((len(xi_values), len(elasticity_values)), np.nan)
# For τ_w, we assume there are n = 5 households.
optimal_tau_w_grid = np.full((len(xi_values), len(elasticity_values), 5), np.nan)

# --- Loop over the grid to compute optimal taxes ---
for i, xi_val in enumerate(xi_values):
    for j, elas in enumerate(elasticity_values):
        print(f"Optimizing for xi = {xi_val:.3f}, elasticity = {elas:.3f}")
        opt_tau_w, opt_tau_z, max_welfare = maximize_welfare(G, xi_val, elas)
        if opt_tau_w is not None and opt_tau_z is not None:
            optimal_tau_w_grid[i, j, :] = opt_tau_w
            optimal_tau_z_grid[i, j] = opt_tau_z
        else:
            # If optimization fails, leave values as NaN.
            optimal_tau_w_grid[i, j, :] = np.full(5, np.nan)
            optimal_tau_z_grid[i, j] = np.nan

# --- Create 3D Surface Plots ---
output_dir = "tax_opt_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3D plot for optimal τ_z.
X, Y = np.meshgrid(elasticity_values, xi_values)  # X axis: elasticity, Y axis: xi

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, optimal_tau_z_grid, cmap="viridis", edgecolor="none")
ax.set_xlabel("Elasticity", fontsize=12)
ax.set_ylabel("xi", fontsize=12)
ax.set_zlabel(r"Optimal $\tau_z$", fontsize=12)
ax.set_title(r"Optimal $\tau_z$ vs. xi and Elasticity", fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=5)
output_path = os.path.join(output_dir, "optimal_tau_z_3d.pdf")
plt.savefig(output_path)
print(f"3D τ_z surface saved to {output_path}")

# Optionally, create a 3D plot for one household’s τ_w (e.g., household 1).
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot(111, projection='3d')
# Plot optimal τ_w for household 1 (index 0)
surf2 = ax2.plot_surface(X, Y, optimal_tau_w_grid[:, :, 0], cmap="plasma", edgecolor="none")
ax2.set_xlabel("Elasticity", fontsize=12)
ax2.set_ylabel("xi", fontsize=12)
ax2.set_zlabel(r"Optimal $\tau_{w,1}$", fontsize=12)
ax2.set_title(r"Optimal $\tau_{w,1}$ vs. xi and Elasticity", fontsize=14)
fig2.colorbar(surf2, shrink=0.5, aspect=5)
output_path2 = os.path.join(output_dir, "optimal_tau_w_hh1_3d.pdf")
plt.savefig(output_path2)
print(f"3D τ_w (household 1) surface saved to {output_path2}")

plt.show()