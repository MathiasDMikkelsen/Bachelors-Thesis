import numpy as np
import matplotlib.pyplot as plt
import os

# Import the maximize_welfare solver from outer_klenert.py.
# It is assumed that outer_klenert.py defines:
#    def maximize_welfare(G, xi, elasticity)
from outer_klenert import maximize_welfare
# Also import the inner solver to check the market residuals.
import inner_solver as solver

# --- Simulation Parameters ---
G = 5.0           # Government consumption fixed at 5.0
xi = 0.1          # Fixed xi value
# Choose 100 elasticity values between 0.2 and 0.9:
elasticity_values = np.linspace(0.2, 0.6, 100)

# Prepare arrays to store results:
# optimal_tau_z is scalar, optimal_tau_w is a vector of length n (n=5 households)
optimal_tau_z_list = []
optimal_tau_w_list = []  # Each element is a vector (length 5)

# Loop over elasticity values:
for elas in elasticity_values:
    print(f"Solving for elasticity = {elas:.2f}")
    opt_tau_w, opt_tau_z, welfare = maximize_welfare(G, xi, elas)
    
    # If optimization fails, store NaNs.
    if opt_tau_w is None or opt_tau_z is None:
        optimal_tau_w_list.append(np.full(5, np.nan))
        optimal_tau_z_list.append(np.nan)
    else:
        # Re-run the inner solver with the obtained optimal values so we can inspect market residuals.
        sol, results, conv = solver.solve(opt_tau_w, opt_tau_z, G)
        # Check market residuals (e.g. budget_errors) – discard any solution where any residual is > 0.1.
        if np.max(np.abs(results["budget_errors"])) > 0.1:
            print(f"Discarding solution for elasticity = {elas:.2f} due to high market residual.")
            optimal_tau_w_list.append(np.full(5, np.nan))
            optimal_tau_z_list.append(np.nan)
        else:
            optimal_tau_w_list.append(opt_tau_w)
            optimal_tau_z_list.append(opt_tau_z)

optimal_tau_z_list = np.array(optimal_tau_z_list)  # shape (100,)
optimal_tau_w_list = np.array(optimal_tau_w_list)  # shape (100, 5)

# --- Normalize τ_w relative to the baseline (first elasticity index) ---
relative_tau_w_list = np.copy(optimal_tau_w_list)
for i in range(relative_tau_w_list.shape[1]):
    baseline = relative_tau_w_list[0, i]
    if baseline != 0:
        relative_tau_w_list[:, i] = relative_tau_w_list[:, i] / baseline
    else:
        relative_tau_w_list[:, i] = np.nan

# Create an output directory for plots if it doesn't exist
output_dir = "tax_opt_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Figure 1: Optimal τ_z vs. Elasticity ---
plt.figure(figsize=(5, 3.5))
plt.plot(elasticity_values, optimal_tau_z_list, linestyle='-')
plt.xlabel("Elasticity")
plt.ylabel(r"Optimal $\tau_z$")
plt.title("Optimal $\tau_z$ vs. Elasticity")
plt.grid(True)
plt.tight_layout()
output_path = os.path.join(output_dir, "optimal_tau_z_vs_elasticity.pdf")
plt.savefig(output_path)
print(f"\nFigure saved to {output_path}")

# --- Figure 2: Relative Optimal τ_w vs. Elasticity (All Households) ---
plt.figure(figsize=(5, 3.5))
for i in range(5):
    plt.plot(elasticity_values, relative_tau_w_list[:, i]
            , linestyle='-', label=f"Household {i+1}")
plt.xlabel("Elasticity")
plt.ylabel("Relative Optimal $\tau_w$")
plt.title("Relative Optimal $\tau_w$ vs. Elasticity\n(Baseline = First Elasticity)")
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
output_path = os.path.join(output_dir, "relative_tau_w_all_households_vs_elasticity.pdf")
plt.savefig(output_path)
print(f"Combined relative figure saved to {output_path}")

plt.show()