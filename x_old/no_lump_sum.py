import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Use the new inner solver that endogenously computes government spending
import inner_lump_sum_full_waste as solver
from inner_lump_sum_full_waste import alpha, beta, gamma, d0, phi, n

phi = n*[(0.2)]
# --- Model Parameters ---
theta = 1.0
xi = 0.1  # Fixed tax aggressiveness parameter

def maximize_welfare_slsqp(xi, elasticity):
    # Objective function for joint optimization over tau_w (vector) and tau_z (scalar)
    def swf_obj(x):
        # x[0:n] are tau_w; x[n] is tau_z.
        tau_w = x[0:n]
        tau_z = x[n]
        try:
            solution, results, converged = solver.solve(tau_w, tau_z)
            if not converged:
                return 1e10  # Penalize non-convergence.
            utilities = results["utilities"]
            agg_polluting = results["z_c"] + results["z_d"]
            # Transformation from no_lump_sum.py:
            welfare = np.sum((1 - elasticity)**(-1) * np.exp(utilities)**(1 - elasticity)- 5*xi * (agg_polluting**theta))
            return -welfare  # We minimize the negative welfare.
        except Exception as e:
            print("Error in objective:", e)
            return 1e10

    # Initial guess: tau_w = zeros for all agents, and tau_z = 0.5.
    x0 = np.concatenate([np.zeros(n), [0.5]])
    # Bounds: for each tau_w, [0,1] and for tau_z, [1e-6, 100]
    bounds = [(0, 1)] * n + [(1e-6, 100.0)]
    
    res = minimize(swf_obj, x0, method="SLSQP", bounds=bounds, options={'disp': False, 'ftol': 1e-7})
    if res.success:
        opt_tau_w = res.x[:n]
        opt_tau_z = res.x[n]
        return opt_tau_w, opt_tau_z, -res.fun
    else:
        print("SLSQP optimization failed:", res.message)
        return None, None, None

# --- Example Run and Plot of Optimal tau_z vs. Elasticity using SLSQP ---
elasticity_values = np.linspace(1.5, 2.5, 100)
optimal_tau_z_list = []

for elas in elasticity_values:
    print(f"Optimizing for elasticity = {elas:.2f} using SLSQP")
    opt_tau_w, opt_tau_z, welfare = maximize_welfare_slsqp(xi, elas)
    if opt_tau_w is None or opt_tau_z is None:
        optimal_tau_z_list.append(np.nan)
    else:
        optimal_tau_z_list.append(opt_tau_z)
optimal_tau_z_list = np.array(optimal_tau_z_list)

plt.figure(figsize=(6,4))
plt.plot(elasticity_values, optimal_tau_z_list, marker='o', linestyle='-', color='blue')
plt.xlabel("Elasticity", fontsize=12)
plt.ylabel(r"Optimal $\tau_z$", fontsize=12)
plt.title(r"Optimal $\tau_z$ vs. Elasticity (SLSQP, $\xi=0.5$)", fontsize=14)
plt.grid(True)
plt.tight_layout()

output_dir = "tax_opt_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "optimal_tau_z_vs_elasticity_slsqp.pdf")
plt.savefig(output_path)
print(f"Figure saved to {output_path}")
plt.show()