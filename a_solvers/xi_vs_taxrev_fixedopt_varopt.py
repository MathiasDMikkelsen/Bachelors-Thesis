# plot_revenue_comparison_2scen.py (Final Adjusted Version with Right Legend)

# --- Changes implemented:
# - All lines linewidth=2
# - Remove box around legend
# - Fixed tau_w scenario lines in orange
# - Variable tau_w scenario lines in blue
# - Changed figsize to (7,5)
# - Legend positioned to the right of the figure in one column

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
import outer_solver
import inner_solver as solver
from inner_solver import n, alpha, beta, gamma, d0, phi, t as T

G_value = 5.0
theta = 1.0

fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
xi_values = np.linspace(0.1, 1.0, 25)

def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0]
        solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
        if not converged:
            return 1e10
        utilities = results['utilities']
        agg_polluting = results['z_c'] + results['z_d']
        valid_utilities = utilities[utilities > -1e5]
        welfare = np.sum(valid_utilities) - 5 * xi_val * (agg_polluting ** theta)
        return -welfare

    result = minimize(swf_obj_fixed_w, [0.5], args=(G, xi, fixed_tau_w_arr),
                      bounds=[(1e-6, 100)], method='SLSQP', options={'ftol': 1e-7})
    return (result.x[0], -result.fun) if result.success else (None, None)

rev_var_env, rev_var_inc, rev_var_tot = [], [], []
rev_fix_env, rev_fix_inc, rev_fix_tot = [], [], []
valid_xi = []

for xi in xi_values:
    valid_xi.append(xi)

    # Variable tau_w
    opt_tau_w, opt_tau_z, _ = outer_solver.maximize_welfare(G_value, xi)
    _, results, converged = solver.solve(opt_tau_w, opt_tau_z, G_value)
    if converged:
        rev_var_env.append(opt_tau_z * (results['z_c'] + results['z_d']))
        labor_income = phi * results['w'] * np.maximum(T - results['l_agents'], 0)
        rev_var_inc.append(np.sum(opt_tau_w * labor_income))
        rev_var_tot.append(rev_var_env[-1] + rev_var_inc[-1])
    else:
        rev_var_env.append(np.nan)
        rev_var_inc.append(np.nan)
        rev_var_tot.append(np.nan)

    # Fixed tau_w
    opt_tau_z_fix, _ = maximize_welfare_fixed_w(G_value, xi, fixed_tau_w_optimal_xi01)
    _, results, converged = solver.solve(fixed_tau_w_optimal_xi01, opt_tau_z_fix, G_value)
    if converged:
        rev_fix_env.append(opt_tau_z_fix * (results['z_c'] + results['z_d']))
        labor_income = phi * results['w'] * np.maximum(T - results['l_agents'], 0)
        rev_fix_inc.append(np.sum(fixed_tau_w_optimal_xi01 * labor_income))
        rev_fix_tot.append(rev_fix_env[-1] + rev_fix_inc[-1])
    else:
        rev_fix_env.append(np.nan)
        rev_fix_inc.append(np.nan)
        rev_fix_tot.append(np.nan)

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(valid_xi, rev_var_tot, '-', color='tab:blue', linewidth=2, label='Total (var. $\\tau_w$)')
ax.plot(valid_xi, rev_var_env, '--', color='tab:blue', linewidth=2, label='Env. (var. $\\tau_w$)')
ax.plot(valid_xi, rev_var_inc, ':', color='tab:blue', linewidth=2, label='Inc. (var. $\\tau_w$)')

ax.plot(valid_xi, rev_fix_tot, '-', color='tab:orange', linewidth=2, label='Total (fixed $\\tau_w$ opt. at $\\xi=0.1$)')
ax.plot(valid_xi, rev_fix_env, '--', color='tab:orange', linewidth=2, label='Env. (fixed $\\tau_w$ opt. at $\\xi=0.1$)')
ax.plot(valid_xi, rev_fix_inc, ':', color='tab:orange', linewidth=2, label='Inc. (fixed $\\tau_w$ opt. at $\\xi=0.1$)')

ax.axhline(G_value, color='gray', linestyle='--', linewidth=2, label=rf'Gov. spending')

ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel('Revenue', fontsize=14)

ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='medium', frameon=False)

plt.subplots_adjust(right=0.75)

output_dir = "c_opttax"
os.makedirs(output_dir, exist_ok=True)
output_filename = "f_revenue.pdf"
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

plt.show()
