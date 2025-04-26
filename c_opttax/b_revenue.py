# plot_revenue_comparison_2scen.py (Legend in Two Rows Version)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys, os
from scipy.optimize import minimize
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_solvers_dir = os.path.join(project_root, "a_solvers")
# ensure both are on sys.path
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)
        
from a_solvers import outer_solver 
import a_solvers.inner_solver as solver
from a_solvers.inner_solver import n, alpha, beta, gamma, d0, phi, t as T
import matplotlib as mpl   # only needed once, before any figures are created
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],      # this line makes Matplotlib insert \usepackage{mathpazo}
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}  % give mathpazo the 'sc' option
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})


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

    result = minimize(
        swf_obj_fixed_w,
        [0.5],
        args=(G, xi, fixed_tau_w_arr),
        bounds=[(1e-6, 100)],
        method='SLSQP',
        options={'ftol': 1e-7}
    )
    if result.success:
        return result.x[0], -result.fun
    else:
        return None, None

# Prepare revenue lists
rev_var_env, rev_var_inc, rev_var_tot = [], [], []
rev_fix_env, rev_fix_inc, rev_fix_tot = [], [], []
valid_xi = []

for xi in xi_values:
    valid_xi.append(xi)

    # Variable tau_w scenario
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

    # Fixed tau_w scenario (optimized at xi=0.1)
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

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Variable tau_w lines
ax.plot(valid_xi, rev_var_tot, '-.',   linewidth=2, color='steelblue',   label='Total (var. inc. tax)')
ax.plot(valid_xi, rev_var_env, '-',  linewidth=2, color='steelblue',   label='Env. (var. inc. tax)')
ax.plot(valid_xi, rev_var_inc, '--',   linewidth=2, color='steelblue',   label='Inc. (var. inc. tax)')

# Fixed tau_w lines
ax.plot(valid_xi, rev_fix_tot, '-.',   linewidth=2, color='tab:green', label='Total (baseline opt. fixed inc tax.)')
ax.plot(valid_xi, rev_fix_env, '-',  linewidth=2, color='tab:green', label='Env. (baseline opt. fixed inc tax.)')
ax.plot(valid_xi, rev_fix_inc, '--',   linewidth=2, color='tab:green', label='Inc. (baseline opt. fixed inc tax.)')

# Government spending line
ax.axhline(G_value, color='gray', linestyle='-', linewidth=2, label=r'Gov. spending')

ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
ax.set_ylabel('Revenue', fontsize=14)
ax.set_xlim(xi_values[0], xi_values[-1])

# Legend in two rows below the plot
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.125),
    ncol=4,                # 4 entries on the first row, 3 on the second
    frameon=False,
    fontsize=9
)

# Adjust bottom margin to make room for the two‐row legend
plt.subplots_adjust(bottom=0.30, right=0.98)

# Save figure
plt.savefig("c_opttax/b_revenue.pdf", bbox_inches='tight')
