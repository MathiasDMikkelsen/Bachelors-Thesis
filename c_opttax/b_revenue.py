# plot_revenue_comparison_2scen.py (Updated for new solver signatures)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import minimize
import matplotlib as mpl   # only needed once, before any figures are created

# —————————————————————————————————————————————————————————————————————————————
# LaTeX styling
# —————————————————————————————————————————————————————————————————————————————
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],      # Matplotlib inserts \usepackage{mathpazo}
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}  % give mathpazo the 'sc' option
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

# —————————————————————————————————————————————————————————————————————————————
# Project setup: add solver paths
# —————————————————————————————————————————————————————————————————————————————
project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_solvers_dir = os.path.join(project_root, "a_solvers")
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# import updated solvers
from a_solvers import outer_solver
import a_solvers.inner_solver as solver
from a_solvers.inner_solver import n, alpha, beta, gamma, d0 as d0_base, phi, t as T

# —————————————————————————————————————————————————————————————————————————————
# Parameters & helper
# —————————————————————————————————————————————————————————————————————————————
G_value = 5.0
theta   = 1.0
# CES exponent and baseline durables demand for new solver
r_base  = -1.0

# pre‐computed “optimal” fixed‐τ_w at ξ=0.1
fixed_tau_w_optimal_xi01 = np.array([-1.12963781, -0.06584074,  0.2043803, 0.38336986, 0.63241591])

# ξ grid for plotting
xi_values = np.linspace(0.1, 1.0, 25)

# helper: optimize τ_z holding τ_w fixed, using new solver signature

def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    def swf_obj_fixed_w(x, G_val, xi_val, fw_arr):
        tau_z = float(x[0])
        sol, res, conv = solver.solve(fw_arr, tau_z, G_val, r_base, d0_base)
        if not conv:
            return 1e10
        utilities   = res['utilities']
        agg_pollute = res['z_c'] + res['z_d']
        welfare     = np.sum(utilities) - 5 * xi_val * (agg_pollute**theta)
        return -welfare

    result = minimize(
        swf_obj_fixed_w,
        x0=[0.5],
        args=(G, xi, fixed_tau_w_arr),
        bounds=[(1e-6, 100.0)],
        method='SLSQP',
        options={'ftol': 1e-7}
    )
    if result.success:
        return result.x[0], -result.fun
    else:
        return None, None

# —————————————————————————————————————————————————————————————————————————————
# 1) Compute & plot: “variable τ_w” vs. “optimal‐fixed τ_w”
# —————————————————————————————————————————————————————————————————————————————
rev_var_env, rev_var_inc, rev_var_tot = [], [], []
rev_fix_env, rev_fix_inc, rev_fix_tot = [], [], []
valid_xi = []

for xi in xi_values:
    valid_xi.append(xi)

    # — Variable τ_w scenario —————————————————————————————
    opt_tw, opt_tz, _ = outer_solver.maximize_welfare(G_value, xi, r_base, d0_base)
    _, res_var, conv = solver.solve(opt_tw, opt_tz, G_value, r_base, d0_base)
    if conv:
        rev_var_env.append(opt_tz * (res_var['z_c'] + res_var['z_d']))
        labor_inc = phi * res_var['w'] * np.maximum(T - res_var['l_agents'], 0)
        rev_var_inc.append(np.sum(opt_tw * labor_inc))
        rev_var_tot.append(rev_var_env[-1] + rev_var_inc[-1])
    else:
        rev_var_env.append(np.nan)
        rev_var_inc.append(np.nan)
        rev_var_tot.append(np.nan)

    # — Fixed τ_w scenario (opt. at ξ=0.1) —————————————————————
    opt_tz_fix, _ = maximize_welfare_fixed_w(G_value, xi, fixed_tau_w_optimal_xi01)
    _, res_fix, conv2 = solver.solve(fixed_tau_w_optimal_xi01, opt_tz_fix, G_value, r_base, d0_base)
    if conv2:
        rev_fix_env.append(opt_tz_fix * (res_fix['z_c'] + res_fix['z_d']))
        labor_inc2 = phi * res_fix['w'] * np.maximum(T - res_fix['l_agents'], 0)
        rev_fix_inc.append(np.sum(fixed_tau_w_optimal_xi01 * labor_inc2))
        rev_fix_tot.append(rev_fix_env[-1] + rev_fix_inc[-1])
    else:
        rev_fix_env.append(np.nan)
        rev_fix_inc.append(np.nan)
        rev_fix_tot.append(np.nan)

# — Plot figure 1 —————————————————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(7, 7))

# Variable τ_w lines
ax.plot(valid_xi, rev_var_tot, '-.',   linewidth=2, color='steelblue', label='Total (var. inc. tax)')
ax.plot(valid_xi, rev_var_env, '-',    linewidth=2, color='steelblue', label='Env. (var. inc. tax)')
ax.plot(valid_xi, rev_var_inc, '--',   linewidth=2, color='steelblue', label='Inc. (var. inc. tax)')

# Fixed τ_w lines
ax.plot(valid_xi, rev_fix_tot, '-.',   linewidth=2, color='tab:green', label='Total (baseline opt. fixed inc tax.)')
ax.plot(valid_xi, rev_fix_env, '-',    linewidth=2, color='tab:green', label='Env. (baseline opt. fixed inc tax.)')
ax.plot(valid_xi, rev_fix_inc, '--',   linewidth=2, color='tab:green', label='Inc. (baseline opt. fixed inc tax.)')

# Government spending line
ax.axhline(G_value, color='gray', linestyle='-', linewidth=2, label=r'Gov. spending')

ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
ax.set_ylabel('Revenue', fontsize=14)
ax.set_xlim(xi_values[0], xi_values[-1])

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.125),
    ncol=4,
    frameon=False,
    fontsize=10
)

plt.subplots_adjust(bottom=0.30, right=0.98)
plt.savefig("c_opttax/b_revenue.pdf", bbox_inches='tight')

# —————————————————————————————————————————————————————————————————————————————
# 2) Compute & plot: “preexisting τ_w” schedule
# —————————————————————————————————————————————————————————————————————————————
pre_tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
rev_pre_env, rev_pre_inc, rev_pre_tot = [], [], []

for xi in xi_values:
    opt_tz_pre, _ = maximize_welfare_fixed_w(G_value, xi, pre_tau_w)
    _, res_pre, conv3 = solver.solve(pre_tau_w, opt_tz_pre, G_value, r_base, d0_base)
    if conv3:
        rev_e = opt_tz_pre * (res_pre['z_c'] + res_pre['z_d'])
        labor_inc_pre = phi * res_pre['w'] * np.maximum(T - res_pre['l_agents'], 0)
        rev_i = np.sum(pre_tau_w * labor_inc_pre)
        rev_t = rev_e + rev_i
    else:
        rev_e = rev_i = rev_t = np.nan
    rev_pre_env.append(rev_e)
    rev_pre_inc.append(rev_i)
    rev_pre_tot.append(rev_t)

fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.plot(xi_values, rev_pre_tot, '-.', linewidth=2, color='tab:orange', label='Total (pre-existing inc. tax)')
ax2.plot(xi_values, rev_pre_env,  '-', linewidth=2, color='tab:orange', label='Env. (pre-existing inc. tax)')
ax2.plot(xi_values, rev_pre_inc,  '--', linewidth=2, color='tab:orange', label='Inc. (pre-existing inc. tax)')

ax2.axhline(G_value, color='gray', linestyle='-', linewidth=2, label=r'Gov. spending')

ax2.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax2.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
ax2.set_ylabel('Revenue', fontsize=14)
ax2.set_xlim(xi_values[0], xi_values[-1])

ax2.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.125),
    ncol=3,
    frameon=False,
    fontsize=10
)

plt.subplots_adjust(bottom=0.30, right=0.98)
plt.savefig("e_appendix/a_revenue_preexisting.pdf", bbox_inches='tight')