import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

import k_outer_solver_ext as outer_solver
import j_inner_solver_ext as solver
from j_inner_solver_ext import n, alpha, beta, gamma, d0, phi, varphi

G_value   = 5.0
theta     = 1.0
xi_values = np.linspace(0.1, 1.0, 10)

fixed_tau_w_preexisting  = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.22559844, -0.11142818, 0.17570669, 0.36519415, 0.62727242])

p_a_list      = [1.0, 3.0, 6.0, 8.0]
varsigma_list = [1.5, 1.75, 2.25, 2.5]

def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr, p_a, varsigma):
    def obj(z, G_val, xi_val, fw, pa, vs):
        tz = float(z[0])
        _, res, conv = solver.solve(fw, tz, G_val, pa, vs)
        if not conv:
            return 1e10
        u    = res['utilities']
        aggz = res['z_c'] + res['z_d']
        return -(np.sum(u) - 5 * xi_val * aggz)

    result = minimize(
        obj,
        x0=[0.5],
        args=(G, xi, fixed_tau_w_arr, p_a, varsigma),
        bounds=[(1e-6, 100.0)],
        method='SLSQP',
        options={'ftol':1e-7, 'disp':False}
    )
    return (result.x[0], -result.fun) if result.success else (None, None)

fig, axes = plt.subplots(2, 2, figsize=(10,7.5))

for ax, pa in zip(axes.flat, p_a_list):
    tz_var, tz_pre, tz_fix = [], [], []

    baseline_tw, _, _ = outer_solver.maximize_welfare(G_value, 0.1, pa, 2.0)

    for xi in xi_values:
        _, tz, _ = outer_solver.maximize_welfare(G_value, xi, pa, 2.0)
        tz_var.append(tz if tz is not None else np.nan)

    for xi in xi_values:
        tz, _ = maximize_welfare_fixed_w(G_value, xi, fixed_tau_w_preexisting, pa, 2.0)
        tz_pre.append(tz if tz is not None else np.nan)

    for xi in xi_values:
        tz, _ = maximize_welfare_fixed_w(G_value, xi, baseline_tw, pa, 2.0)
        tz_fix.append(tz if tz is not None else np.nan)

    ax.plot(xi_values, tz_var, '-',  linewidth=2, color='steelblue',
            label='Variable inc. tax')
    ax.plot(xi_values, tz_pre, '--', linewidth=2, color='tab:orange',
            label='Fixed inc. tax (pre-existing)')
    ax.plot(xi_values, tz_fix, ':',  linewidth=2, color='tab:green',
            label='Fixed inc. tax (opt. at $\\xi=0.1$)')

    ax.set_xlim(xi_values[0], xi_values[-1])
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.set_title(f'$p_a={pa}$', fontsize=16)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig("x_figs/fig9_robustness_p_a.pdf", bbox_inches='tight')

xi_values = np.linspace(0.1, 1.0, 5)

fig2, axes2 = plt.subplots(2, 2, figsize=(10,7.5))

for ax, vs in zip(axes2.flat, varsigma_list):
    tz_var, tz_pre, tz_fix = [], [], []

    baseline_tw, _, _ = outer_solver.maximize_welfare(G_value, 0.1, 5.0, vs)

    for xi in xi_values:
        _, tz, _ = outer_solver.maximize_welfare(G_value, xi, 5.0, vs)
        tz_var.append(tz if tz is not None else np.nan)
    for xi in xi_values:
        tz, _ = maximize_welfare_fixed_w(G_value, xi, fixed_tau_w_preexisting, 5.0, vs)
        tz_pre.append(tz if tz is not None else np.nan)
    for xi in xi_values:
        tz, _ = maximize_welfare_fixed_w(G_value, xi, baseline_tw, 5.0, vs)
        tz_fix.append(tz if tz is not None else np.nan)

    ax.plot(xi_values, tz_var, '-',  linewidth=2, color='steelblue',
            label='Variable inc. tax')
    ax.plot(xi_values, tz_pre, '--', linewidth=2, color='tab:orange',
            label='Fixed inc. tax (pre-existing)')
    ax.plot(xi_values, tz_fix, ':',  linewidth=2, color='tab:green',
            label='Fixed inc. tax (opt. at $\\xi=0.1$)')

    ax.set_xlim(xi_values[0], xi_values[-1])
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.set_title(f'$\\varsigma={vs}$', fontsize=16)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig("x_figs/fig14_robustness_varsigma.pdf", bbox_inches='tight')
