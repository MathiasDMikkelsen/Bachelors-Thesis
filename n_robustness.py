import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

import b_outer_solver
import a_inner_solver as solver_module
from a_inner_solver import n, alpha, beta, gamma, d0, phi, t as T

G_base   = 5.0
r_base   = -1.0
d0_base  = d0

fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
xi_values = np.linspace(0.1, 1.0, 5)


def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr, r, d0):
    def swf_obj(x, G_val, xi_val, fw_arr):
        tau_z = float(x[0])
        _, results, conv = solver_module.solve(fw_arr, tau_z, G_val, r, d0)
        if not conv:
            return 1e10
        utils = results['utilities']
        agg_z = results['z_c'] + results['z_d']
        return -(np.sum(utils) - 5 * xi_val * agg_z)

    res = minimize(
        swf_obj,
        x0=[0.5],
        args=(G, xi, fixed_tau_w_arr),
        bounds=[(1e-6, 100.0)],
        method='SLSQP',
        options={'ftol':1e-7, 'disp':False}
    )
    if res.success:
        return res.x[0], -res.fun
    return None, None


def compute_tau_z_series(G, r, d0):
    tz_var, tz_pre, tz_opt = [], [], []
    xi_valid = []

    baseline_tw, _, _ = b_outer_solver.maximize_welfare(G, 0.1, r, d0)

    for xi in xi_values:
        _, tz1, conv1 = b_outer_solver.maximize_welfare(G, xi, r, d0)
        tz_var.append(tz1 if conv1 else np.nan)

        tz2, _ = maximize_welfare_fixed_w(G, xi, fixed_tau_w_preexisting, r, d0)
        tz_pre.append(tz2 if tz2 is not None else np.nan)

        tz3, _ = maximize_welfare_fixed_w(G, xi, baseline_tw, r, d0)
        tz_opt.append(tz3 if tz3 is not None else np.nan)

        xi_valid.append(xi)

    return np.array(xi_valid), np.array(tz_var), np.array(tz_pre), np.array(tz_opt)


color_var = 'steelblue'
color_pre = 'tab:orange'
color_opt = 'tab:green'

g_values = [1.5, 3.0, 7.0, 9.5]
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, G in zip(axes1.flat, g_values):
    xi, tz1, tz2, tz3 = compute_tau_z_series(G, r_base, d0_base)
    ax.plot(xi, tz1, '-', color=color_var, lw=2, label='Variable inc. tax')
    ax.plot(xi, tz2, '--', color=color_pre, lw=2, label='Fixed inc. tax (pre-existing)')
    ax.plot(xi, tz3, ':', color=color_opt, lw=2, label='Fixed inc. tax (optimal at $\\xi=0.1$)')
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.set_title(rf'$\mathcal{{G}} = {G}$', fontsize=16)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.savefig('x_figs/fig11_robustness_g.pdf', bbox_inches='tight')

d0_values = [0.4, 0.45, 0.55, 0.6]
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, d0_val in zip(axes2.flat, d0_values):
    xi, tz1, tz2, tz3 = compute_tau_z_series(G_base, r_base, d0_val)
    ax.plot(xi, tz1, '-', color=color_var, lw=2, label='Variable inc. tax')
    ax.plot(xi, tz2, '--', color=color_pre, lw=2, label='Fixed inc. tax (pre-existing)')
    ax.plot(xi, tz3, ':', color=color_opt, lw=2, label='Fixed inc. tax (optimal at $\\xi=0.1$)')
    ax.set_title(f'$d_0 = {d0_val}$', fontsize=16)
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.savefig('x_figs/fig12_robustness_d0.pdf', bbox_inches='tight')

sigma_values = [0.35, 0.4, 0.6, 0.65]
fig3, axes3 = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, sigma in zip(axes3.flat, sigma_values):
    r_val = 1.0 - 1.0/sigma
    xi, tz1, tz2, tz3 = compute_tau_z_series(G_base, r_val, d0_base)
    ax.plot(xi, tz1, '-', color=color_var, lw=2, label='Variable inc. tax')
    ax.plot(xi, tz2, '--', color=color_pre, lw=2, label='Fixed inc. tax (pre-existing)')
    ax.plot(xi, tz3, ':', color=color_opt, lw=2, label='Fixed inc. tax (optimal at $\\xi=0.1$)')
    ax.set_title(f'$\sigma = {sigma}$', fontsize=16)
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.savefig('x_figs/fig13_robustness_sigma.pdf', bbox_inches='tight')

print('robustness plots generated.')