import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import make_interp_spline

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a_inner_solver import solve, phi, t, n, tau_w

tau_w_init = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_opt  = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
g = 5.0
tau_z_values = np.linspace(5, 10.0, 50)
tau_z_baseline = 0.1
xi_values = np.linspace(0.1, 4, 4)  # Four values in (1,4)
theta = 1  # as given

# For Tax System 1
penalty_sys1 = {}
for xi in xi_values:
    penalty_curve = np.zeros(len(tau_z_values))
    for j, tau_z in enumerate(tau_z_values):
        sol, res, conv = solve(tau_w_init, tau_z, g)
        if not conv:
            sol_base, res_base, conv_base = solve(tau_w_init, tau_z_baseline, g, xi=xi)
            agg_polluting = res_base['z_c'] + res_base['z_d']
        else:
            agg_polluting = res['z_c'] + res['z_d']
        penalty_curve[j] = 5 * xi * (agg_polluting ** theta)
    penalty_sys1[xi] = penalty_curve

# For Tax System 2
penalty_sys2 = {}
for xi in xi_values:
    penalty_curve = np.zeros(len(tau_z_values))
    for j, tau_z in enumerate(tau_z_values):
        sol, res, conv = solve(tau_w_opt, tau_z, g)
        if not conv:
            sol_base, res_base, conv_base = solve(tau_w_opt, tau_z_baseline, g)
            agg_polluting = res_base['z_c'] + res_base['z_d']
        else:
            agg_polluting = res['z_c'] + res['z_d']
        penalty_curve[j] = 5 * xi * (agg_polluting ** theta)
    penalty_sys2[xi] = penalty_curve

# Smoothing for Tax System 1
tau_z_smooth = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
penalty_sys1_smooth = {}
for xi in xi_values:
    spline = make_interp_spline(tau_z_values, penalty_sys1[xi])
    penalty_sys1_smooth[xi] = spline(tau_z_smooth)

# Smoothing for Tax System 2
penalty_sys2_smooth = {}
for xi in xi_values:
    spline = make_interp_spline(tau_z_values, penalty_sys2[xi])
    penalty_sys2_smooth[xi] = spline(tau_z_smooth)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for idx, xi in enumerate(xi_values):
    ax1.plot(tau_z_smooth, penalty_sys1_smooth[xi], color=colors[idx],
             linewidth=2, label=f'ξ = {xi:.2f}')
    ax2.plot(tau_z_smooth, penalty_sys2_smooth[xi], color=colors[idx],
             linewidth=2, label=f'ξ = {xi:.2f}')

ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
ax1.set_xlabel(r'$\tau_z$', fontsize=12)
ax1.set_ylabel(r'Penalty $= 5\,ξ\,(agg\_polluting)$', fontsize=12)
ax1.set_title('Tax System 1', fontsize=14)
ax2.set_xlabel(r'$\tau_z$', fontsize=12)
ax2.set_title('Tax System 2', fontsize=14)
ax1.legend(fontsize=12, loc='best')
ax2.legend(fontsize=12, loc='best')
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.savefig("b_dynamics/x_greensw.pdf")
plt.show()