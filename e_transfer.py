import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":   ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

import a_inner_solver as solver
from a_inner_solver import n, d0

G_value    = 5.0
r_base     = -1.0
d0_base    = d0

tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

tau_w_optimal = np.array([
   -1.12963781, -0.06584074,  0.20438030,
    0.38336986,  0.63241591
])

tau_z_grid = np.linspace(1.0, 20.0, 50)

def compute_l(tau_w_vec):
    l_vals = []
    for tz in tau_z_grid:
        _, res, conv = solver.solve(tau_w_vec, tz, G_value, r_base, d0_base)
        l_vals.append(res['l'] if conv else np.nan)
    return np.array(l_vals)

l_pre = compute_l(tau_w_preexisting)
l_opt = compute_l(tau_w_optimal)

fig, ax = plt.subplots(figsize=(5, 4))

light_blue = "#7aabd4"

ax.plot(
    tau_z_grid, l_pre,
    color=light_blue, linestyle='-', linewidth=2,
    label='Pre-existing income tax system'
)
ax.plot(
    tau_z_grid, l_opt,
    color=light_blue, linestyle='--', linewidth=2,
    label='Baseline optimal income tax system'
)

ax.set_xlim(tau_z_grid.min(), tau_z_grid.max())
ax.set_ylim(0.25, 2.00)
ax.set_xlabel(r'Environmental tax ($\tau_z$)', fontsize=12)
ax.set_ylabel(r'Lump‐sum transfer per household ($l$)', fontsize=12)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig("x_figs/fig3_transfer.pdf", bbox_inches='tight')