import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl   # only needed once, before any figures are created
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
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_solvers_dir = os.path.join(project_root, "a_solvers")
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import a_solvers.inner_solver as solver
from a_solvers.inner_solver import n

# –– PARAMETERS ––
G_value = 5.0

# 1) Pre-existing income tax system (fill this in)
tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

# 2) Optimal at baseline (ξ=0.1) from your last run
tau_w_optimal = np.array([
   -1.12963781, -0.06584074,  0.20438030,
    0.38336986,  0.63241591
])

# grid of τ_z to sweep
tau_z_grid = np.linspace(1.0, 20.0, 50)

def compute_l(tau_w_vec):
    l_vals = []
    for τz in tau_z_grid:
        sol, results, converged = solver.solve(tau_w_vec, τz, G_value)
        if not converged or 'l' not in results:
            l_vals.append(np.nan)
        else:
            l_vals.append(results['l'])
    return np.array(l_vals)

# compute both series
l_pre  = compute_l(tau_w_preexisting)
l_opt  = compute_l(tau_w_optimal)

# –– PLOTTING ––
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(tau_z_grid, l_pre, color="#7aabd4", linestyle='-', linewidth=2, label='Pre-existing income tax system')
ax.plot(tau_z_grid, l_opt, color="#7aabd4", linestyle='--', linewidth=2, label='Baseline optimal income tax system')


ax.set_xlim(tau_z_grid.min(), tau_z_grid.max())
ax.set_ylim(0.25, 2.00)
ax.set_xlabel(r'Environmental tax ($\tau_z$)', fontsize=12)
ax.set_ylabel(r'Lump‐sum transfer per household ($l$)', fontsize=12)
ax.grid(True, ls='--', lw=0.5, alpha=0.7)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("c_opttax/e_transfer.pdf", bbox_inches='tight')