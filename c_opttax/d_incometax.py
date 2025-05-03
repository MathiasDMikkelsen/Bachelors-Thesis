# plot_xi_sensitivity.py (Final Corrected Version with Legend Box)

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX styling (match other figures)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Project setup: ensure a_solvers on path
# ─────────────────────────────────────────────────────────────────────────────
project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_solvers_dir = os.path.join(project_root, "a_solvers")
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────────────
# Imports from updated solvers
# ─────────────────────────────────────────────────────────────────────────────
from a_solvers import outer_solver
from a_solvers.inner_solver import n, d0 as d0_base

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
G_value    = 5.0          # government spending
r_base     = -1.0         # CES exponent (sigma = 1/(1-r))
d0_base    = d0_base      # baseline durables demand
xi_values  = np.linspace(0.1, 1.0, 20)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline optimization at first xi
# ─────────────────────────────────────────────────────────────────────────────
print(f"Calculating baseline taxes at xi = {xi_values[0]:.4f}")
baseline_tau_w, _, _ = outer_solver.maximize_welfare(
    G_value, xi_values[0], r_base, d0_base
)
if baseline_tau_w is None:
    raise ValueError(f"Baseline optimization failed at xi = {xi_values[0]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Run optimization across xi values
# ─────────────────────────────────────────────────────────────────────────────
optimal_tau_w_results = []
valid_xi_values       = []

print("Starting optimizations across xi values...")
for xi in xi_values:
    print(f"  Optimizing at xi = {xi:.4f}")
    opt_tau_w, _, _ = outer_solver.maximize_welfare(
        G_value, xi, r_base, d0_base
    )
    if opt_tau_w is not None:
        optimal_tau_w_results.append(opt_tau_w)
        valid_xi_values.append(xi)
    else:
        print(f"    Optimization failed at xi = {xi:.4f}, skipping...")

optimal_tau_w_results = np.array(optimal_tau_w_results)
valid_xi_values       = np.array(valid_xi_values)

# ─────────────────────────────────────────────────────────────────────────────
# Percentage-point changes from baseline
# ─────────────────────────────────────────────────────────────────────────────
pct_diff_from_baseline = (optimal_tau_w_results - baseline_tau_w) * 100

# ─────────────────────────────────────────────────────────────────────────────
# Colormap setup
# ─────────────────────────────────────────────────────────────────────────────
blue_cmap = plt.cm.Blues
colors    = [blue_cmap(0.3 + 0.5 * i / (n - 1)) for i in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for i in range(n):
    ax.plot(
        valid_xi_values,
        pct_diff_from_baseline[:, i],
        linestyle='-', linewidth=2, color=colors[i],
        label=rf'Household {i+1}'
    )

ax.set_xlim(xi_values[0], xi_values[-1])
ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=12)
ax.set_ylabel(r'Change in income tax rate (\%-points)', fontsize=12)
ax.legend(loc='best', fancybox=True, framealpha=0.8)
ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig("c_opttax/d_incometax.pdf", bbox_inches='tight')
