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

import b_outer_solver
from a_inner_solver import n, d0 as d0_base

G_value    = 5.0
r_base     = -1.0
d0_base    = d0_base
xi_values  = np.linspace(0.1, 1.0, 20)

baseline_tau_w, _, _ = b_outer_solver.maximize_welfare(
    G_value, xi_values[0], r_base, d0_base
)
if baseline_tau_w is None:
    raise ValueError(f"Baseline optimization failed at xi = {xi_values[0]:.4f}")

optimal_tau_w_results = []
valid_xi_values       = []

for xi in xi_values:
    opt_tau_w, _, _ = b_outer_solver.maximize_welfare(
        G_value, xi, r_base, d0_base
    )
    if opt_tau_w is not None:
        optimal_tau_w_results.append(opt_tau_w)
        valid_xi_values.append(xi)

optimal_tau_w_results = np.array(optimal_tau_w_results)
valid_xi_values       = np.array(valid_xi_values)

pct_diff_from_baseline = (optimal_tau_w_results - baseline_tau_w) * 100

blue_cmap = plt.cm.Blues
colors    = [blue_cmap(0.3 + 0.5 * i / (n - 1)) for i in range(n)]

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

plt.savefig("x_figs/fig5_incometax.pdf", bbox_inches='tight')