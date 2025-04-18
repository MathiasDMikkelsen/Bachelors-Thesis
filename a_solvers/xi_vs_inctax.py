# plot_xi_sensitivity.py (Final Corrected Version with Legend Box)

import numpy as np
import matplotlib.pyplot as plt
import outer_solver
from inner_solver import n
import os

# --- Simulation Parameters ---
G_value = 5.0
xi_values = np.linspace(0.04, 1.0, 25)

# --- Baseline Calculation ---
print(f"Calculating baseline taxes at xi = {xi_values[0]:.4f}")
baseline_tau_w, _, baseline_success = outer_solver.maximize_welfare(G_value, xi_values[0])
if baseline_tau_w is None:
    raise ValueError(f"Baseline optimization failed at xi = {xi_values[0]:.4f}")

# --- Run Optimization for each xi ---
optimal_tau_w_results = []
valid_xi_values = []

print("Starting optimizations across xi values...")

for xi in xi_values:
    print(f"  Optimizing at xi = {xi:.4f}")
    opt_tau_w, _, success = outer_solver.maximize_welfare(G_value, xi)
    if opt_tau_w is not None:
        optimal_tau_w_results.append(opt_tau_w)
        valid_xi_values.append(xi)
    else:
        print(f"    Optimization failed at xi = {xi:.4f}, skipping...")

optimal_tau_w_results = np.array(optimal_tau_w_results)
valid_xi_values = np.array(valid_xi_values)

# --- Percentage-point difference from baseline ---
pct_diff_from_baseline = optimal_tau_w_results - baseline_tau_w

# --- Colormap Setup ---
blue_cmap = plt.cm.Blues
colors = [blue_cmap(0.3 + 0.5 * i / (n - 1)) for i in range(n)]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(n):
    ax.plot(valid_xi_values, pct_diff_from_baseline[:, i],
            linestyle='-', linewidth=2, color=colors[i], label=rf'$\Delta \tau_w^{{{i+1}}} \cdot 100$')

ax.set_xlabel(r'$\xi$', fontsize=12)
ax.set_ylabel('Pct. points', fontsize=12)
ax.legend(loc='best', fancybox=True, framealpha=0.8)
ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

# --- Save Plot ---
output_dir = "c_opttax"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "d_tauw.pdf")
plt.tight_layout()
plt.savefig(output_path)
print(f"\nPlot saved successfully to {output_path}")

plt.show()