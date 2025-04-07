import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set up the project root so that a_solvers can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve, phi, t, n, tau_w

tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g = 5.0
tau_z_values = np.linspace(0.1, 25.0, 50)

tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge at tau_z = 0.1")

w_base = res_base["w"]
l_base = res_base["l"]
l_agents_base = res_base["l_agents"]

income_baseline = np.zeros(n)
for i in range(n):
    income_baseline[i] = phi[i] * w_base * (1 - tau_w[i]) * (t - l_agents_base[i]) + l_base

income_new_const = np.zeros((n, len(tau_z_values)))
income_new_update = np.zeros((n, len(tau_z_values)))

# Arrays to store lump sum and leisure
lump_sum_values = np.zeros((n, len(tau_z_values)))
leisure_values = np.zeros((n, len(tau_z_values)))

for j, tau_z in enumerate(tau_z_values):
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge for tau_z = {tau_z:.2f}; using baseline income.")
        for i in range(n):
            income_new_const[i, j] = income_baseline[i]
            income_new_update[i, j] = income_baseline[i]
            lump_sum_values[i, j] = l_base  # Use baseline lump sum
            leisure_values[i, j] = l_agents_base[i]  # Use baseline leisure
    else:
        w_new = res_new["w"]
        l_new = res_new["l"]
        l_agents_new = res_new["l_agents"]
        for i in range(n):
            income_new_const[i, j] = (phi[i] * w_new * (1 - tau_w[i]) *
                                        (t - l_agents_base[i]) + l_base)
            income_new_update[i, j] = (phi[i] * w_new * (1 - tau_w[i]) *
                                       (t - l_agents_new[i]) + l_new)
            lump_sum_values[i, j] = l_new
            leisure_values[i, j] = l_agents_new[i]

absolute_drop_const = income_new_const - income_baseline.reshape(-1, 1)
absolute_drop_update = income_new_update - income_baseline.reshape(-1, 1)

# --- Plot side-by-side subplots ---
fig, axes = plt.subplots(2, 2, figsize=(8, 7))  # Smaller overall figure size
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Top Left subplot: Leisure Held Constant
ax = axes[0, 0]
for i in range(n):
    ax.plot(tau_z_values, absolute_drop_const[i, :],
            color=colors[i], linestyle='-', linewidth=2,  # Thicker lines
            label=f'$i={i+1}$')
ax.set_xlabel(r'$\tau_z$', fontsize=10)  # Larger fontsize
ax.set_ylabel(r'absolute $m^d_i+l$ drop', fontsize=10)  # Larger fontsize
ax.set_title(r'$\ell_i$ constant', fontsize=10)  # Title added
ax.tick_params(axis='both', which='major', labelsize=8)  # Larger tick labels
ax.legend(fontsize=8, loc='upper right')  # Larger legend

# Top Right subplot: Updated Leisure Outcomes
ax = axes[0, 1]
for i in range(n):
    ax.plot(tau_z_values, absolute_drop_update[i, :],
            color=colors[i], linestyle='-', linewidth=2,  # Thicker lines
            label=f'$i=${i+1}')
ax.set_xlabel(r'$\tau_z$', fontsize=10)  # Larger fontsize
ax.set_ylabel(r'absolute $m^d_i+l$ drop', fontsize=10)  # Larger fontsize
ax.set_title(r'$\ell_i$ changes', fontsize=10)  # Title added
ax.tick_params(axis='both', which='major', labelsize=8)  # Larger tick labels
ax.legend(fontsize=8, loc='upper right')  # Larger legend

# Bottom Left subplot: Lump Sum per Household
ax = axes[1, 0]
for i in range(n):
    ax.plot(tau_z_values, lump_sum_values[i, :],
            color=colors[i], linestyle='-', linewidth=2,  # Thicker lines
            label=f'$i=${i+1}')
ax.set_xlabel(r'$\tau_z$', fontsize=10)  # Larger fontsize
ax.set_ylabel(r'$l$', fontsize=10)  # Larger fontsize
ax.tick_params(axis='both', which='major', labelsize=8)  # Larger tick labels

# Bottom Right subplot: Leisure for each household
ax = axes[1, 1]
for i in range(n):
    ax.plot(tau_z_values, leisure_values[i, :],
            color=colors[i], linestyle='-', linewidth=2,  # Thicker lines
            label=f'$i=${i+1}')
ax.set_xlabel(r'$\tau_z$', fontsize=10)  # Larger fontsize
ax.set_ylabel(r'$\ell_i$', fontsize=10)  # Larger fontsize
ax.tick_params(axis='both', which='major', labelsize=8)  # Larger tick labels
ax.legend(fontsize=8, loc='upper right')  # Larger legend

plt.tight_layout()
plt.savefig("b_dynamics/c_income.pdf")
plt.show()