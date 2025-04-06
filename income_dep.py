import numpy as np
import matplotlib.pyplot as plt
from inner_solver import solve, phi, t, n, tau_w

tau_w = np.array([0] * n)
g = 0.0
tau_z_values = np.linspace(0.1, 3.0, 50)

# --- Baseline Equilibrium (tau_z_baseline = 0.1) ---
tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge at tau_z = 0.1")

w_base = res_base["w"]
l_base = res_base["l"]
l_agents_base = res_base["l_agents"]

# Baseline disposable income for each household:
# m_base = phi[i] * w_base * (1-tau_w[i]) * (t - l_agents_base[i]) + l_base
income_baseline = np.zeros(n)
for i in range(n):
    income_baseline[i] = phi[i] * w_base * (1 - tau_w[i]) * (t - l_agents_base[i]) + l_base

# Initialize arrays to store new disposable income:
# (1) Holding leisure constant (using baseline leisure)
# (2) Using updated leisure outcomes
income_new_const = np.zeros((n, len(tau_z_values)))
income_new_update = np.zeros((n, len(tau_z_values)))

for j, tau_z in enumerate(tau_z_values):
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge for tau_z = {tau_z:.2f}; using baseline income.")
        for i in range(n):
            income_new_const[i, j] = income_baseline[i]
            income_new_update[i, j] = income_baseline[i]
    else:
        w_new = res_new["w"]
        l_new = res_new["l"]
        l_agents_new = res_new["l_agents"]
        for i in range(n):
            # (1) Holding leisure constant: use baseline leisure (l_agents_base, l_base)
            income_new_const[i, j] = (phi[i] * w_new * (1 - tau_w[i]) *
                                        (t - l_agents_base[i]) + l_base)
            # (2) Using updated leisure outcomes:
            income_new_update[i, j] = (phi[i] * w_new * (1 - tau_w[i]) *
                                       (t - l_agents_new[i]) + l_new)

# Compute relative drop in disposable income: (m_base - m_new) / m_base.
relative_drop_const = (income_baseline.reshape(-1, 1) - income_new_const) / income_baseline.reshape(-1, 1)
relative_drop_update = (income_baseline.reshape(-1, 1) - income_new_update) / income_baseline.reshape(-1, 1)

# --- Plot side-by-side subplots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Left subplot: Leisure Held Constant
ax = axes[0]
for i in range(n):
    ax.plot(tau_z_values, relative_drop_const[i, :],
            color=colors[i], linestyle='-', linewidth=2,
            label=f'Household {i+1}')
ax.set_xlabel(r'$\tau_z$', fontsize=14)
ax.set_ylabel('Relative Income Drop', fontsize=14)
ax.set_title('Leisure Held Constant', fontsize=16)
ax.legend(fontsize=10)

# Right subplot: Updated Leisure Outcomes
ax = axes[1]
for i in range(n):
    ax.plot(tau_z_values, relative_drop_update[i, :],
            color=colors[i], linestyle='-', linewidth=2,
            label=f'Household {i+1}')
ax.set_xlabel(r'$\tau_z$', fontsize=14)
ax.set_title('Updated Leisure Outcomes', fontsize=16)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("relative_income_drop_side_by_side.pdf")
plt.show()