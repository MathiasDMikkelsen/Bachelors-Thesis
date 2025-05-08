import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set project root so that inner_solver can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_inner_solver import solve

# a. Calibration
alpha = 0.7
beta  = 0.2
gamma = 0.2
t     = 24.0     
d0    = 0.5
p_c   = 1.0      
phi   = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n     = len(phi)

# b. Initial policy vector and government parameter
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g     = 5.0

# c. Baseline equilibrium (choose a baseline τ_z)
tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge.")

w_base        = res_base["w"]
p_d_base      = res_base["p_d"]
l_base        = res_base["l"]
l_agents_base = res_base["l_agents"]
c_agents_base = res_base["c_agents"]
d_agents_base = res_base["d_agents"]

# Retrieve baseline utilities from the inner solver (for utility loss)
baseline_util = res_base["utilities"]

# Extract z_c and z_d from baseline output if available; otherwise use defaults.
if "z_c" in res_base and "z_d" in res_base:
    z_c = res_base["z_c"]
    z_d = res_base["z_d"]

# Also compute U_base (dual formulation) for expenditure calculations.
U_base = np.zeros(n)
for i in range(n):
    U_base[i] = (alpha * np.log(c_agents_base[i]) +
                 beta  * np.log(d_agents_base[i] - d0) +
                 gamma * np.log(l_agents_base[i]))

# d. Expenditure function (dual demand)
A = alpha + beta + gamma
def E_star(p_d, U_target):
    return (A/alpha) * np.exp((U_target - beta*np.log(beta/(alpha*p_d))
                                - gamma*np.log(gamma/alpha)) / A) + p_d*d0

# e. Compute baseline expenditure and income:
baseline_exp = np.zeros(n)
baseline_income = np.zeros(n)
for i in range(n):
    baseline_exp[i] = E_star(p_d_base, U_base[i])
    baseline_income[i] = (phi[i] * w_base *
                          (1 - tau_w[i]) * (t - l_agents_base[i]) + l_base)
# At baseline, by duality, baseline_exp equals baseline_income so TV = 0.

# f. Set up arrays for TV, Utility Loss, and SWF (using same tau_z grid)
tau_z_values = np.linspace(0.1, 2.0, 50)
TV_array     = np.zeros((n, len(tau_z_values)))
abs_UL_array = np.zeros((n, len(tau_z_values)))  # Absolute utility loss: baseline_util - U_new

# Prepare SWF arrays for two sigma values.
swf_sigma_05 = np.zeros(len(tau_z_values))
swf_sigma_15 = np.zeros(len(tau_z_values))
xi = 0.1

# Loop over policy values
for j, tau_z in enumerate(tau_z_values):
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge for τ_z = {tau_z:.2f}")
    p_d_new      = res_new["p_d"]
    w_new        = res_new["w"]
    l_new        = res_new["l"]
    l_agents_new = res_new["l_agents"]
    U_new        = res_new["utilities"]  # utilities from inner solver

    # Compute SWF for sigma = 0.5 and sigma = 1.5 using the inner solver utilities:
    swf_sigma_05[j] = np.sum(U_new) - 5*xi*(z_c+z_d)
    swf_sigma_15[j] = np.sum(U_new.min()) - 5*xi*(z_c+z_d)
    
    for i in range(n):
        E_new_val = E_star(p_d_new, U_base[i])
        income_new = (phi[i] * w_new *
                      (1 - tau_w[i]) * (t - l_agents_new[i]) + l_new)
        # Total Variation: change in required expenditure + change in income.
        TV_array[i, j] = (E_new_val - baseline_exp[i]) - (income_new - baseline_income[i])
        # Utility loss: baseline utility (from inner solver) minus new utility.
        abs_UL_array[i, j] = baseline_util[i] - U_new[i]

# Compute relative TV (as fraction of baseline income)
relative_TV_array = np.zeros_like(TV_array)
for i in range(n):
    relative_TV_array[i, :] = TV_array[i, :] / baseline_income[i]

# Compute relative utility loss (as fraction of baseline utility)
rel_UL_array = np.zeros_like(abs_UL_array)
for i in range(n):
    rel_UL_array[i, :] = abs_UL_array[i, :] / baseline_util[i]

# --- Combined Figure: 6 subplots in one figure (2 rows x 3 columns) ---
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Subplot 1: Absolute TV
for i in range(n):
    axs[0, 0].plot(tau_z_values, TV_array[i, :], label=f'Household {i+1}', linestyle='-')
axs[0, 0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 0].set_ylabel('Absolute TV', fontsize=12)
axs[0, 0].set_title('Absolute Total Variation', fontsize=14)
axs[0, 0].legend()

# Subplot 2: Relative TV (TV / baseline income)
for i in range(n):
    axs[0, 1].plot(tau_z_values, relative_TV_array[i, :], label=f'Household {i+1}', linestyle='-')
axs[0, 1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 1].set_ylabel('Relative TV', fontsize=12)
axs[0, 1].set_title('TV as Fraction of Baseline Income', fontsize=14)
axs[0, 1].legend()

# Subplot 3: Absolute Utility Loss
for i in range(n):
    axs[0, 2].plot(tau_z_values, abs_UL_array[i, :], label=f'Household {i+1}', linestyle='-')
axs[0, 2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 2].set_ylabel('Absolute Utility Loss', fontsize=12)
axs[0, 2].set_title('Absolute Utility Loss', fontsize=14)
axs[0, 2].legend()

# Subplot 4: Relative Utility Loss (loss / baseline utility)
for i in range(n):
    axs[1, 0].plot(tau_z_values, rel_UL_array[i, :], label=f'Household {i+1}', linestyle='-')
axs[1, 0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 0].set_ylabel('Relative Utility Loss', fontsize=12)
axs[1, 0].set_title('Utility Loss as Fraction of Baseline Utility', fontsize=14)
axs[1, 0].legend()

# Subplot 5: SWF for sigma = 1.0
axs[1, 1].plot(tau_z_values, swf_sigma_05, label='SWF, σ = 1.0', linewidth=2, color='blue')
axs[1, 1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 1].set_ylabel('SWF', fontsize=12)
axs[1, 1].set_title('SWF for σ = 0.5', fontsize=14)
axs[1, 1].legend()

# Subplot 6: SWF for sigma = 1.5
axs[1, 2].plot(tau_z_values, swf_sigma_15, label='SWF, σ = 1.5', linewidth=2, color='red')
axs[1, 2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 2].set_ylabel('SWF', fontsize=12)
axs[1, 2].set_title('SWF for σ = 1.5', fontsize=14)
axs[1, 2].legend()

plt.tight_layout()
plt.savefig("b_dynamics/e_tv.pdf")
plt.show()