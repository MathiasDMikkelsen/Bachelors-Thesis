import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set project root so that inner_solver can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

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
# Original income tax combo:
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
# New (alternative) income tax combo:
tau_w_alt = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
g     = 5.0

# c. Baseline equilibrium for original tax combo (choose a baseline τ_z)
tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge (original combo).")

w_base        = res_base["w"]
p_d_base      = res_base["p_d"]
l_base        = res_base["l"]
l_agents_base = res_base["l_agents"]
c_agents_base = res_base["c_agents"]
d_agents_base = res_base["d_agents"]

# Retrieve baseline utilities from the inner solver (for utility loss)
baseline_util = res_base["utilities"]

# Extract z_c and z_d from baseline output if available (used in SWF calculations)
if "z_c" in res_base and "z_d" in res_base:
    z_c = res_base["z_c"]
    z_d = res_base["z_d"]

# Compute U_base (dual formulation) for expenditure calculations for original combo.
U_base = np.zeros(n)
for i in range(n):
    U_base[i] = (alpha * np.log(c_agents_base[i]) +
                 beta  * np.log(d_agents_base[i] - d0) +
                 gamma * np.log(l_agents_base[i]))

# d. Compute baseline expenditure and income for original combo:
def E_star(p_d, U_target):
    A = alpha + beta + gamma
    return (A/alpha) * np.exp((U_target - beta*np.log(beta/(alpha*p_d))
                                - gamma*np.log(gamma/alpha)) / A) + p_d*d0

baseline_exp = np.zeros(n)
baseline_income = np.zeros(n)
for i in range(n):
    baseline_exp[i] = E_star(p_d_base, U_base[i])
    baseline_income[i] = (phi[i] * w_base *
                          (1 - tau_w[i]) * (t - l_agents_base[i]) + l_base)
# At baseline, by duality, baseline_exp equals baseline_income so TV = 0.

# d-alt. Baseline equilibrium for the alternative tax combo:
sol_base_alt, res_base_alt, conv_base_alt = solve(tau_w_alt, tau_z_baseline, g)
if not conv_base_alt:
    print("Baseline model did not converge (alternative combo).")

w_base_alt        = res_base_alt["w"]
p_d_base_alt      = res_base_alt["p_d"]
l_base_alt        = res_base_alt["l"]
l_agents_base_alt = res_base_alt["l_agents"]
c_agents_base_alt = res_base_alt["c_agents"]
d_agents_base_alt = res_base_alt["d_agents"]
baseline_util_alt = res_base_alt["utilities"]

# Compute U_base for alternative combo.
U_base_alt = np.zeros(n)
for i in range(n):
    U_base_alt[i] = (alpha * np.log(c_agents_base_alt[i]) +
                     beta  * np.log(d_agents_base_alt[i] - d0) +
                     gamma * np.log(l_agents_base_alt[i]))

baseline_exp_alt = np.zeros(n)
baseline_income_alt = np.zeros(n)
for i in range(n):
    baseline_exp_alt[i] = E_star(p_d_base_alt, U_base_alt[i])
    baseline_income_alt[i] = (phi[i] * w_base_alt *
                              (1 - tau_w_alt[i]) * (t - l_agents_base_alt[i]) + l_base_alt)

# e. Prepare arrays for TV, Utility Loss, and SWF (using same tau_z grid)
tau_z_values = np.linspace(0.1, 20.0, 50)
TV_array     = np.zeros((n, len(tau_z_values)))
abs_UL_array = np.zeros((n, len(tau_z_values)))
TV_array_alt     = np.zeros((n, len(tau_z_values)))
abs_UL_array_alt = np.zeros((n, len(tau_z_values)))

# Prepare SWF arrays for two sigma values for both combos.
swf_sigma_05 = np.zeros(len(tau_z_values))
swf_sigma_15 = np.zeros(len(tau_z_values))
swf_sigma_05_alt = np.zeros(len(tau_z_values))
swf_sigma_15_alt = np.zeros(len(tau_z_values))
xi = 0.1

# f. Loop over policy values (varying tau_z) and compute outcome metrics for both tax combos
for j, tau_z in enumerate(tau_z_values):
    # --- Original income tax combo simulation ---
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge (original) for τ_z = {tau_z:.2f}")
    p_d_new      = res_new["p_d"]
    w_new        = res_new["w"]
    l_new        = res_new["l"]
    l_agents_new = res_new["l_agents"]
    U_new        = res_new["utilities"]

    swf_sigma_05[j] = np.sum(U_new) - 5*xi*(z_c+z_d)
    swf_sigma_15[j] = np.sum(U_new.min()) - 5*xi*(z_c+z_d)
    
    for i in range(n):
        E_new_val = E_star(p_d_new, U_base[i])
        income_new = (phi[i] * w_new *
                      (1 - tau_w[i]) * (t - l_agents_new[i]) + l_new)
        # Total Variation (TV): change in required expenditure + change in income.
        TV_array[i, j] = (E_new_val - baseline_exp[i]) - (income_new - baseline_income[i])
        # Absolute utility loss: baseline utility minus new utility.
        abs_UL_array[i, j] = baseline_util[i] - U_new[i]
        
    # --- Alternative income tax combo simulation ---
    sol_new_alt, res_new_alt, conv_new_alt = solve(tau_w_alt, tau_z, g)
    if not conv_new_alt:
        print(f"Model did not converge (alternative) for τ_z = {tau_z:.2f}")
    p_d_new_alt      = res_new_alt["p_d"]
    w_new_alt        = res_new_alt["w"]
    l_new_alt        = res_new_alt["l"]
    l_agents_new_alt = res_new_alt["l_agents"]
    U_new_alt        = res_new_alt["utilities"]

    swf_sigma_05_alt[j] = np.sum(U_new_alt) - 5*xi*(z_c+z_d)
    swf_sigma_15_alt[j] = np.sum(U_new_alt.min()) - 5*xi*(z_c+z_d)
    
    for i in range(n):
        E_new_val_alt = E_star(p_d_new_alt, U_base_alt[i])
        income_new_alt = (phi[i] * w_new_alt *
                          (1 - tau_w_alt[i]) * (t - l_agents_new_alt[i]) + l_new_alt)
        TV_array_alt[i, j] = (E_new_val_alt - baseline_exp_alt[i]) - (income_new_alt - baseline_income_alt[i])
        abs_UL_array_alt[i, j] = baseline_util_alt[i] - U_new_alt[i]

# Compute relative TV (as fraction of baseline income)
relative_TV_array = np.zeros_like(TV_array)
relative_TV_array_alt = np.zeros_like(TV_array_alt)
for i in range(n):
    relative_TV_array[i, :] = TV_array[i, :] / baseline_income[i]
    relative_TV_array_alt[i, :] = TV_array_alt[i, :] / baseline_income_alt[i]

# Compute relative utility loss (as fraction of baseline utility)
rel_UL_array = np.zeros_like(abs_UL_array)
rel_UL_array_alt = np.zeros_like(abs_UL_array_alt)
for i in range(n):
    rel_UL_array[i, :] = abs_UL_array[i, :] / baseline_util[i]
    rel_UL_array_alt[i, :] = abs_UL_array_alt[i, :] / baseline_util_alt[i]

# --- Combined Figure: 6 subplots in one figure (2 rows x 3 columns) ---
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Subplot 1: Absolute TV
for i in range(n):
    axs[0, 0].plot(tau_z_values, TV_array[i, :], label=f'HH {i+1} (orig)', linestyle='-')
    axs[0, 0].plot(tau_z_values, TV_array_alt[i, :], label=f'HH {i+1} (alt)', linestyle='--')
axs[0, 0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 0].set_ylabel('Absolute TV', fontsize=12)
axs[0, 0].set_title('Absolute Total Variation', fontsize=14)
axs[0, 0].legend()

# Subplot 2: Relative TV (TV / baseline income)
for i in range(n):
    axs[0, 1].plot(tau_z_values, relative_TV_array[i, :], label=f'HH {i+1} (orig)', linestyle='-')
    axs[0, 1].plot(tau_z_values, relative_TV_array_alt[i, :], label=f'HH {i+1} (alt)', linestyle='--')
axs[0, 1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 1].set_ylabel('Relative TV', fontsize=12)
axs[0, 1].set_title('TV as Fraction of Baseline Income', fontsize=14)
axs[0, 1].legend()

# Subplot 3: Absolute Utility Loss
for i in range(n):
    axs[0, 2].plot(tau_z_values, abs_UL_array[i, :], label=f'HH {i+1} (orig)', linestyle='-')
    axs[0, 2].plot(tau_z_values, abs_UL_array_alt[i, :], label=f'HH {i+1} (alt)', linestyle='--')
axs[0, 2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0, 2].set_ylabel('Absolute Utility Loss', fontsize=12)
axs[0, 2].set_title('Absolute Utility Loss', fontsize=14)
axs[0, 2].legend()

# Subplot 4: Relative Utility Loss (loss / baseline utility)
for i in range(n):
    axs[1, 0].plot(tau_z_values, rel_UL_array[i, :], label=f'HH {i+1} (orig)', linestyle='-')
    axs[1, 0].plot(tau_z_values, rel_UL_array_alt[i, :], label=f'HH {i+1} (alt)', linestyle='--')
axs[1, 0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 0].set_ylabel('Relative Utility Loss', fontsize=12)
axs[1, 0].set_title('Utility Loss as Fraction of Baseline Utility', fontsize=14)
axs[1, 0].legend()

# Subplot 5: SWF for sigma = 0.5
axs[1, 1].plot(tau_z_values, swf_sigma_05, label='SWF, σ = 0.5 (orig)', linewidth=2, color='blue')
axs[1, 1].plot(tau_z_values, swf_sigma_05_alt, label='SWF, σ = 0.5 (alt)', linewidth=2, linestyle='--', color='blue')
axs[1, 1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 1].set_ylabel('SWF', fontsize=12)
axs[1, 1].set_title('SWF for σ = 0.5', fontsize=14)
axs[1, 1].legend()

# Subplot 6: SWF for sigma = 1.5
axs[1, 2].plot(tau_z_values, swf_sigma_15, label='SWF, σ = 1.5 (orig)', linewidth=2, color='red')
axs[1, 2].plot(tau_z_values, swf_sigma_15_alt, label='SWF, σ = 1.5 (alt)', linewidth=2, linestyle='--', color='red')
axs[1, 2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1, 2].set_ylabel('SWF', fontsize=12)
axs[1, 2].set_title('SWF for σ = 1.5', fontsize=14)
axs[1, 2].legend()

plt.tight_layout()
plt.savefig("b_dynamics/e_tv.pdf")
plt.show()