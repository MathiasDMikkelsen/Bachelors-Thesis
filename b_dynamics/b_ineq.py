import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Set project root so that inner_solver can be imported.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

# =============================================================================
# 1. Calibration and Policy Parameters
# =============================================================================
alpha = 0.7
beta  = 0.2
gamma = 0.2
t     = 24.0     
d0    = 0.5
p_c   = 1.0      
phi   = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n     = len(phi)

# Define income tax (wage tax) vectors:
tau_w     = np.array([0.015, 0.072, 0.115, 0.156, 0.24])                # Original combo
tau_w_alt = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])  # Alternative combo
g = 5.0

# Dual expenditure function: minimum expenditure required to achieve U_target.
def E_star(p_d, U_target):
    A = alpha + beta + gamma
    return (A/alpha) * np.exp((U_target - beta * np.log(beta/(alpha*p_d))
                                - gamma * np.log(gamma/alpha)) / A) + p_d*d0

# =============================================================================
# 2. Baseline Equilibrium (tau_z_baseline = 1.0) for each income tax combo
# =============================================================================
tau_z_baseline = 1.0

# -- Baseline for original income tax combo --
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge for original combo.")
w_base        = res_base["w"]
p_d_base      = res_base["p_d"]
l_base        = res_base["l"]
l_agents_base = res_base["l_agents"]
c_agents_base = res_base["c_agents"]
d_agents_base = res_base["d_agents"]

# Baseline utility (for CV and utility change) per household.
U_base_orig = np.zeros(n)
for i in range(n):
    U_base_orig[i] = (alpha * np.log(c_agents_base[i]) +
                      beta  * np.log(d_agents_base[i] - d0) +
                      gamma * np.log(l_agents_base[i]))
# Baseline expenditure (dual) and disposable income for original combo.
baseline_exp_orig = np.zeros(n)
baseline_income_orig = np.zeros(n)
for i in range(n):
    baseline_exp_orig[i] = E_star(p_d_base, U_base_orig[i])
    baseline_income_orig[i] = phi[i] * w_base * (1 - tau_w[i]) * (t - l_agents_base[i]) + l_base

# -- Baseline for alternative income tax combo --
sol_base_alt, res_base_alt, conv_base_alt = solve(tau_w_alt, tau_z_baseline, g)
if not conv_base_alt:
    print("Baseline model did not converge for alternative combo.")
w_base_alt        = res_base_alt["w"]
p_d_base_alt      = res_base_alt["p_d"]
l_base_alt        = res_base_alt["l"]
l_agents_base_alt = res_base_alt["l_agents"]
c_agents_base_alt = res_base_alt["c_agents"]
d_agents_base_alt = res_base_alt["d_agents"]

U_base_alt = np.zeros(n)
for i in range(n):
    U_base_alt[i] = (alpha * np.log(c_agents_base_alt[i]) +
                     beta  * np.log(d_agents_base_alt[i] - d0) +
                     gamma * np.log(l_agents_base_alt[i]))
baseline_exp_alt = np.zeros(n)
baseline_income_alt = np.zeros(n)
for i in range(n):
    baseline_exp_alt[i] = E_star(p_d_base_alt, U_base_alt[i])
    baseline_income_alt[i] = phi[i] * w_base_alt * (1 - tau_w_alt[i]) * (t - l_agents_base_alt[i]) + l_base_alt

# =============================================================================
# 3. Experiment: Varying Environmental Tax (tau_z) and computing outcomes
# =============================================================================
tau_z_values = np.linspace(1.0, 20.0, 50)
m = len(tau_z_values)

# Initialize arrays:
# CV relative to baseline income.
CV_rel_orig = np.zeros((n, m))
CV_rel_alt  = np.zeros((n, m))
# Change in total disposable income (new income minus baseline).
income_change_orig = np.zeros((n, m))
income_change_alt  = np.zeros((n, m))
# Change in exponentially transformed utility relative to baseline.
utilities_change_orig = np.zeros((n, m))
utilities_change_alt  = np.zeros((n, m))

# Loop over tau_z values.
for j, tau_z in enumerate(tau_z_values):
    # --- Simulation for original income tax combo ---
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge (original) for tau_z = {tau_z:.2f}")
        continue
    p_d_new      = res_new["p_d"]
    w_new        = res_new["w"]
    l_new        = res_new["l"]
    l_agents_new = res_new["l_agents"]
    U_new        = res_new["utilities"]
    for i in range(n):
        CV_abs = E_star(p_d_new, U_base_orig[i]) - baseline_exp_orig[i]
        CV_rel_orig[i, j] = CV_abs / baseline_income_orig[i]
        new_income = phi[i] * w_new * (1 - tau_w[i]) * (t - l_agents_new[i]) + l_new
        income_change_orig[i, j] = new_income - baseline_income_orig[i]
        utilities_change_orig[i, j] = np.exp(U_new[i]) - np.exp(U_base_orig[i])
    
    # --- Simulation for alternative income tax combo ---
    sol_new_alt, res_new_alt, conv_new_alt = solve(tau_w_alt, tau_z, g)
    if not conv_new_alt:
        print(f"Model did not converge (alternative) for tau_z = {tau_z:.2f}")
        continue
    p_d_new_alt      = res_new_alt["p_d"]
    w_new_alt        = res_new_alt["w"]
    l_new_alt        = res_new_alt["l"]
    l_agents_new_alt = res_new_alt["l_agents"]
    U_new_alt        = res_new_alt["utilities"]
    for i in range(n):
        CV_abs_alt = E_star(p_d_new_alt, U_base_alt[i]) - baseline_exp_alt[i]
        CV_rel_alt[i, j] = CV_abs_alt / baseline_income_alt[i]
        new_income_alt = phi[i] * w_new_alt * (1 - tau_w_alt[i]) * (t - l_agents_new_alt[i]) + l_new_alt
        income_change_alt[i, j] = new_income_alt - baseline_income_alt[i]
        utilities_change_alt[i, j] = np.exp(U_new_alt[i]) - np.exp(U_base_alt[i])

# =============================================================================
# 4. Plotting: Arrange the figures so that Panel 1 (CV relative to income) is on the top‐left,
#    Panel 2 (Total income change) is on the top‐right, Panel 3 (Change in utility) is placed directly
#    below Panel 1, and the legend is to the right of Panel 3.
# =============================================================================
# Define a common color palette for households.
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Create a figure with a 2x2 grid.
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

# Top left: Panel 1 (CV relative to income)
ax0 = fig.add_subplot(gs[0, 0])
# Top right: Panel 2 (Total income change)
ax1 = fig.add_subplot(gs[0, 1])
# Bottom left: Panel 3 (Change in utility)
ax2 = fig.add_subplot(gs[1, 0])
# Bottom right: Legend
ax3 = fig.add_subplot(gs[1, 1])

# Panel 1: CV Relative to Income
for i in range(n):
    ax0.plot(tau_z_values, CV_rel_orig[i, :], linestyle='-', color=colors[i])
    ax0.plot(tau_z_values, CV_rel_alt[i, :], linestyle='--', color=colors[i])
ax0.set_xlabel(r'$\tau_z$', fontsize=15)
ax0.set_ylabel(r'$CV/m^d_i$', fontsize=15, labelpad=20)
ax0.set_title('CV relative to income', fontsize=15)
ax0.tick_params(labelsize=12)
ax0.set_xlim(tau_z_values[0], tau_z_values[-1])
ax0.yaxis.set_label_coords(-0.15, 0.5)

# Panel 2: Total Income Change
for i in range(n):
    ax1.plot(tau_z_values, income_change_orig[i, :], linestyle='-', color=colors[i])
    ax1.plot(tau_z_values, income_change_alt[i, :], linestyle='--', color=colors[i])
ax1.axhline(0, color='grey', linestyle=':', linewidth=1.5)
ax1.set_xlabel(r'$\tau_z$', fontsize=15)
ax1.set_ylabel(r'$\Delta m_i^d+\Delta l$', fontsize=15, labelpad=20)
ax1.set_title('Total income change', fontsize=15)
ax1.tick_params(labelsize=12)
ax1.set_xlim(tau_z_values[0], tau_z_values[-1])
ax1.yaxis.set_label_coords(-0.15, 0.5)

# Panel 3: Change in Exponentially Transformed Utility
for i in range(n):
    ax2.plot(tau_z_values, utilities_change_orig[i, :], linestyle='-', color=colors[i])
    ax2.plot(tau_z_values, utilities_change_alt[i, :], linestyle='--', color=colors[i])
ax2.axhline(0, color='grey', linestyle=':', linewidth=1.5)
ax2.set_xlabel(r'$\tau_z$', fontsize=15)
ax2.set_ylabel(r'$\Delta u_i$', fontsize=15, labelpad=20)
ax2.set_title('Change in utility', fontsize=15)
ax2.tick_params(labelsize=12)
ax2.set_xlim(tau_z_values[0], tau_z_values[-1])
ax2.yaxis.set_label_coords(-0.15, 0.5)

# Panel 4: Legend (placed to the right of Panel 3)
ax3.axis('off')  # Turn off the axis for the legend panel.
legend_handles = [Line2D([0], [0], color=colors[i], lw=3, label=rf'$i={i+1}$') for i in range(n)]
ax3.legend(handles=legend_handles, loc='center', fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("b_dynamics/b_ineq.pdf")
plt.show()