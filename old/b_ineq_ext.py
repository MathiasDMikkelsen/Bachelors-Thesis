# plot_distributional_effects_abatement.py (Using Abatement + Direct HH Redist Solver)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# --- MODIFIED: Ensure correct path and import LATEST inner_solver ---
# Set up project root so we can import the solver

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import j_inner_solver_ext as solver # Use alias solver
from j_inner_solver_ext import n, p_a, alpha, beta, gamma, t, d0, phi

# --- Assert n=5 ---
assert n == 5, "This script expects n=5 based on the imported inner_solver."

# =============================================================================
# 1. Calibration and Policy Parameters
# =============================================================================
# alpha, beta, gamma, t, d0, p_c=1.0, phi, n, p_a imported from inner_solver

# Define income tax (wage tax) vectors:
# Using tau_w values from user's previous plotting script version
tau_w_init = np.array([0.015, 0.072, 0.115, 0.156, 0.24])      # Pre-existing (used as tau_w)
tau_w_alt = np.array([-1.08858208, -0.04377549,  0.22144972,  0.39697164,  0.64084534]) # Optimal Baseline (used as tau_w_alt)
g = 5.0

# Dual expenditure function (Unchanged)
def E_star(p_d, U_target):
    # U_target here is log utility
    A = alpha + beta + gamma
    # Handle potential log(0) or negative inputs if U_target is extremely low
    exp_arg = (U_target - beta * np.log(beta/(alpha*p_d)) - gamma * np.log(gamma/alpha)) / A
    # Prevent overflow/underflow in exp
    if exp_arg > 700: return np.inf # Avoid overflow
    if exp_arg < -700: return p_d*d0 # Avoid issues with exp(very small number) -> 0
    return (A/alpha) * np.exp(exp_arg) + p_d*d0

# =============================================================================
# 2. Baseline Equilibrium (tau_z_baseline = 1.0) for each income tax combo
# =============================================================================
tau_z_baseline = 1.0 # Using same baseline as user code
print("Calculating baseline equilibrium...")

# -- Baseline for 'original' income tax combo (tau_w_init) --
sol_base1, res_base1, conv_base1 = solver.solve(tau_w_init, tau_z_baseline, g)
if not conv_base1 or res_base1 is None:
    print("FATAL: Baseline model failed for tax system 1. Cannot proceed.")
    exit()
# Store necessary baseline values
w_base1 = res_base1["w"]
p_d_base1 = res_base1["p_d"]
l_base1 = res_base1["l"]
l_agents_base1 = res_base1["l_agents"]
U_base1 = res_base1["utilities"] # log utility
a_c_base1 = res_base1["a_c"]
a_d_base1 = res_base1["a_d"]

baseline_exp1 = np.zeros(n)
baseline_income1 = np.zeros(n)
total_abatement_cost_base1 = p_a * (a_c_base1 + a_d_base1) # Calculate baseline abatement cost
for i in range(n):
    # Calculate baseline expenditure needed for baseline utility U_base1[i] at baseline price p_d_base1
    baseline_exp1[i] = E_star(p_d_base1, U_base1[i])
    # Calculate baseline income CONSISTENT with how h_i is calculated in solver
    baseline_income1[i] = (phi[i] * w_base1 * (1 - tau_w_init[i]) * (t - l_agents_base1[i]) # Labor income
                           + l_base1                                                      # Gov transfer
                           + (total_abatement_cost_base1 / n))                          # Redistributed abatement

# -- Baseline for 'alternative' income tax combo (tau_w_alt) --
sol_base2, res_base2, conv_base2 = solve(tau_w_alt, tau_z_baseline, g)
if not conv_base2 or res_base2 is None:
    print("FATAL: Baseline model failed for tax system 2. Cannot proceed.")
    exit()
# Store necessary baseline values
w_base2 = res_base2["w"]
p_d_base2 = res_base2["p_d"]
l_base2 = res_base2["l"]
l_agents_base2 = res_base2["l_agents"]
U_base2 = res_base2["utilities"] # log utility
a_c_base2 = res_base2["a_c"]
a_d_base2 = res_base2["a_d"]

baseline_exp2 = np.zeros(n)
baseline_income2 = np.zeros(n)
total_abatement_cost_base2 = p_a * (a_c_base2 + a_d_base2) # Calculate baseline abatement cost
for i in range(n):
    baseline_exp2[i] = E_star(p_d_base2, U_base2[i])
    baseline_income2[i] = (phi[i] * w_base2 * (1 - tau_w_alt[i]) * (t - l_agents_base2[i]) # Labor income
                           + l_base2                                                       # Gov transfer
                           + (total_abatement_cost_base2 / n))                           # Redistributed abatement

print("Baseline calculations complete.")
# =============================================================================
# 3. Experiment: Varying Environmental Tax (tau_z) and computing outcomes
# =============================================================================
tau_z_values = np.linspace(1.0, 5.0, 25) # Adjusted range from user code
m = len(tau_z_values)

# Initialize arrays:
CV_rel1 = np.zeros((n, m)); CV_rel2 = np.zeros((n, m))
income_change1 = np.zeros((n, m)); income_change2 = np.zeros((n, m))
utility_change1 = np.zeros((n, m)); utility_change2 = np.zeros((n, m)) # Change in LEVEL utility

print("Running simulations across tau_z range...")
for j, tau_z in enumerate(tau_z_values):
    # --- Simulation for original income tax combo (tau_w_init) ---
    sol_new1, res_new1, conv_new1 = solve(tau_w_init, tau_z, g)
    if conv_new1 and res_new1 is not None:
        p_d_new1=res_new1["p_d"]; w_new1=res_new1["w"]; l_new1=res_new1["l"]
        l_agents_new1=res_new1["l_agents"]; U_new1=res_new1["utilities"]
        a_c_new1=res_new1["a_c"]; a_d_new1=res_new1["a_d"]
        total_abatement_cost_new1 = p_a * (a_c_new1 + a_d_new1)
        for i in range(n):
            # CV Calculation (Expenditure needed at new price for BASE utility - BASE expenditure)
            CV_abs1 = E_star(p_d_new1, U_base1[i]) - baseline_exp1[i]
            # Relative CV (handle potential zero baseline income?)
            CV_rel1[i, j] = CV_abs1 / baseline_income1[i] if baseline_income1[i] != 0 else np.nan
            # Income Change Calculation (Corrected: includes abatement redistribution)
            new_income1 = (phi[i] * w_new1 * (1 - tau_w_init[i]) * (t - l_agents_new1[i])
                           + l_new1
                           + (total_abatement_cost_new1 / n))
            income_change1[i, j] = new_income1 - baseline_income1[i]
            # Utility Change (Level) Calculation
            # Check for invalid log utility values before exponentiating
            if U_new1[i] > -1e8 and U_base1[i] > -1e8:
                 utility_change1[i, j] = np.exp(U_new1[i]) - np.exp(U_base1[i])
            else:
                 utility_change1[i, j] = np.nan # Mark as invalid if log utility was bad
    else:
        print(f"Warning: Model did not converge (tau_w_init) for tau_z = {tau_z:.2f}")
        CV_rel1[:, j] = np.nan; income_change1[:, j] = np.nan; utility_change1[:, j] = np.nan;

    # --- Simulation for alternative income tax combo (tau_w_alt) ---
    sol_new2, res_new2, conv_new2 = solve(tau_w_alt, tau_z, g)
    if conv_new2 and res_new2 is not None:
        p_d_new2=res_new2["p_d"]; w_new2=res_new2["w"]; l_new2=res_new2["l"]
        l_agents_new2=res_new2["l_agents"]; U_new2=res_new2["utilities"]
        a_c_new2=res_new2["a_c"]; a_d_new2=res_new2["a_d"]
        total_abatement_cost_new2 = p_a * (a_c_new2 + a_d_new2)
        for i in range(n):
            CV_abs2 = E_star(p_d_new2, U_base2[i]) - baseline_exp2[i]
            CV_rel2[i, j] = CV_abs2 / baseline_income2[i] if baseline_income2[i] != 0 else np.nan
            # Income Change Calculation (Corrected: includes abatement redistribution)
            new_income2 = (phi[i] * w_new2 * (1 - tau_w_alt[i]) * (t - l_agents_new2[i])
                           + l_new2
                           + (total_abatement_cost_new2 / n))
            income_change2[i, j] = new_income2 - baseline_income2[i]
            # Utility Change (Level) Calculation
            if U_new2[i] > -1e8 and U_base2[i] > -1e8:
                 utility_change2[i, j] = np.exp(U_new2[i]) - np.exp(U_base2[i])
            else:
                 utility_change2[i, j] = np.nan
    else:
        print(f"Warning: Model did not converge (tau_w_alt) for tau_z = {tau_z:.2f}")
        CV_rel2[:, j] = np.nan; income_change2[:, j] = np.nan; utility_change2[:, j] = np.nan;

print("Simulations finished.")
# =============================================================================
# 4. Plotting: Use same GridSpec layout as user provided. DO NOT CHANGE FORMATTING.
# =============================================================================
print("Generating plot...")
# Create a list of blue shades
blue_cmap = plt.cm.Blues
colors = [blue_cmap(0.3 + 0.5 * i/(n-1)) for i in range(n)]
line_width = 2.5

# Create a figure with the specified GridSpec layout.
fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.01)
ax0 = fig.add_subplot(gs[0, 0]) # Top left: CV
ax1 = fig.add_subplot(gs[0, 1]) # Top right: Income Change
ax2 = fig.add_subplot(gs[1, 0]) # Bottom left: Utility Change
ax3 = fig.add_subplot(gs[1, 1]) # Bottom right: Legend

# Panel 1: CV Relative to Income
for i in range(n):
    ax0.plot(tau_z_values, CV_rel1[i, :], linestyle='-', lw=line_width, color=colors[i])
    ax0.plot(tau_z_values, CV_rel2[i, :], linestyle='--', lw=line_width, color=colors[i])
ax0.set_xlabel(r'$\tau_z$', fontsize=15)
ax0.set_ylabel(r'$CV/m^d_i$', fontsize=15, labelpad=20)
ax0.set_title('CV relative to income', fontsize=15) # Keep original title
ax0.tick_params(labelsize=12)
ax0.set_xlim(tau_z_values[0], tau_z_values[-1])
ax0.yaxis.set_label_coords(-0.15, 0.5)
ax0.set_box_aspect(1)

# Panel 2: Total Income Change
for i in range(n):
    ax1.plot(tau_z_values, income_change1[i, :], linestyle='-', lw=line_width, color=colors[i])
    ax1.plot(tau_z_values, income_change2[i, :], linestyle='--', lw=line_width, color=colors[i])
ax1.axhline(0, color='grey', linestyle='-', linewidth=2.5) # Changed from '--' in user code, keeping as '-'
ax1.set_xlabel(r'$\tau_z$', fontsize=15)
ax1.set_ylabel(r'$\Delta m_i^d+\Delta l + \Delta AbateShare$', fontsize=15, labelpad=20) # Modified label slightly
ax1.set_title('Total income change', fontsize=15) # Keep original title
ax1.tick_params(labelsize=12)
ax1.set_xlim(tau_z_values[0], tau_z_values[-1])
ax1.yaxis.set_label_coords(-0.15, 0.5)
ax1.set_box_aspect(1)

# Panel 3: Change in Exponentially Transformed Utility (Level)
for i in range(n):
    ax2.plot(tau_z_values, utility_change1[i, :], linestyle='-', lw=line_width, color=colors[i])
    ax2.plot(tau_z_values, utility_change2[i, :], linestyle='--', lw=line_width, color=colors[i])
ax2.axhline(0, color='grey', linestyle='-', linewidth=2.5) # Changed from '--' in user code, keeping as '-'
ax2.set_xlabel(r'$\tau_z$', fontsize=15)
ax2.set_ylabel(r'$\Delta u_i$', fontsize=15, labelpad=20) # Keep original label (implicitly level change now)
ax2.set_title(r'Change in utility', fontsize=15) # Keep original title
ax2.tick_params(labelsize=12)
ax2.set_xlim(tau_z_values[0], tau_z_values[-1])
ax2.yaxis.set_label_coords(-0.15, 0.5)
ax2.set_box_aspect(1)

# Apply grid to plot axes
for ax in [ax0, ax1, ax2]:
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

# Panel 4: Legend
ax3.axis('off')
legend_handles = [Line2D([0], [0], color=colors[i], lw=line_width, label=rf'$i={i+1}$') for i in range(n)]
# Add lines for the two scenarios
legend_handles.append(Line2D([0], [0], color='black', lw=line_width, linestyle='-', label='Pre-existing $\\tau_w$'))
legend_handles.append(Line2D([0], [0], color='black', lw=line_width, linestyle='--', label='Optimal Baseline $\\tau_w$'))
ax3.legend(handles=legend_handles, loc='center left', fontsize=15, frameon=False) # Adjusted loc slightly

plt.tight_layout() # Use standard tight_layout

# Save figure
output_dir = "distributional_effects_abatement" # New directory name
if not os.path.exists(output_dir): os.makedirs(output_dir)
output_path = os.path.join(output_dir, "distributional_impacts_abatement.pdf") # New filename
plt.savefig(output_path)#, bbox_inches='tight', pad_inches=0.05) # Removed bbox args
print(f"\nDistributional plot saved to {output_path}")
plt.show()