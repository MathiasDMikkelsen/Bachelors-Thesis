# plot_aggregate_and_abatement_responses.py (Using Abatement + Direct HH Redist Solver)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- MODIFIED: Ensure correct path and import LATEST inner_solver ---
try:
    # Import the latest inner_solver.py (with abatement + direct HH redist)
    import inner_solver_ext as solver
    from inner_solver_ext import n, phi, t as T # Import needed params
except ImportError:
    print("Error: Could not import 'inner_solver_ext'.") # Ensure filename is correct
    print("Ensure the solver file exists and is accessible.")
    exit()
# --- End Import Modification ---

# a. Set policies
tau_w_orig = np.array([0.015, 0.072, 0.115, 0.156, 0.24]) # Pre-existing
tau_w_new = np.array([-1.08858208, -0.04377549,  0.22144972,  0.39697164,  0.64084534]) # Optimal (calculated previously)
g = 5.0

# b. Range for tau_z
tau_z_values = np.linspace(1.0, 5.0, 40) # Adjusted range from user code

# Prepare storage lists
t_c_orig=[]; t_d_orig=[]; z_c_orig=[]; z_d_orig=[]; a_c_orig=[]; a_d_orig=[]; w_orig=[]; p_d_orig=[]; l_agents_orig=[]; c_agents_orig=[]; d_agents_orig=[]
t_c_new=[]; t_d_new=[]; z_c_new=[]; z_d_new=[]; a_c_new=[]; a_d_new=[]; w_new=[]; p_d_new=[]; l_agents_new=[]; c_agents_new=[]; d_agents_new=[]

# c. Loop through tau_z values and solve
print("Running simulations for plots (Abatement Model)...")
for tau_z in tau_z_values:
    # Scenario 1: Pre-existing tau_w
    sol_orig, results_orig, converged_orig = solver.solve(tau_w_orig, tau_z, g)
    if converged_orig and results_orig is not None:
        t_c_orig.append(results_orig["t_c"]); t_d_orig.append(results_orig["t_d"])
        z_c_orig.append(results_orig["z_c"]); z_d_orig.append(results_orig["z_d"])
        a_c_orig.append(results_orig["a_c"]); a_d_orig.append(results_orig["a_d"]) # Store abatement
        w_orig.append(results_orig["w"]); p_d_orig.append(results_orig["p_d"])
        l_agents_orig.append(results_orig["l_agents"]); c_agents_orig.append(results_orig["c_agents"]); d_agents_orig.append(results_orig["d_agents"])
    else:
        print(f"Warning: Orig solve failed for tau_z={tau_z:.2f}")
        # Append NaNs to all lists
        for lst in [t_c_orig, t_d_orig, z_c_orig, z_d_orig, a_c_orig, a_d_orig, w_orig, p_d_orig]: lst.append(np.nan)
        for lst in [l_agents_orig, c_agents_orig, d_agents_orig]: lst.append([np.nan]*n)

    # Scenario 2: "Optimal" baseline tau_w
    sol_new, results_new, converged_new = solver.solve(tau_w_new, tau_z, g)
    if converged_new and results_new is not None:
        t_c_new.append(results_new["t_c"]); t_d_new.append(results_new["t_d"])
        z_c_new.append(results_new["z_c"]); z_d_new.append(results_new["z_d"])
        a_c_new.append(results_new["a_c"]); a_d_new.append(results_new["a_d"]) # Store abatement
        w_new.append(results_new["w"]); p_d_new.append(results_new["p_d"])
        l_agents_new.append(results_new["l_agents"]); c_agents_new.append(results_new["c_agents"]); d_agents_new.append(results_new["d_agents"])
    else:
        print(f"Warning: New solve failed for tau_z={tau_z:.2f}")
        # Append NaNs to all lists
        for lst in [t_c_new, t_d_new, z_c_new, z_d_new, a_c_new, a_d_new, w_new, p_d_new]: lst.append(np.nan)
        for lst in [l_agents_new, c_agents_new, d_agents_new]: lst.append([np.nan]*n)
print("Simulations finished.")

# d. Calculate aggregate results
def sum_nan(list_of_arrays):
    sums = [];
    for arr in list_of_arrays: sums.append(np.nan if np.any(np.isnan(arr)) else np.sum(arr))
    return sums

total_leisure_orig = sum_nan(l_agents_orig); total_consumption_orig = sum_nan(c_agents_orig); total_demand_orig = sum_nan(d_agents_orig)
total_leisure_new = sum_nan(l_agents_new); total_consumption_new = sum_nan(c_agents_new); total_demand_new = sum_nan(d_agents_new)

# e. Convert all lists to numpy arrays
lists_to_convert_orig = [t_c_orig, t_d_orig, z_c_orig, z_d_orig, a_c_orig, a_d_orig, w_orig, p_d_orig, total_leisure_orig, total_consumption_orig, total_demand_orig]
lists_to_convert_new = [t_c_new, t_d_new, z_c_new, z_d_new, a_c_new, a_d_new, w_new, p_d_new, total_leisure_new, total_consumption_new, total_demand_new]
results_np_orig = {name: np.array(data) for name, data in zip(['t_c','t_d','z_c','z_d','a_c','a_d','w','p_d','total_leisure','total_consumption','total_demand'], lists_to_convert_orig)}
results_np_new = {name: np.array(data) for name, data in zip(['t_c','t_d','z_c','z_d','a_c','a_d','w','p_d','total_leisure','total_consumption','total_demand'], lists_to_convert_new)}

# --- Plotting Section ---

plt.style.use('seaborn-v0_8-whitegrid')
line_color = 'teal' # Re-using color

# --- FIGURE 1: Original 3x3 Grid ---
fig1, axs1 = plt.subplots(3, 3, figsize=(12, 10))
fig1.suptitle('Aggregate Variable Responses to Env. Tax (Abatement Model)', fontsize=16, y=0.98)

# Helper function for plotting
def plot_ax(ax, x, y_orig, y_new, title, ylabel):
    ax.plot(x, y_orig, color=line_color, linestyle='-', linewidth=1.5)
    ax.plot(x, y_new, color=line_color, linestyle='--', linewidth=1.5)
    ax.set_xlabel(r'$\tau_z$', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10, rotation=90)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(True, color='grey', linestyle=':', linewidth=0.5, alpha=0.6)

# Plotting each subplot for Figure 1
plot_ax(axs1[0,0], tau_z_values, results_np_orig['t_c'], results_np_new['t_c'], 'Labor, Clean Sector', r'$t_c$')
plot_ax(axs1[0,1], tau_z_values, results_np_orig['t_d'], results_np_new['t_d'], 'Labor, Dirty Sector', r'$t_d$')
plot_ax(axs1[0,2], tau_z_values, results_np_orig['z_c'], results_np_new['z_c'], 'Pollution, Clean Sector', r'$z_c$')
plot_ax(axs1[1,0], tau_z_values, results_np_orig['z_d'], results_np_new['z_d'], 'Pollution, Dirty Sector', r'$z_d$')
plot_ax(axs1[1,1], tau_z_values, results_np_orig['p_d'], results_np_new['p_d'], 'Price, Dirty Good', r'$p_d$')
plot_ax(axs1[1,2], tau_z_values, results_np_orig['w'], results_np_new['w'], 'Wage', r'$w$')
plot_ax(axs1[2,0], tau_z_values, results_np_orig['total_leisure'], results_np_new['total_leisure'], 'Total Leisure', r'$\sum \ell_i$')
plot_ax(axs1[2,1], tau_z_values, results_np_orig['total_consumption'], results_np_new['total_consumption'], 'Clean Good Supply', r'$\sum c_i$')
plot_ax(axs1[2,2], tau_z_values, results_np_orig['total_demand'], results_np_new['total_demand'], 'Dirty Good Supply', r'$\sum d_i$')

# Global legend for Figure 1
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=line_color, lw=1.5, linestyle='-'),
                Line2D([0], [0], color=line_color, lw=1.5, linestyle='--')]
fig1.legend(custom_lines, [r"Pre-existing $\tau_w$ (Abatement Model)", r"Optimal Baseline $\tau_w$ (Abatement Model)"],
           loc="lower center", ncol=2, frameon=False, fontsize=11)

fig1.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save Figure 1
output_dir = "aggregate_dynamics_abatement" # Use same directory
if not os.path.exists(output_dir): os.makedirs(output_dir)
output_path1 = os.path.join(output_dir, "aggregate_impulse_abatement.pdf") # Original filename
fig1.savefig(output_path1)
print(f"\nAggregate plot saved to {output_path1}")

# --- FIGURE 2: Abatement Schedules ---
fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4.5)) # 1 row, 2 columns
fig2.suptitle('Abatement Response to Environmental Tax (Abatement Model)', fontsize=14, y=0.98)

# Plotting abatement for clean sector
plot_ax(axs2[0], tau_z_values, results_np_orig['a_c'], results_np_new['a_c'], 'Abatement, Clean Sector', r'$a_c$')

# Plotting abatement for dirty sector
plot_ax(axs2[1], tau_z_values, results_np_orig['a_d'], results_np_new['a_d'], 'Abatement, Dirty Sector', r'$a_d$')

# Global legend for Figure 2 (reuse definitions from above)
fig2.legend(custom_lines, [r"Pre-existing $\tau_w$ (Abatement Model)", r"Optimal Baseline $\tau_w$ (Abatement Model)"],
           loc="lower center", ncol=2, frameon=False, fontsize=11)

fig2.tight_layout(rect=[0, 0.08, 1, 0.93]) # Adjust rect for legend

# Save Figure 2
output_path2 = os.path.join(output_dir, "abatement_impulse_abatement.pdf") # New filename for abatement plot
fig2.savefig(output_path2)
print(f"Abatement plot saved to {output_path2}")

plt.show() # Show both figures