# a_impulse_2hh.py (Modified for 2-household solvers)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

# --- MODIFIED: Import correct 2-hh inner solver ---
try:
    # Assuming the inner solver is in a subdirectory 'a_solvers' relative to script's parent
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    solver_path = os.path.join(project_root, 'a_solvers')
    if solver_path not in sys.path:
        sys.path.insert(0, solver_path)
    import inner_solver_ext as solver # Use alias solver
    # Import n=2 if needed elsewhere, but solver function is primary use
    from inner_solver_ext import n
    assert n == 2, "Loaded inner solver is not for n=2 households."
except (ImportError, ModuleNotFoundError):
    print("Error: Could not import 'inner_solver_ext'.")
    print("Ensure 'inner_solver_ext.py' exists in the 'a_solvers' directory relative to this script's parent directory.")
    sys.exit(1)
except FileNotFoundError:
     print("Warning: Could not automatically determine project root. Assuming solver is importable.")
     import inner_solver_ext as solver
     from inner_solver_ext import n
     assert n == 2, "Loaded inner solver is not for n=2 households."
# ---------------------------------------------------

# a. Set policies for 2 households [tau_w_d, tau_w_c]
# --- MODIFIED: Define tau_w policies of length 2 ---
tau_w_policy_a = np.array([0.10, 0.15]) # Example Policy A
tau_w_policy_b = np.array([0.05, 0.20]) # Example Policy B
# ---------------------------------------------------
g = 5.0

# b. Range for tau_z
tau_z_values = np.linspace(0.1, 20.0, 50)

# Prepare storage lists for the two policy scenarios:
# Policy A results
t_c_a = []
t_d_a = []
z_c_a = []
z_d_a = []
w_c_a = [] # Storing w_c
w_d_a = [] # Storing w_d
p_d_a = []
l_agents_a = [] # List of arrays (each length 2)
c_agents_a = [] # List of arrays (each length 2)
d_agents_a = [] # List of arrays (each length 2)

# Policy B results
t_c_b = []
t_d_b = []
z_c_b = []
z_d_b = []
w_c_b = [] # Storing w_c
w_d_b = [] # Storing w_d
p_d_b = []
l_agents_b = [] # List of arrays (each length 2)
c_agents_b = [] # List of arrays (each length 2)
d_agents_b = [] # List of arrays (each length 2)


print("Running simulations for 2 policies...")
for i, tau_z in enumerate(tau_z_values):
    print(f"  Calculating for tau_z = {tau_z:.2f} ({i+1}/{len(tau_z_values)})")
    # --- Policy A ---
    try:
        # --- MODIFIED: Call solver with policy A ---
        solution_a, results_a, converged_a = solver.solve(tau_w_policy_a, tau_z, g)
        if not converged_a:
             print(f"    Warning: Policy A solver did not converge for tau_z={tau_z:.2f}")
             # Append NaN on non-convergence
             results_a = {key: np.nan for key in results_a} # Create NaN dict
             results_a.update({ # Ensure array keys exist with correct NaN shape
                "l_agents": np.full(n, np.nan), "c_agents": np.full(n, np.nan),
                "d_agents": np.full(n, np.nan)
                })

        # --- MODIFIED: Extract results for policy A ---
        t_c_a.append(results_a.get("t_c", np.nan))
        t_d_a.append(results_a.get("t_d", np.nan))
        z_c_a.append(results_a.get("z_c", np.nan))
        z_d_a.append(results_a.get("z_d", np.nan))
        w_c_a.append(results_a.get("w_c", np.nan)) # Store w_c
        w_d_a.append(results_a.get("w_d", np.nan)) # Store w_d
        p_d_a.append(results_a.get("p_d", np.nan))
        l_agents_a.append(results_a.get("l_agents")) # Store array [l_d, l_c]
        c_agents_a.append(results_a.get("c_agents")) # Store array [c_d, c_c]
        d_agents_a.append(results_a.get("d_agents")) # Store array [d_d, d_c]
        # -------------------------------------------

    except Exception as e:
        print(f"    Error during Policy A solve for tau_z={tau_z:.2f}: {e}")
        # Append NaNs if solve fails unexpectedly
        t_c_a.append(np.nan); t_d_a.append(np.nan); z_c_a.append(np.nan); z_d_a.append(np.nan)
        w_c_a.append(np.nan); w_d_a.append(np.nan); p_d_a.append(np.nan)
        l_agents_a.append(np.full(n, np.nan)); c_agents_a.append(np.full(n, np.nan)); d_agents_a.append(np.full(n, np.nan))


    # --- Policy B ---
    try:
        # --- MODIFIED: Call solver with policy B ---
        solution_b, results_b, converged_b = solver.solve(tau_w_policy_b, tau_z, g)
        if not converged_b:
             print(f"    Warning: Policy B solver did not converge for tau_z={tau_z:.2f}")
             results_b = {key: np.nan for key in results_b}
             results_b.update({
                "l_agents": np.full(n, np.nan), "c_agents": np.full(n, np.nan),
                "d_agents": np.full(n, np.nan)
                })

        # --- MODIFIED: Extract results for policy B ---
        t_c_b.append(results_b.get("t_c", np.nan))
        t_d_b.append(results_b.get("t_d", np.nan))
        z_c_b.append(results_b.get("z_c", np.nan))
        z_d_b.append(results_b.get("z_d", np.nan))
        w_c_b.append(results_b.get("w_c", np.nan)) # Store w_c
        w_d_b.append(results_b.get("w_d", np.nan)) # Store w_d
        p_d_b.append(results_b.get("p_d", np.nan))
        l_agents_b.append(results_b.get("l_agents")) # Store array [l_d, l_c]
        c_agents_b.append(results_b.get("c_agents")) # Store array [c_d, c_c]
        d_agents_b.append(results_b.get("d_agents")) # Store array [d_d, d_c]
        # -------------------------------------------

    except Exception as e:
        print(f"    Error during Policy B solve for tau_z={tau_z:.2f}: {e}")
        t_c_b.append(np.nan); t_d_b.append(np.nan); z_c_b.append(np.nan); z_d_b.append(np.nan)
        w_c_b.append(np.nan); w_d_b.append(np.nan); p_d_b.append(np.nan)
        l_agents_b.append(np.full(n, np.nan)); c_agents_b.append(np.full(n, np.nan)); d_agents_b.append(np.full(n, np.nan))

print("Simulations complete. Aggregating and plotting...")

# Aggregate results (sum over the 2 agents)
# Use np.nansum to handle potential NaNs robustly
total_leisure_a = [np.nansum(la) for la in l_agents_a]
total_consumption_c_a = [np.nansum(ca) for ca in c_agents_a] # Sum of c_d, c_c
total_consumption_d_a = [np.nansum(da) for da in d_agents_a] # Sum of d_d, d_c

total_leisure_b = [np.nansum(la) for la in l_agents_b]
total_consumption_c_b = [np.nansum(ca) for ca in c_agents_b]
total_consumption_d_b = [np.nansum(da) for da in d_agents_b]

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 3, figsize=(9, 8)) # Slightly larger figure

# Define consistent colors
color_a = "#1f77b4" # Blueish for policy A
color_b = "#ff7f0e" # Orangish for policy B
color_wc = "#1f77b4" # Blue for w_c
color_wd = "#ff7f0e" # Orange for w_d

# Top row: Labor and Clean Pollution
axs[0,0].plot(tau_z_values, t_c_a, color=color_a, linestyle='-', linewidth=2)
axs[0,0].plot(tau_z_values, t_c_b, color=color_b, linestyle='--', linewidth=2)
axs[0,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,0].set_ylabel(r'$t_c$', fontsize=12, rotation=90)
axs[0,0].set_title('Labor Demand (Clean)', fontsize=12)

axs[0,1].plot(tau_z_values, t_d_a, color=color_a, linestyle='-', linewidth=2)
axs[0,1].plot(tau_z_values, t_d_b, color=color_b, linestyle='--', linewidth=2)
axs[0,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,1].set_ylabel(r'$t_d$', fontsize=12, rotation=90)
axs[0,1].set_title('Labor Demand (Dirty)', fontsize=12)

axs[0,2].plot(tau_z_values, z_c_a, color=color_a, linestyle='-', linewidth=2)
axs[0,2].plot(tau_z_values, z_c_b, color=color_b, linestyle='--', linewidth=2)
axs[0,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,2].set_ylabel(r'$z_c$', fontsize=12, rotation=90)
axs[0,2].set_title('Pollution Input (Clean)', fontsize=12)

# Middle row: Dirty Pollution, Price, Wages
axs[1,0].plot(tau_z_values, z_d_a, color=color_a, linestyle='-', linewidth=2)
axs[1,0].plot(tau_z_values, z_d_b, color=color_b, linestyle='--', linewidth=2)
axs[1,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,0].set_ylabel(r'$z_d$', fontsize=12, rotation=90)
axs[1,0].set_title('Pollution Input (Dirty)', fontsize=12)

axs[1,1].plot(tau_z_values, p_d_a, color=color_a, linestyle='-', linewidth=2)
axs[1,1].plot(tau_z_values, p_d_b, color=color_b, linestyle='--', linewidth=2)
axs[1,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,1].set_ylabel(r'$p_d$', fontsize=12, rotation=90)
axs[1,1].set_title('Price (Dirty Good)', fontsize=12)

# --- MODIFIED: Plot w_c and w_d for both policies ---
axs[1,2].plot(tau_z_values, w_c_a, color=color_wc, linestyle='-', linewidth=2, label=r'$w_c$ (Policy A)')
axs[1,2].plot(tau_z_values, w_d_a, color=color_wd, linestyle='-', linewidth=2, label=r'$w_d$ (Policy A)')
axs[1,2].plot(tau_z_values, w_c_b, color=color_wc, linestyle='--', linewidth=2, label=r'$w_c$ (Policy B)')
axs[1,2].plot(tau_z_values, w_d_b, color=color_wd, linestyle='--', linewidth=2, label=r'$w_d$ (Policy B)')
axs[1,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,2].set_ylabel(r'Wages', fontsize=12, rotation=90)
axs[1,2].set_title('Sector Wages ($w_c, w_d$)', fontsize=12)
axs[1,2].legend(fontsize=9) # Local legend for wages
# ----------------------------------------------------

# Bottom row: Aggregates over 2 households
axs[2,0].plot(tau_z_values, total_leisure_a, color=color_a, linestyle='-', linewidth=2)
axs[2,0].plot(tau_z_values, total_leisure_b, color=color_b, linestyle='--', linewidth=2)
axs[2,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,0].set_ylabel(r'$\sum \ell_i$', fontsize=12, rotation=90)
axs[2,0].set_title('Total Leisure (2 HH)', fontsize=12)

axs[2,1].plot(tau_z_values, total_consumption_c_a, color=color_a, linestyle='-', linewidth=2)
axs[2,1].plot(tau_z_values, total_consumption_c_b, color=color_b, linestyle='--', linewidth=2)
axs[2,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,1].set_ylabel(r'$\sum c_i$', fontsize=12, rotation=90)
axs[2,1].set_title('Total Clean Good Consumption', fontsize=12) # Updated Title

axs[2,2].plot(tau_z_values, total_consumption_d_a, color=color_a, linestyle='-', linewidth=2)
axs[2,2].plot(tau_z_values, total_consumption_d_b, color=color_b, linestyle='--', linewidth=2)
axs[2,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,2].set_ylabel(r'$\sum d_i$', fontsize=12, rotation=90)
axs[2,2].set_title('Total Dirty Good Consumption', fontsize=12) # Updated Title

# Apply grid to all subplots
for ax in axs.flat:
    ax.grid(True, color='grey', linestyle=':', linewidth=0.5, alpha=0.7)

# --- MODIFIED: Global legend labels ---
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=color_a, lw=2, linestyle='-'),
                Line2D([0], [0], color=color_b, lw=2, linestyle='--')]
# Describe the policies being compared
policy_a_label = f"Policy A ($\\tau_w$={tau_w_policy_a})"
policy_b_label = f"Policy B ($\\tau_w$={tau_w_policy_b})"
fig.legend(custom_lines, [policy_a_label, policy_b_label],
           loc="lower center", ncol=1, # Use 1 column if labels get long
           bbox_to_anchor=(0.5, -0.02), # Adjust position below plots
           frameon=False, fontsize=11)
# -----------------------------------

plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust rect tighter, leave space below for legend

# Save the figure
output_dir = "b_dynamics_2hh" # New directory name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "a_impulse_2hh.pdf") # New filename
plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches='tight'
print(f"\nPlot saved to {output_path}")

plt.show()