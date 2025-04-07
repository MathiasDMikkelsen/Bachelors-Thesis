import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set up the project root so that a_solvers can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

# a. Set policy
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g = 5.0

# b. Range for tau_z
tau_z_values = np.linspace(0.1, 10.0, 50)

# c. Lists to store results for impulse responses (top two rows)
t_c_values   = []
t_d_values   = []
z_c_values   = []
z_d_values   = []
w_values     = []
p_d_values   = []

# Also prepare lists to store household-level variables for totals
l_agents_values = []   # leisure (ℓ_i)
c_agents_values = []   # consumption (c_i)
d_agents_values = []   # demand (d_i)

# d. Evaluate model and store results
for tau_z in tau_z_values:
    _, results, _ = solve(tau_w, tau_z, g)
    t_c_values.append(results["t_c"])
    t_d_values.append(results["t_d"])
    z_c_values.append(results["z_c"])
    z_d_values.append(results["z_d"])
    w_values.append(results["w"])
    p_d_values.append(results["p_d"])
    
    l_agents_values.append(results["l_agents"])
    c_agents_values.append(results["c_agents"])
    d_agents_values.append(results["d_agents"])

# Compute totals (aggregated over households) for each tau_z value:
total_leisure     = [np.sum(la) for la in l_agents_values]
total_consumption = [np.sum(ca) for ca in c_agents_values]
total_demand      = [np.sum(da) for da in d_agents_values]

# e. Create a subplot grid with 3 rows and 3 columns (keep existing figure size and style)
fig, axs = plt.subplots(3, 3, figsize=(8.5, 6.5))

# --- Top Row: First three impulse-response plots ---
# Subplot (0,0): t_c vs. tau_z
axs[0,0].plot(tau_z_values, t_c_values, color="blue")
axs[0,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,0].set_ylabel(r'$t_c$', fontsize=12, rotation=90)
# Subplot (0,1): t_d vs. tau_z
axs[0,1].plot(tau_z_values, t_d_values, color="orange")
axs[0,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,1].set_ylabel(r'$t_d$', fontsize=12, rotation=90)
# Subplot (0,2): z_c vs. tau_z
axs[0,2].plot(tau_z_values, z_c_values, color="green")
axs[0,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,2].set_ylabel(r'$z_c$', fontsize=12, rotation=90)

# --- Middle Row: Next three impulse-response plots ---
# Subplot (1,0): z_d vs. tau_z
axs[1,0].plot(tau_z_values, z_d_values, color="red")
axs[1,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,0].set_ylabel(r'$z_d$', fontsize=12, rotation=90)
# Subplot (1,1): p_d vs. tau_z
axs[1,1].plot(tau_z_values, p_d_values, color="purple")
axs[1,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,1].set_ylabel(r'$p_d$', fontsize=12, rotation=90)
# Subplot (1,2): w vs. tau_z
axs[1,2].plot(tau_z_values, w_values, color="brown")
axs[1,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,2].set_ylabel(r'$w$', fontsize=12, rotation=90)

# --- Bottom Row: Total household leisure, consumption, and demand ---
# Subplot (2,0): Total Leisure vs. tau_z
axs[2,0].plot(tau_z_values, total_leisure, 'b-', linewidth=2)
axs[2,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,0].set_ylabel(r'$\sum \ell_i$', fontsize=12, rotation=90)

# Subplot (2,1): Total Consumption vs. tau_z
axs[2,1].plot(tau_z_values, total_consumption, 'r-', linewidth=2)
axs[2,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,1].set_ylabel(r'$\sum c_i$', fontsize=12, rotation=90)

# Subplot (2,2): Total Demand vs. tau_z
axs[2,2].plot(tau_z_values, total_demand, 'g-', linewidth=2)
axs[2,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,2].set_ylabel(r'$\sum d_i$', fontsize=12, rotation=90)


plt.tight_layout()
plt.savefig("b_dynamics/a_impulse.pdf")

