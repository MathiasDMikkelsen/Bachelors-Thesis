import numpy as np
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

# a. Set policies
tau_w_orig = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_new = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
g = 5.0

# b. Range for tau_z
tau_z_values = np.linspace(0.1, 20.0, 50)

# Prepare storage lists for the two scenarios:
t_c_orig       = []
t_d_orig       = []
z_c_orig       = []
z_d_orig       = []
w_orig         = []
p_d_orig       = []
l_agents_orig  = []
c_agents_orig  = []
d_agents_orig  = []

t_c_new        = []
t_d_new        = []
z_c_new        = []
z_d_new        = []
w_new          = []
p_d_new        = []
l_agents_new   = []
c_agents_new   = []
d_agents_new   = []

for tau_z in tau_z_values:
    _, results_orig, _ = solve(tau_w_orig, tau_z, g)
    t_c_orig.append(results_orig["t_c"])
    t_d_orig.append(results_orig["t_d"])
    z_c_orig.append(results_orig["z_c"])
    z_d_orig.append(results_orig["z_d"])
    w_orig.append(results_orig["w"])
    p_d_orig.append(results_orig["p_d"])
    l_agents_orig.append(results_orig["l_agents"])
    c_agents_orig.append(results_orig["c_agents"])
    d_agents_orig.append(results_orig["d_agents"])
    
    _, results_new, _ = solve(tau_w_new, tau_z, g)
    t_c_new.append(results_new["t_c"])
    t_d_new.append(results_new["t_d"])
    z_c_new.append(results_new["z_c"])
    z_d_new.append(results_new["z_d"])
    w_new.append(results_new["w"])
    p_d_new.append(results_new["p_d"])
    l_agents_new.append(results_new["l_agents"])
    c_agents_new.append(results_new["c_agents"])
    d_agents_new.append(results_new["d_agents"])

total_leisure_orig     = [np.sum(la) for la in l_agents_orig]
total_consumption_orig = [np.sum(ca) for ca in c_agents_orig]
total_demand_orig      = [np.sum(da) for da in d_agents_orig]

total_leisure_new     = [np.sum(la) for la in l_agents_new]
total_consumption_new = [np.sum(ca) for ca in c_agents_new]
total_demand_new      = [np.sum(da) for da in d_agents_new]

fig, axs = plt.subplots(3, 3, figsize=(8.5, 7.5))

# Use light blue for all lines
light_blue = "#7aabd4"

# Top row
axs[0,0].plot(tau_z_values, t_c_orig, color=light_blue, linestyle='-', linewidth=2)
axs[0,0].plot(tau_z_values, t_c_new, color=light_blue, linestyle='--', linewidth=2)
axs[0,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,0].set_ylabel(r'$t_c$', fontsize=12, rotation=90)
axs[0,0].set_title('Labor, clean sector', fontsize=12)

axs[0,1].plot(tau_z_values, t_d_orig, color=light_blue, linestyle='-', linewidth=2)
axs[0,1].plot(tau_z_values, t_d_new, color=light_blue, linestyle='--', linewidth=2)
axs[0,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,1].set_ylabel(r'$t_d$', fontsize=12, rotation=90)
axs[0,1].set_title('Labor, dirty sector', fontsize=12)

axs[0,2].plot(tau_z_values, z_c_orig, color=light_blue, linestyle='-', linewidth=2)
axs[0,2].plot(tau_z_values, z_c_new, color=light_blue, linestyle='--', linewidth=2)
axs[0,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,2].set_ylabel(r'$z_c$', fontsize=12, rotation=90)
axs[0,2].set_title('Pollution, clean sector', fontsize=12)

# Middle row
axs[1,0].plot(tau_z_values, z_d_orig, color=light_blue, linestyle='-', linewidth=2)
axs[1,0].plot(tau_z_values, z_d_new, color=light_blue, linestyle='--', linewidth=2)
axs[1,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,0].set_ylabel(r'$z_d$', fontsize=12, rotation=90)
axs[1,0].set_title('Pollution, dirty sector', fontsize=12)

axs[1,1].plot(tau_z_values, p_d_orig, color=light_blue, linestyle='-', linewidth=2)
axs[1,1].plot(tau_z_values, p_d_new, color=light_blue, linestyle='--', linewidth=2)
axs[1,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,1].set_ylabel(r'$p_d$', fontsize=12, rotation=90)
axs[1,1].set_title('Price, dirty good', fontsize=12)

axs[1,2].plot(tau_z_values, w_orig, color=light_blue, linestyle='-', linewidth=2)
axs[1,2].plot(tau_z_values, w_new, color=light_blue, linestyle='--', linewidth=2)
axs[1,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,2].set_ylabel(r'$w$', fontsize=12, rotation=90)
axs[1,2].set_title('Wage', fontsize=12)

# Bottom row
axs[2,0].plot(tau_z_values, total_leisure_orig, color=light_blue, linestyle='-', linewidth=2)
axs[2,0].plot(tau_z_values, total_leisure_new, color=light_blue, linestyle='--', linewidth=2)
axs[2,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,0].set_ylabel(r'$\sum \ell_i$', fontsize=12, rotation=90)
axs[2,0].set_title('Total leisure', fontsize=12)

axs[2,1].plot(tau_z_values, total_consumption_orig, color=light_blue, linestyle='-', linewidth=2)
axs[2,1].plot(tau_z_values, total_consumption_new, color=light_blue, linestyle='--', linewidth=2)
axs[2,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,1].set_ylabel(r'$\sum c_i$', fontsize=12, rotation=90)
axs[2,1].set_title('Clean good supply', fontsize=12)

axs[2,2].plot(tau_z_values, total_demand_orig, color=light_blue, linestyle='-', linewidth=2)
axs[2,2].plot(tau_z_values, total_demand_new, color=light_blue, linestyle='--', linewidth=2)
axs[2,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,2].set_ylabel(r'$\sum d_i$', fontsize=12, rotation=90)
axs[2,2].set_title('Dirty good supply', fontsize=12)

for ax in axs.flat:
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

# Global legend below all plots
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=light_blue, lw=2, linestyle='-'),
                Line2D([0], [0], color=light_blue, lw=2, linestyle='--')]
fig.legend(custom_lines, [r"Pre-existing income tax $(\tau_{w,i}^0)$", r"Baseline optimal income tax $(\tau_{w,i}^{opt})$"],
           loc="lower center", ncol=2, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("b_dynamics/a_impulse.pdf")
