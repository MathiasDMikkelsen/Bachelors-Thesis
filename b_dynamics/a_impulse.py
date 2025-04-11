import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set up the project root so that a_solvers can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

# a. Set policies
# First tau_w (original curve)
tau_w_orig = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
# Second tau_w (new impulse response)
tau_w_new = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
g = 5.0

# b. Range for tau_z
tau_z_values = np.linspace(1.0, 15.0, 50)

# Prepare storage lists for the two scenarios (each list is a list of impulse responses for that variable):
# For original tau_w:
t_c_orig   = []
t_d_orig   = []
z_c_orig   = []
z_d_orig   = []
w_orig     = []
p_d_orig   = []
l_agents_orig = []   # leisure (ℓ_i)
c_agents_orig = []   # consumption (c_i)
d_agents_orig = []   # demand (d_i)

# For new tau_w:
t_c_new   = []
t_d_new   = []
z_c_new   = []
z_d_new   = []
w_new     = []
p_d_new   = []
l_agents_new = []
c_agents_new = []
d_agents_new = []

# c. Loop over tau_z_values for both sets of tau_w:
for tau_z in tau_z_values:
    # Evaluate for original tau_w
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
    
    # Evaluate for new tau_w
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

# Compute totals (aggregated over households) for each tau_z value:
total_leisure_orig     = [np.sum(la) for la in l_agents_orig]
total_consumption_orig = [np.sum(ca) for ca in c_agents_orig]
total_demand_orig      = [np.sum(da) for da in d_agents_orig]

total_leisure_new     = [np.sum(la) for la in l_agents_new]
total_consumption_new = [np.sum(ca) for ca in c_agents_new]
total_demand_new      = [np.sum(da) for da in d_agents_new]

# d. Create subplots grid with 3 rows and 3 columns
fig, axs = plt.subplots(3, 3, figsize=(8.5, 6.5))

# --- Top Row: First three impulse-response plots ---
# Subplot (0,0): t_c vs. tau_z
axs[0,0].plot(tau_z_values, t_c_orig, color="blue", label=r"$\tau_{w,i}^0$")
axs[0,0].plot(tau_z_values, t_c_new, color="blue", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[0,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,0].set_ylabel(r'$t_c$', fontsize=12, rotation=90)
axs[0,0].legend(fontsize=10)

# Subplot (0,1): t_d vs. tau_z
axs[0,1].plot(tau_z_values, t_d_orig, color="orange", label=r"$\tau_{w,i}^0$")
axs[0,1].plot(tau_z_values, t_d_new, color="orange", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[0,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,1].set_ylabel(r'$t_d$', fontsize=12, rotation=90)
axs[0,1].legend(fontsize=10)

# Subplot (0,2): z_c vs. tau_z
axs[0,2].plot(tau_z_values, z_c_orig, color="green", label=r"$\tau_{w,i}^0$")
axs[0,2].plot(tau_z_values, z_c_new, color="green", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[0,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[0,2].set_ylabel(r'$z_c$', fontsize=12, rotation=90)
axs[0,2].legend(fontsize=10)

# --- Middle Row: Next three impulse-response plots ---
# Subplot (1,0): z_d vs. tau_z
axs[1,0].plot(tau_z_values, z_d_orig, color="red", label=r"$\tau_{w,i}^0$")
axs[1,0].plot(tau_z_values, z_d_new, color="red", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[1,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,0].set_ylabel(r'$z_d$', fontsize=12, rotation=90)
axs[1,0].legend(fontsize=10)

# Subplot (1,1): p_d vs. tau_z
axs[1,1].plot(tau_z_values, p_d_orig, color="purple", label=r"$\tau_{w,i}^0$")
axs[1,1].plot(tau_z_values, p_d_new, color="purple", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[1,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,1].set_ylabel(r'$p_d$', fontsize=12, rotation=90)
axs[1,1].legend(fontsize=10)

# Subplot (1,2): w vs. tau_z
axs[1,2].plot(tau_z_values, w_orig, color="brown", label=r"$\tau_{w,i}^0$")
axs[1,2].plot(tau_z_values, w_new, color="brown", linestyle="--", label=r"$\tau_{w,i}^{opt}$")
axs[1,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[1,2].set_ylabel(r'$w$', fontsize=12, rotation=90)
axs[1,2].legend(fontsize=10)

# --- Bottom Row: Total household leisure, consumption, and demand ---
# Subplot (2,0): Total Leisure vs. tau_z
axs[2,0].plot(tau_z_values, total_leisure_orig, 'b-', linewidth=2, label=r"$\tau_{w,i}^0$")
axs[2,0].plot(tau_z_values, total_leisure_new, 'b--', linewidth=2, label=r"$\tau_{w,i}^{opt}$")
axs[2,0].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,0].set_ylabel(r'$\sum \ell_i$', fontsize=12, rotation=90)
axs[2,0].legend(fontsize=10)

# Subplot (2,1): Total Consumption vs. tau_z
axs[2,1].plot(tau_z_values, total_consumption_orig, 'r-', linewidth=2, label=r"$\tau_{w,i}^0$")
axs[2,1].plot(tau_z_values, total_consumption_new, 'r--', linewidth=2, label=r"$\tau_{w,i}^{opt}$")
axs[2,1].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,1].set_ylabel(r'$\sum c_i$', fontsize=12, rotation=90)
axs[2,1].legend(fontsize=10)

# Subplot (2,2): Total Demand vs. tau_z
axs[2,2].plot(tau_z_values, total_demand_orig, 'g-', linewidth=2, label=r"$\tau_{w,i}^0$")
axs[2,2].plot(tau_z_values, total_demand_new, 'g--', linewidth=2, label=r"$\tau_{w,i}^{opt}$")
axs[2,2].set_xlabel(r'$\tau_z$', fontsize=12)
axs[2,2].set_ylabel(r'$\sum d_i$', fontsize=12, rotation=90)
axs[2,2].legend(fontsize=10)

plt.tight_layout()
plt.savefig("b_dynamics/a_impulse.pdf")
plt.show()