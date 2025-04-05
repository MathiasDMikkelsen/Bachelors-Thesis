import numpy as np
import matplotlib.pyplot as plt
from inner_solver import solve

# Fixed parameters for the model
tau_w = np.array([0.0, 0.0, 0.0, 0.0, 0.00])
g = 5.0

# Define a range for tau_z
tau_z_values = np.linspace(0.1, 3.0, 50)

# Preallocate lists for the quantities we want to plot.
t_c_values = []
t_d_values = []
z_c_values = []
z_d_values = []
w_values   = []
p_d_values = []

# Evaluate the model for each tau_z and collect the results.
for tau_z in tau_z_values:
    _, results, _ = solve(tau_w, tau_z, g)
    t_c_values.append(results["t_c"])
    t_d_values.append(results["t_d"])
    z_c_values.append(results["z_c"])
    z_d_values.append(results["z_d"])
    w_values.append(results["w"])
    p_d_values.append(results["p_d"])

# Create a 2x3 grid figure (2 rows, 3 columns) with individual x-axis labels.
fig, axs = plt.subplots(2, 3, figsize=(8, 4))  # Not sharing x-axis to ensure x-labels on all plots

# Subplot 1: firm input t_c
axs[0, 0].plot(tau_z_values, t_c_values, color="blue")
axs[0, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 0].set_ylabel(r'$t_c$', fontsize=8)

# Subplot 2: firm input t_d
axs[0, 1].plot(tau_z_values, t_d_values, color="orange")
axs[0, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 1].set_ylabel(r'$t_d$', fontsize=8)

# Subplot 3: z_c
axs[0, 2].plot(tau_z_values, z_c_values, color="green")
axs[0, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 2].set_ylabel(r'$z_c$', fontsize=8)

# Subplot 4: z_d
axs[1, 0].plot(tau_z_values, z_d_values, color="red")
axs[1, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 0].set_ylabel(r'$z_d$', fontsize=8)

# Subplot 5: price p_d
axs[1, 1].plot(tau_z_values, p_d_values, color="purple")
axs[1, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 1].set_ylabel(r'$p_d$', fontsize=8)

# Subplot 6: wage w
axs[1, 2].plot(tau_z_values, w_values, color="brown")
axs[1, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 2].set_ylabel(r'$w$', fontsize=8)

plt.tight_layout()
plt.savefig("combined_figure.pdf")
plt.show()