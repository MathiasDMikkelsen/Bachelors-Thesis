import numpy as np
import matplotlib.pyplot as plt
from inner_solver import solve

# a. set policy
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g = 5.0

# b. range for tau_z
tau_z_values = np.linspace(0.1, 3.0, 50)

# c. lists to store results
t_c_values = []
t_d_values = []
z_c_values = []
z_d_values = []
w_values   = []
p_d_values = []

# d. evaluate model and store results
for tau_z in tau_z_values:
    _, results, _ = solve(tau_w, tau_z, g)
    t_c_values.append(results["t_c"])
    t_d_values.append(results["t_d"])
    z_c_values.append(results["z_c"])
    z_d_values.append(results["z_d"])
    w_values.append(results["w"])
    p_d_values.append(results["p_d"])

# e. plot
fig, axs = plt.subplots(2, 3, figsize=(8, 4)) 

# e1. subplot 1
axs[0, 0].plot(tau_z_values, t_c_values, color="blue")
axs[0, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 0].set_ylabel(r'$t_c$', fontsize=8)

# e2. subplot 2
axs[0, 1].plot(tau_z_values, t_d_values, color="orange")
axs[0, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 1].set_ylabel(r'$t_d$', fontsize=8)

# e3. subplot 3
axs[0, 2].plot(tau_z_values, z_c_values, color="green")
axs[0, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 2].set_ylabel(r'$z_c$', fontsize=8)

# e4. subplot 4
axs[1, 0].plot(tau_z_values, z_d_values, color="red")
axs[1, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 0].set_ylabel(r'$z_d$', fontsize=8)

# e5. subplot 5
axs[1, 1].plot(tau_z_values, p_d_values, color="purple")
axs[1, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 1].set_ylabel(r'$p_d$', fontsize=8)

# e6. subplot 6
axs[1, 2].plot(tau_z_values, w_values, color="brown")
axs[1, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 2].set_ylabel(r'$w$', fontsize=8)

plt.tight_layout()
plt.savefig("a_impulse_resp.pdf")
plt.show()