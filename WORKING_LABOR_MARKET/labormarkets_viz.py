import numpy as np
import matplotlib.pyplot as plt
from inner_labormarkets import solve  # assuming the new model is saved in this file

# a. set policy
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g = 5.0

# b. range for tau_z
tau_z_values = np.linspace(0.01, 2.0, 50)

# c. lists to store results
t_c_values = []
t_d_values = []
z_c_values = []
z_d_values = []
w_c_values = []
w_d_values = []
p_d_values = []
agg_c_values = []
agg_d_values = []
agg_l_values = []

# d. evaluate model and store results
for tau_z in tau_z_values:
    _, results, _ = solve(tau_w, tau_z, g)
    t_c_values.append(results["t_c"])
    t_d_values.append(results["t_d"])
    z_c_values.append(results["z_c"])
    z_d_values.append(results["z_d"])
    w_c_values.append(results["w_c"])
    w_d_values.append(results["w_d"])
    p_d_values.append(results["p_d"])
    agg_c_values.append(results["agg_c"])
    agg_d_values.append(results["agg_d"])
    agg_l_values.append(results["agg_labor"])

# e. plot
fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Adjusted figsize for better spacing

# e1. subplot 1
axs[0, 0].plot(tau_z_values, t_c_values, color="blue")
axs[0, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 0].set_ylabel(r'$t_c$', fontsize=8)
axs[0, 0].set_title('Sector C Labor vs. Tau_z')  # Added title
axs[0, 0].grid(True)  # Added grid

# e2. subplot 2
axs[0, 1].plot(tau_z_values, t_d_values, color="orange")
axs[0, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 1].set_ylabel(r'$t_d$', fontsize=8)
axs[0, 1].set_title('Sector D Labor vs. Tau_z')  # Added title
axs[0, 1].grid(True)  # Added grid

# e3. subplot 3
axs[0, 2].plot(tau_z_values, z_c_values, color="green")
axs[0, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[0, 2].set_ylabel(r'$z_c$', fontsize=8)
axs[0, 2].set_title('Sector C Pollution vs. Tau_z')  # Added title
axs[0, 2].grid(True)  # Added grid

# e4. subplot 4
axs[1, 0].plot(tau_z_values, z_d_values, color="red")
axs[1, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 0].set_ylabel(r'$z_d$', fontsize=8)
axs[1, 0].set_title('Sector D Pollution vs. Tau_z')  # Added title
axs[1, 0].grid(True)  # Added grid

# e5. subplot 5
axs[1, 1].plot(tau_z_values, p_d_values, color="purple")
axs[1, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 1].set_ylabel(r'$p_d$', fontsize=8)
axs[1, 1].set_title('Relative Price vs. Tau_z')  # Added title
axs[1, 1].grid(True)  # Added grid

# e6. subplot 6: show both wages
axs[1, 2].plot(tau_z_values, w_c_values, label=r'$w_c$', color="brown")
axs[1, 2].plot(tau_z_values, w_d_values, label=r'$w_d$', color="gray", linestyle="--")
axs[1, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[1, 2].set_ylabel(r'wages', fontsize=8)
axs[1, 2].set_title('Wages vs. Tau_z')  # Added title
axs[1, 2].legend(fontsize=7)
axs[1, 2].grid(True)

# e7. subplot 7: Aggregate Consumption
axs[2, 0].plot(tau_z_values, agg_c_values, color="cyan")
axs[2, 0].set_xlabel(r'$\tau_z$', fontsize=8)
axs[2, 0].set_ylabel(r'$C$', fontsize=8)
axs[2, 0].set_title('Aggregate Consumption vs. Tau_z')  # Added title
axs[2, 0].grid(True)

# e8. subplot 8: Aggregate Dirty Good Consumption
axs[2, 1].plot(tau_z_values, agg_d_values, color="magenta")
axs[2, 1].set_xlabel(r'$\tau_z$', fontsize=8)
axs[2, 1].set_ylabel(r'$D$', fontsize=8)
axs[2, 1].set_title('Aggregate Dirty Good vs. Tau_z')  # Added title
axs[2, 1].grid(True)

# e9. subplot 9: Aggregate Labor
axs[2, 2].plot(tau_z_values, agg_l_values, color="olive")
axs[2, 2].set_xlabel(r'$\tau_z$', fontsize=8)
axs[2, 2].set_ylabel(r'$L$', fontsize=8)
axs[2, 2].set_title('Aggregate Labor vs. Tau_z')  # Added title
axs[2, 2].grid(True)

plt.tight_layout()
plt.savefig("a_impulse_resp_two_markets.pdf")
plt.show()
