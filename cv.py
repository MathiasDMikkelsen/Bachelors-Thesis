import numpy as np
import matplotlib.pyplot as plt
from inner_solver import solve

# a. calibration
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
t         = 24.0     
d0        = 0.5
p_c       = 1.0      
phi       = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n         = len(phi)

# b. initial polciy vector
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g     = 5.0

# c. baseline equlibrium 
tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge.")
w_base        = res_base["w"]
p_d_base      = res_base["p_d"]
l_base        = res_base["l"]
l_agents_base = res_base["l_agents"]
c_agents_base = res_base["c_agents"]
d_agents_base = res_base["d_agents"]

U_base = np.zeros(n)
for i in range(n):
    U_base[i] = (alpha * np.log(c_agents_base[i]) +
                 beta  * np.log(d_agents_base[i] - d0) +
                 gamma * np.log(l_agents_base[i]))

# d. expenditure function
A = alpha + beta + gamma
def E_star(p_d, U_target):
    return (A/alpha) * np.exp((U_target - beta*np.log(beta/(alpha*p_d)) - gamma*np.log(gamma/alpha)) / A) + p_d*d0

# e. income base defined as minimum expediture needed to achieve baseline case utility (should be equal to actual baseline income due to duality)
income_base = np.zeros(n)
for i in range(n):
    income_base[i] = E_star(p_d_base, U_base[i])
    
# f. comupte cv
tau_z_values = np.linspace(0.1, 3.0, 50)
CV_array     = np.zeros((n, len(tau_z_values)))
rel_CV_array = np.zeros((n, len(tau_z_values)))

for j, tau_z in enumerate(tau_z_values):
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge for tau_z = {tau_z:.2f}")
    p_d_new = res_new["p_d"]
    for i in range(n):
        expenditure_needed = E_star(p_d_new, U_base[i])
        CV_array[i, j] = expenditure_needed - income_base[i]
        rel_CV_array[i, j] = CV_array[i, j] / income_base[i]

# h. plot
plt.figure(figsize=(7, 5))
for i in range(n):
    plt.plot(tau_z_values, rel_CV_array[i, :],
             label=f'$i={i+1}$', linestyle='-')
plt.xlabel(r'$\tau_z$', fontsize=14)
plt.ylabel(r'$\frac{CV_i}{m^d_i}$', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("relative_cv.pdf")
plt.show()

# i. compensate wage