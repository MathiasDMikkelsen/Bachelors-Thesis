import numpy as np
import matplotlib.pyplot as plt
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve

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
def E_star(p_d, w, U_target):
    A = alpha + beta + gamma
    # Compute the multiplicative factor K:
    K = (( (A/alpha)*p_c )**alpha * ((A/beta)*p_d)**beta * ((A/gamma)*(1-tau_w[i])*phi[i]*w)**gamma )**(1/A)
    # Expenditure function:
    E_net = K * np.exp(U_target / A)
    return p_d * d0 + E_net

# e. income base defined as minimum expediture needed to achieve baseline case utility (should be equal to actual baseline income due to duality)
income_base = np.zeros(n)
for i in range(n):
    income_base[i] = E_star(p_d_base, w_base, U_base[i])
    
# f. comupte cv
tau_z_values = np.linspace(0.1, 10.0, 50)
CV_array     = np.zeros((n, len(tau_z_values)))
rel_CV_array = np.zeros((n, len(tau_z_values)))

for j, tau_z in enumerate(tau_z_values):
    sol_new, res_new, conv_new = solve(tau_w, tau_z, g)
    if not conv_new:
        print(f"Model did not converge for tau_z = {tau_z:.2f}")
    p_d_new = res_new["p_d"]
    for i in range(n):
        expenditure_needed = E_star(p_d_new, w_base, U_base[i])
        CV_array[i, j] = expenditure_needed - income_base[i]
        rel_CV_array[i, j] = CV_array[i, j] / income_base[i]

# h. plot
plt.figure(figsize=(5, 3.5))
for i in range(n):
    plt.plot(tau_z_values, rel_CV_array[i, :],
             label=f'$i={i+1}$', linestyle='-')
plt.xlabel(r'$\tau_z$', fontsize=14)
plt.ylabel(r'$\frac{CV_i}{m^d_i}$', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("b_dynamics/b_cv.pdf")

