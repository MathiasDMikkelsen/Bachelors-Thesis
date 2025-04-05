import numpy as np
import matplotlib.pyplot as plt
from inner_solver import solve

# ------------------------------
# 1. Model and Calibration Parameters
# ------------------------------
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
t         = 24.0     
d0        = 0.5
p_c       = 1.0      
phi       = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n         = len(phi)

# All households have tau_w = 0 so that differences in income come solely from phi.
tau_w = np.zeros(n)
g     = 0.0

# ------------------------------
# 2. Baseline Equilibrium (tau_z_baseline = 0.1)
# ------------------------------
tau_z_baseline = 0.1
sol_base, res_base, conv_base = solve(tau_w, tau_z_baseline, g)
if not conv_base:
    print("Baseline model did not converge.")

# Extract baseline equilibrium variables
w_base        = res_base["w"]
p_d_base      = res_base["p_d"]
l_base        = res_base["l"]
l_agents_base = res_base["l_agents"]
c_agents_base = res_base["c_agents"]
d_agents_base = res_base["d_agents"]

# Compute baseline utility for each household using Stone–Geary utility:
# U = alpha*ln(c_i) + beta*ln(d_i - d0) + gamma*ln(ell_i)
U_base = np.zeros(n)
for i in range(n):
    U_base[i] = (alpha * np.log(c_agents_base[i]) +
                 beta  * np.log(d_agents_base[i] - d0) +
                 gamma * np.log(l_agents_base[i]))

# ------------------------------
# 3. Closed-Form Expenditure Function E*(p_d, U)
# ------------------------------
# Derived analytically:
# E*(p_d, U) = (alpha+beta+gamma)/alpha * exp((U - beta ln(beta/(alpha p_d)) - gamma ln(gamma/alpha))/(alpha+beta+gamma)) + p_d*d0
A = alpha + beta + gamma
def E_star(p_d, U_target):
    return (A/alpha) * np.exp((U_target - beta*np.log(beta/(alpha*p_d)) - gamma*np.log(gamma/alpha)) / A) + p_d*d0

# ------------------------------
# 4. Define Baseline Income Consistently
# ------------------------------
# We define the baseline income for each household as the minimal expenditure needed 
# to achieve the baseline utility at the baseline price.
income_base = np.zeros(n)
for i in range(n):
    income_base[i] = E_star(p_d_base, U_base[i])
    
# By construction, at tau_z_baseline the compensating variation (CV) will be zero.

# ------------------------------
# 5. Compute Compensating Variation (CV) Using the Closed-Form
# ------------------------------
# For a new environmental tax (which changes p_d), we have:
# CV = E*(p_d_new, U_base) - income_base, and relative CV = CV/income_base.
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

# ------------------------------
# 6. Plot Relative CV vs. Environmental Tax (tau_z)
# ------------------------------
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(tau_z_values, rel_CV_array[i, :],
             label=f'Household {i+1}', linestyle='-')
plt.xlabel(r'$\tau_z$', fontsize=12)
plt.ylabel('Relative CV (CV / Baseline Income)', fontsize=12)
plt.title('Relative Compensating Variation vs. Environmental Tax', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("relative_CV_closed_form.pdf")
plt.show()