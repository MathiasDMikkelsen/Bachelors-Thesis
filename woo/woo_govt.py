import numpy as np
from scipy.optimize import minimize
from clean_inner import solve_and_return  # This file defines solve_and_return(tau_z, tau_w_input)

# SWF parameters:
xi = 0.1    # Penalty weight on aggregate pollution (Z_C+Z_D)
n = 5       # Number of households

def objective_policy(policy):
    """
    Policy vector is 6-dimensional:
      - first 5 entries: household-specific tau_w's,
      - 6th entry: common tau_z.
      
    SWF is defined as:
      SWF = sum_i U_i - 5*xi*(Z_C+Z_D)
    where U_i = C_i^alpha + (D_i-D0)^beta + l_i^gamma.
    
    This function calls solve_and_return. If equilibrium fails or returns NaN,
    it assigns a heavy penalty so that the optimizer avoids that region.
    Returns -SWF (plus any penalty) for minimization.
    """
    tau_w_vec = np.array(policy[:n])
    tau_z = policy[n]
    sol_result = solve_and_return(tau_z, tau_w_vec)
    if sol_result is None:
        return 1e6  # heavy penalty if equilibrium doesn't converge
    sum_util, sum_Z = sol_result
    if np.isnan(sum_util) or np.isnan(sum_Z):
        return 1e6
    # Enforce nonnegative pollution
    penalty = 0.0
    if sum_Z < 0:
        penalty += 1e6 + abs(sum_Z)*1e3
    SWF = sum_util - n * xi * sum_Z
    return -SWF + penalty

def constraint_sum_z(policy):
    """
    Nonlinear constraint: aggregate complementary input Z_C+Z_D must be >= 0.
    """
    tau_w_vec = np.array(policy[:n])
    tau_z = policy[n]
    sol_result = solve_and_return(tau_z, tau_w_vec)
    if sol_result is None:
        return -1e6
    _, sum_Z = sol_result
    return sum_Z

# Initial guess for policy: for example, tau_w = [0,0,0,0,0] and tau_z = 1.0.
initial_policy = np.concatenate((np.zeros(n), [1.0]))

# Bounds: each tau_w in (-5,5) and tau_z in (0.001, 0.5)
bounds = [(-5, 5)] * n + [(0.5, 100.0)]
constraints = [{'type': 'ineq', 'fun': constraint_sum_z}]

# Use SLSQP optimizer
res = minimize(objective_policy, x0=initial_policy, bounds=bounds,
               constraints=constraints, method='SLSQP')

print("Optimization result:")
print(res)
optimal_policy = res.x
opt_tau_w = optimal_policy[:n]
opt_tau_z = optimal_policy[n]
print("\nOptimal policy parameters:")
for i in range(n):
    print(f"tau_w[{i+1}] = {opt_tau_w[i]:.4f}")
print(f"tau_z = {opt_tau_z:.4f}")

# Finally, solve equilibrium at the optimal policy and print details:
result = solve_and_return(opt_tau_z, opt_tau_w)
if result is not None:
    sum_util, sum_Z = result
    SWF_opt = sum_util - n * xi * sum_Z
    print("\nAt the optimal policy:")
    print("Sum of household utilities =", sum_util)
    print("Aggregate complementary input (Z_C+Z_D) =", sum_Z)
    print("Optimal SWF =", SWF_opt)