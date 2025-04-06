import numpy as np
from scipy.optimize import minimize
from clean_inner import solve_and_return  # This file defines solve_and_return(tau_z, tau_w_input)

# SWF parameters:
xi = 0.1    # Penalty weight on aggregate pollution (Z_C+Z_D)
n = 5       # Number of households

def objective_policy(policy):

    tau_w_vec = np.array(policy[:n])
    tau_z = policy[n]
    sol_result = solve_and_return(tau_z, tau_w_vec)
    if sol_result is None:
        return 1e6  # heavy penalty if equilibrium fails
    sum_util, sum_Z = sol_result
    if np.isnan(sum_util) or np.isnan(sum_Z):
        return 1e6
    if sum_Z < 0:
        return 1e6 + abs(sum_Z)*1e3
    SWF = sum_util - n * xi * sum_Z
    return -SWF

def finite_diff_jac(fun, x, eps=1e-6):

    f0 = fun(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += eps
        f1 = fun(x1)
        x2 = x.copy()
        x2[i] -= eps
        f2 = fun(x2)
        grad[i] = (f1 - f2) / (2 * eps)
    return grad

# We'll use the trust-constr method with our finite-difference jacobian.
# Initial guess for policy: for example, tau_w = [0, 0, 0, 0, 0] and tau_z = 1.0.
initial_policy = np.concatenate((np.zeros(n), [1.0]))

# Bounds: each tau_w in (-5,5) and tau_z in (0.001, 5.0)
bounds = [(-5, 5)] * n + [(0.001, 100.0)]

# Optionally, we can include a constraint that the aggregate complementary input is >= 0.
def constraint_sum_z(policy):
    tau_w_vec = np.array(policy[:n])
    tau_z = policy[n]
    sol_result = solve_and_return(tau_z, tau_w_vec)
    if sol_result is None:
        return -1e6
    _, sum_Z = sol_result
    return sum_Z  # Must be >= 0

constraints = [{'type': 'ineq', 'fun': constraint_sum_z}]

options={'verbose': 1, 'gtol': 1e-3, 'xtol': 1e-3, 'barrier_tol': 1e-3, 'maxiter': 1000}

res = minimize(objective_policy, x0=initial_policy, bounds=bounds,
               constraints=constraints, method='SLSQP',
               jac=lambda x: finite_diff_jac(objective_policy, x),
               options=options)

print("Optimization result:")
print(res)
optimal_policy = res.x
opt_tau_w = optimal_policy[:n]
opt_tau_z = optimal_policy[n]
print("\nOptimal policy parameters:")
for i in range(n):
    print(f"tau_w[{i+1}] = {opt_tau_w[i]:.4f}")
print(f"tau_z = {opt_tau_z:.4f}")

# Finally, solve equilibrium at the optimal policy and print detailed results:
result = solve_and_return(opt_tau_z, opt_tau_w)
if result is not None:
    sum_util, sum_Z = result
    SWF_opt = sum_util - n * xi * sum_Z
    print("\nAt the optimal policy:")
    print("Sum of household utilities =", sum_util)
    print("Aggregate complementary input (Z_C+Z_D) =", sum_Z)
    print("Optimal SWF =", SWF_opt)