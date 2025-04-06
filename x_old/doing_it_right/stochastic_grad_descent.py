import numpy as np
from clean_inner_ic import solve_and_return  
# solve_and_return(tau_z, tau_w_input) should return a tuple:
# (sum_util, sum_Z, C_agents, D_agents, l_agents, I_agents, Z_C, Z_D, U_agents)

# SWF parameters:
xi = 4.0    # Penalty weight on aggregate pollution (Z_C+Z_D)
n = 5       # Number of households

def objective_policy(policy):

    try:
        tau_w_vec = np.array(policy[:n])
        tau_z = policy[n]
        sol = solve_and_return(tau_z, tau_w_vec)
        if sol is None:
            return 1e6
        sum_util, sum_Z, C_agents, D_agents, l_agents, I_agents, Z_C, Z_D, U_agents = sol
        
        # Combine key outcomes for checking:
        outcomes = np.concatenate((
            np.array([sum_util, sum_Z, Z_C, Z_D]),
            C_agents, D_agents, l_agents, I_agents, U_agents
        ))
        if not np.all(np.isfinite(outcomes)):
            return 1e6
        
        # Enforce that all outcomes are strictly positive:
        if (sum_util <= 0 or sum_Z <= 0 or Z_C <= 0 or Z_D <= 0 or
            np.any(C_agents <= 0) or np.any(D_agents <= 0) or 
            np.any(l_agents <= 0) or np.any(I_agents <= 0)):
            return 1e6
        
        SWF = sum_util - n * xi * sum_Z
        return -SWF
    except Exception as e:
        return 1e6

def finite_diff_grad(fun, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += eps
        f1 = fun(x1)
        x2 = x.copy()
        x2[i] -= eps
        f2 = fun(x2)
        if not (np.isfinite(f1) and np.isfinite(f2)):
            grad[i] = 0.0
        else:
            grad[i] = (f1 - f2) / (2 * eps)
    # Replace any NaN with zero
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad

def project_policy(policy, bounds):
    """Project policy vector onto bounds."""
    proj = np.copy(policy)
    for i, (lb, ub) in enumerate(bounds):
        proj[i] = np.clip(proj[i], lb, ub)
    return proj

# Define bounds: tau_w in (-5, 5) and tau_z in (0.001, 5.0)
bounds = [(-5, 5)] * n + [(0.001, 5.0)]

# SGD parameters
learning_rate = 1e-3
max_iters = 10000

# Initial policy vector: for example, tau_w = [0,0,0,0,0] and tau_z = 1.0.
policy = np.concatenate((np.zeros(n), [1.0]))

best_policy = policy.copy()
best_obj = objective_policy(policy)
print("Initial objective:", best_obj)

# We'll use a simple gradient descent loop as our "stochastic" optimizer.
for it in range(max_iters):
    grad = finite_diff_grad(objective_policy, policy)
    policy = policy - learning_rate * grad
    policy = project_policy(policy, bounds)
    current_obj = objective_policy(policy)
    if current_obj < best_obj:
        best_obj = current_obj
        best_policy = policy.copy()
    if it % 1000 == 0:
        print(f"Iteration {it}, objective = {current_obj}")
        
print("SGD optimization complete.")
print("Best objective (negative SWF):", best_obj)
print("Best policy:", best_policy)

# Unpack best policy:
best_tau_w = best_policy[:n]
best_tau_z = best_policy[n]
print("Optimal tau_w:", best_tau_w)
print("Optimal tau_z:", best_tau_z)

# Solve equilibrium at the optimal policy and print detailed outcomes:
sol = solve_and_return(best_tau_z, best_tau_w)
if sol is not None:
    sum_util, sum_Z, C_agents, D_agents, l_agents, I_agents, Z_C, Z_D, U_agents = sol
    SWF_opt = sum_util - n * xi * sum_Z
    print("\nAt the optimal policy:")
    print("Sum of household utilities =", sum_util)
    print("Aggregate complementary input (Z_C+Z_D) =", sum_Z)
    print("Optimal SWF =", SWF_opt)
    print("\nHousehold Outcomes:")
    for i in range(n):
        print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}, I = {I_agents[i]:.4f}")
    print(f"\nSector Complementary Inputs: Z_C = {Z_C:.4f}, Z_D = {Z_D:.4f}")