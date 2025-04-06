import numpy as np
from scipy.optimize import minimize
from clean_inner_ic import solve_and_return  
# solve_and_return(tau_z, tau_w_input) must return:
# (sum_util, sum_Z, C_agents, D_agents, l_agents, I_agents, Z_C, Z_D, U_agents)

# SWF and model parameters:
xi = 5.0    # Penalty weight on aggregate pollution (Z_C+Z_D)
n = 5       # Number of households
T_val = 50.0  # Time endowment per household
eps_val = 1e-6  # A small number for strict inequalities

def objective_policy(policy):
    """
    Policy vector (length 6):
      - first 5 entries: household-specific tau_w's,
      - 6th entry: common tau_z.
      
    SWF = sum_i U_i - 5*xi*(Z_C+Z_D), where
         U_i = C_i^alpha + (D_i-D0)^beta + l_i^gamma.
    Returns -SWF (plus a heavy penalty) if any outcome is nonpositive or if the equilibrium fails.
    """
    try:
        tau_w_vec = np.array(policy[:n])
        tau_z = policy[n]
        sol = solve_and_return(tau_z, tau_w_vec)
        if sol is None:
            return 1e6
        sum_util, sum_Z, C_agents, D_agents, l_agents, I_agents, Z_C, Z_D, U_agents = sol
        
        outcomes = np.concatenate((
            np.array([sum_util, sum_Z, Z_C, Z_D]),
            C_agents, D_agents, l_agents, I_agents, U_agents
        ))
        if not np.all(np.isfinite(outcomes)):
            return 1e6
        # If any key outcomes are nonpositive, penalize.
        if (sum_util <= 0 or sum_Z <= 0 or Z_C <= 0 or Z_D <= 0 or 
            np.any(C_agents <= 0) or np.any(D_agents <= 0) or np.any(l_agents <= 0) or np.any(I_agents <= 0)):
            return 1e6
        
        SWF = sum_util - n*xi*sum_Z
        return -SWF
    except Exception as e:
        return 1e6

def finite_diff_jac(fun, x, eps=1e-6):
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
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

def constraint_sum_z(policy):
    """
    Nonlinear constraint: aggregate complementary input (Z_C+Z_D) must be >= 0.
    """
    try:
        tau_w_vec = np.array(policy[:n])
        tau_z = policy[n]
        sol = solve_and_return(tau_z, tau_w_vec)
        if sol is None:
            return -1e6
        _, sum_Z, _, _, _, _, _, _, _ = sol
        if not np.isfinite(sum_Z):
            return -1e6
        return sum_Z
    except Exception as e:
        return -1e6

def constraint_positive_outcomes(policy):
    """
    Returns a vector of inequality constraints:
      - For each household i, C_i - eps >= 0,
      - For each household i, D_i - eps >= 0,
      - For each household i, (T_val - l_i) - eps >= 0, i.e. l_i <= T_val - eps.
    """
    try:
        tau_w_vec = np.array(policy[:n])
        tau_z = policy[n]
        sol = solve_and_return(tau_z, tau_w_vec)
        if sol is None:
            return np.full(3*n, -1e6)
        _, _, C_agents, D_agents, l_agents, _, _, _, _ = sol
        cons = []
        for i in range(n):
            cons.append(C_agents[i] - eps_val)  # must be >= 0
            cons.append(D_agents[i] - eps_val)  # must be >= 0
            cons.append((T_val - l_agents[i]) - eps_val)  # ensures l_i < T_val
        return np.array(cons)
    except Exception as e:
        return np.full(3*n, -1e6)

def compute_pretend_utility(i, j, C_agents, D_agents, l_agents, tau_w_vec):
    """
    Placeholder function to compute U_j^i, the utility that household i would receive
    if it were assigned household j's bundle but uses its own tax rate.
    
    Let I_j = φ_j*(T_val - l_agents[j]) (ignoring w for relative comparisons),
    then set l_j^i = T_val - I_j/((1-τ_w_i)*φ_i).
    Finally, define U_j^i = (C_agents[j])^α + (D_agents[j]-D0)^β + (l_j^i)^γ.
    
    **NOTE:** This is a simple placeholder. Replace with your model's actual formula.
    """
    alpha_local = 0.7
    beta_local = 0.2
    gamma_local = 0.2
    T_val_local = 50.0
    D0_local = 5.0
    phi_vals = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
    
    I_j = phi_vals[j] * (T_val_local - l_agents[j])
    denom = (1 - tau_w_vec[i]) * phi_vals[i]
    if denom <= 0:
        return 1e6
    l_j_i = T_val_local - I_j / denom
    if l_j_i <= 0:
        return 1e6
    U_j_i = (C_agents[j]**alpha_local) + ((D_agents[j]-D0_local)**beta_local) + (l_j_i**gamma_local)
    return U_j_i

def ic_constraints(policy):
    """
    Returns an array of incentive compatibility (IC) constraints.
    For each pair i ≠ j, require:
         IC_{i,j} = U_i - U_j^i ≥ 0,
    where U_i is household i’s actual utility and U_j^i is computed by compute_pretend_utility.
    """
    try:
        tau_w_vec = np.array(policy[:n])
        tau_z = policy[n]
        sol = solve_and_return(tau_z, tau_w_vec)
        if sol is None:
            return np.full(n*(n-1), -1e6)
        _, _, C_agents, D_agents, l_agents, _, _, _, U_agents = sol
        ic_list = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                U_actual = U_agents[i]
                U_pretend = compute_pretend_utility(i, j, C_agents, D_agents, l_agents, tau_w_vec)
                ic_list.append(U_actual - U_pretend)
        return np.array(ic_list)
    except Exception as e:
        return np.full(n*(n-1), -1e6)

# Initial guess for policy: e.g., tau_w = [0,0,0,0,0] and tau_z = 1.0.
initial_policy = np.concatenate(([-0.5, -0.5, 0, 0.5, 0.5], [0.3]))

# Bounds: each tau_w in (-5, 5) and tau_z in (0.001, 5.0)
bounds = [(-5, 5)] * n + [(0.001, 5.0)]
constraints = [
    {'type': 'ineq', 'fun': constraint_sum_z},
    {'type': 'ineq', 'fun': constraint_positive_outcomes},
    {'type': 'ineq', 'fun': ic_constraints}
]

options = {
    'verbose': 1,
    'gtol': 1e-3,
    'xtol': 1e-3,
    'barrier_tol': 1e-3,
    'maxiter': 1000
}

res = minimize(objective_policy, x0=initial_policy, bounds=bounds,
               constraints=constraints, method='trust-constr',
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

sol = solve_and_return(opt_tau_z, opt_tau_w)
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