import numpy as np
from scipy.optimize import minimize
import woo.blocks as blocks  # Assuming the blocks module is available with the required functions

np.set_printoptions(suppress=True, precision=8)

# ---------------------------
# Global and fixed parameters
# ---------------------------
n = 5
G = 1.0      # Government spending
xi = 0.1     # Pollution weight in welfare
theta = 1.0  # Pollution curvature
# Use the planner’s version of phi:
phi_planner = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

# ---------------------------
# Utility functions (from the solver)
# ---------------------------
def transform(u, params, n=5):
    # Unpack parameters
    d0 = params['d0']
    
    d_raw   = u[0:5]
    ell_raw = u[5:10]
    lam     = u[10:15]
    d       = d0 + np.exp(d_raw)
    ell     = 1.0/(1.0 + np.exp(-ell_raw))
    
    t_c  = n/(1.0 + np.exp(-u[15]))
    t_d  = n/(1.0 + np.exp(-u[16]))
    z_c  = np.exp(u[17])
    z_d  = np.exp(u[18])
    p_d  = np.exp(u[19])
    w    = np.exp(u[20])
    
    return d, ell, lam, t_c, t_d, z_c, z_d, p_d, w

def market_clearing(u, params, n=5):
    """
    Computes the two market clearing conditions:
      eq_d_mkt: Goods (or dirty input) market clearing
      eq_l_mkt: Labor market clearing
    """
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(u, params, n)
    
    # Production by dirty firm (using the blocks module)
    y_d = blocks.firm_d_production(t_d, z_d, params['epsilon_d'], params['r'], params['x'])
    
    eq_d_mkt = np.sum(d) + 0.5 - y_d + 0.5 * params['G'] / p_d
    eq_l_mkt = np.sum(ell) + t_c + t_d - n * params['t_total']
    return np.array([eq_d_mkt, eq_l_mkt])

def compute_equilibrium_outputs(u, tau_w, tau_z, params, n=5):
    """
    From the equilibrium state u, extract household and firm outcomes.
    Returns consumption c, utilities, and other key variables.
    """
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(u, params, n)
    
    # Compute tax revenue and lump-sum transfer (l_val)
    tax_rev = 0.0
    for i in range(n):
        tax_rev += tau_w[i] * params['phi'][i] * w * (params['t_total'] - ell[i])
    tax_rev += (z_c + z_d) * tau_z
    l_val = (tax_rev - params['G']) / n
    
    # Back out household consumption from the budget constraint:
    c = np.empty(n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i]) * params['phi'][i] * w * (params['t_total'] - ell[i]) + l_val
        c[i] = (inc_i - p_d * d[i]) / params['p_c']
    
    # Compute household utilities: u_i = alpha ln(c_i) + beta ln(d_i-d0) + gamma ln(ell_i)
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = params['alpha'] * np.log(c[i]) + \
                         params['beta'] * np.log(d[i] - params['d0']) + \
                         params['gamma'] * np.log(ell[i])
    
    return c, utilities, l_val, d, ell, w, p_d, z_c, z_d

def build_params(tau_w, tau_z):
    """
    Constructs the parameter dictionary using the fixed model parameters
    and the decision variables tau_w and tau_z.
    """
    params = {
        'alpha': 0.7,
        'beta': 0.2,
        'gamma': 0.2,
        'd0': 0.1,
        'x': 100.0,
        'p_c': 1.0,
        'epsilon_c': 0.995,
        'epsilon_d': 0.92,
        'r': -1.0,
        'tau_z': tau_z,
        'tau_w': tau_w,
        'phi': phi_planner,    # use planner’s phi
        't_total': 1.0,
        'G': G
    }
    return params

# ---------------------------
# Optimization formulation
# ---------------------------
# Decision variables: x = [tau_w (5), tau_z (1), u (21)]
dim_u = 21
dim_policy = n + 1
dim_total = dim_policy + dim_u

def swf_obj(x):
    """
    Social welfare objective:
       Maximize sum_i {utility_i} - xi * (aggregate_pollution^theta)
    (We return negative welfare since we use a minimizer.)
    """
    tau_w = x[:n]
    tau_z = x[n]
    u = x[n+1:]
    
    params = build_params(tau_w, tau_z)
    _, utilities, _, _, _, w, _, z_c, z_d = compute_equilibrium_outputs(u, tau_w, tau_z, params, n)
    
    agg_polluting = z_c + z_d
    welfare = np.sum(utilities) - xi * (agg_polluting ** theta)
    return -welfare  # negative for minimization

def market_clearing_constraint(x):
    """
    Equality constraints enforcing market clearing.
    """
    tau_w = x[:n]
    tau_z = x[n]
    u = x[n+1:]
    
    params = build_params(tau_w, tau_z)
    return market_clearing(u, params, n)

def ic_constraints(x):
    """
    Mirrlees incentive compatibility constraints.
    For every pair (i,j) with i != j, require U_i - U_i^j >= 0.
    """
    tau_w = x[:n]
    tau_z = x[n]
    u = x[n+1:]
    
    params = build_params(tau_w, tau_z)
    c, utilities, l_val, d, ell, w, p_d, z_c, z_d = compute_equilibrium_outputs(u, tau_w, tau_z, params, n)
    
    T = params['t_total']
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j]) * (1.0 - tau_w[j]) * params['phi'][j] * w + l_val
    
    g_list = []
    for i in range(n):
        U_i = utilities[i]
        for j in range(n):
            if i == j:
                continue
            denom = (1.0 - tau_w[j]) * params['phi'][i] * w
            if denom <= 0:
                g_list.append(-1e6)
                continue
            ell_i_j = T - I[j] / denom
            if ell_i_j <= 0 or c[j] <= 0 or d[j] <= params['d0']:
                U_i_j = -1e6
            else:
                U_i_j = params['alpha'] * np.log(c[j]) + \
                        params['beta'] * np.log(d[j] - params['d0']) + \
                        params['gamma'] * np.log(ell_i_j)
            g_list.append(U_i - U_i_j)
    return np.array(g_list)

# ---------------------------
# Initial Guess and Bounds
# ---------------------------
# Initial guess for tax parameters (policy variables)
initial_tau_w = [0.05] * n
initial_tau_z = [0.5]
# Initial guess for equilibrium variables (same as in the original solver)
u0 = np.zeros(dim_u)
u0[0:5]   = np.log(0.99)      # for d_raw (recall: d = d0 + exp(d_raw))
u0[5:10]  = 0.0               # for ell_raw (implies ell ~ 0.5)
u0[10:15] = 1.0               # for lambda
u0[15]    = 0.0               # for t_c
u0[16]    = 0.0               # for t_d
u0[17]    = 0.0               # for ln(z_c)
u0[18]    = 0.0               # for ln(z_d)
u0[19]    = 0.0               # for ln(p_d)
u0[20]    = np.log(5.0)       # for ln(w)

x0 = np.concatenate([initial_tau_w, initial_tau_z, u0])

# Bounds:
# For tau_w in [0,1], tau_z in [0.01, 5.0], and no bounds for u (set to None)
bounds = [(0.0, 1.0)] * n + [(0.01, 5.0)] + [(None, None)] * dim_u

# ---------------------------
# Assemble constraints for the optimizer
# ---------------------------
constraints = [
    {'type': 'eq', 'fun': market_clearing_constraint},
    {'type': 'ineq', 'fun': ic_constraints}
]

# ---------------------------
# Solve the Planner Problem
# ---------------------------
res = minimize(swf_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if res.success:
    # Extract optimal policy and equilibrium state from the decision vector
    opt_tau_w = res.x[:n]
    opt_tau_z = res.x[n]
    opt_u     = res.x[n+1:]
    
    params = build_params(opt_tau_w, opt_tau_z)
    c, utilities, l_val, d, ell, w, p_d, z_c, z_d = compute_equilibrium_outputs(opt_u, opt_tau_w, opt_tau_z, params, n)
    max_welfare = -res.fun
    print("Social Welfare Optimization Successful!")
    print("Optimal tau_w:", opt_tau_w)
    print("Optimal tau_z:", opt_tau_z)
    print("Maximized Social Welfare:", max_welfare)
    print("")
    print("Equilibrium outcomes:")
    for i in range(n):
        print(f"Household {i+1}: consumption = {c[i]:.4f}, dirty consumption = {d[i]:.4f}, labor = {ell[i]:.4f}, utility = {utilities[i]:.4f}")
    print("")
    print("Wage (w):", w)
    print("Dirty good price (p_d):", p_d)
    print("Lump-sum transfer (l_val):", l_val)
    
    # Check market clearing residuals
    mc_res = market_clearing(opt_u, params, n)
    print("Market clearing residuals (should be near zero):", mc_res)
    
    # Check IC constraints (all should be nonnegative)
    ic_vals = ic_constraints(res.x)
    print("IC constraint values (should be >= 0):", ic_vals)
else:
    print("Optimization Failed!")
    print("Solver message:", res.message)