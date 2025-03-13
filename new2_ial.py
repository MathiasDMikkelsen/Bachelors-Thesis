import numpy as np                                                      
from scipy.optimize import root, minimize_scalar
import importlib

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

##############################################################################################################################################################
# 1) Household Functions (vectorized over n households)
##############################################################################################################################################################
def households_focs_array(c, d, ell, p_c, p_d, w,
                           alpha, beta, gamma, d0, mult, psi):
    n = c.shape[0]
    res = np.empty(3 * n)
    for i in range(n):
        if (c[i] <= 0.0) or (d[i] <= d0) or (ell[i] <= 0.0) or (ell[i] >= 1.0):
            res[3*i : 3*i+3] = np.array([1e6, 1e6, 1e6])
        else:
            foc_c   = alpha * 1/c[i] - mult[i] * p_c
            foc_d   = beta  * 1/(d[i] - d0) - mult[i] * p_d
            foc_ell = gamma * 1/ell[i] - mult[i] * w * psi[i]
            res[3*i : 3*i+3] = np.array([foc_c, foc_d, foc_ell])
    return res

def households_budget_array(c, d, ell, p_c, p_d, w, psi, tau_z, z_d, z_c):
    n = c.shape[0]
    res = np.empty(n)
    for i in range(n):
        res[i] = p_c * c[i] + p_d * d[i] - (w * (1.0 - ell[i]) * psi[i] + tau_z*(z_d+z_c)*0.2)
    return res

##############################################################################################################################################################
# 2) Firms: Production & FOCs
##############################################################################################################################################################
def firm_c_production(t_c, z_c, epsilon_c, r):
    inside = epsilon_c * (t_c**r) + (1.0 - epsilon_c) * (z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)

def firm_d_production(t_d, z_d, epsilon_d, r):
    inside = epsilon_d * (t_d**r) + (1.0 - epsilon_d) * (z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)

def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c * (t_c**(r - 1.0)) * (y_c**(1.0 - r))
    dy_dz_c = (1.0 - epsilon_c) * (z_c**(r - 1.0)) * (y_c**(1.0 - r))
    foc_t_c = p_c * dy_dt_c - w
    foc_z_c = p_c * dy_dz_c - tau_z
    return np.array([foc_t_c, foc_z_c])

def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d * (t_d**(r - 1.0)) * (y_d**(1.0 - r))
    dy_dz_d = (1.0 - epsilon_d) * (z_d**(r - 1.0)) * (y_d**(1.0 - r))
    foc_t_d = p_d * dy_dt_d - w
    foc_z_d = p_d * dy_dz_d - tau_z
    return np.array([foc_t_d, foc_z_d])

##############################################################################################################################################################
# 3) Market Clearing (3 Equations)
##############################################################################################################################################################
def market_clearing(c, d, ell, t_c, z_c, t_d, z_d, p_c, p_d, w, epsilon_c, epsilon_d, r):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    eq_c = np.sum(c) - y_c
    eq_d = np.sum(d) - y_d
    eq_l = np.sum(ell) + t_c + t_d - c.shape[0]
    return np.array([eq_c, eq_d, eq_l])

##############################################################################################################################################################
# 4) Transformation Function: 26 Unknowns for 5 Households
##############################################################################################################################################################
def transform(u, d0, n=5):
    c    = np.exp(u[0:5])
    d    = d0 + np.exp(u[5:10])
    ell  = 1.0 / (1.0 + np.exp(-u[10:15]))
    mult = u[15:20]
    t_c  = n * (1.0 / (1.0 + np.exp(-u[20])))
    t_d  = n * (1.0 / (1.0 + np.exp(-u[21])))
    z_c  = np.exp(u[22])
    z_d  = np.exp(u[23])
    p_d  = np.exp(u[24])
    w    = np.exp(u[25])
    return c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w

##############################################################################################################################################################
# 5) Full System Function: 26 Equations
##############################################################################################################################################################
def full_system(u, params, n=5):
    d0 = params['d0']
    (c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w) = transform(u, d0, n)
    p_c = params['p_c']
    alpha     = params['alpha']
    beta      = params['beta']
    gamma     = params['gamma']
    tau_z     = params['tau_z']
    epsilon_c = params['epsilon_c']
    epsilon_d = params['epsilon_d']
    r         = params['r']
    psi       = params['psi']
    
    hh_focs = households_focs_array(c, d, ell, p_c, p_d, w,
                                    alpha, beta, gamma, d0, mult, psi)
    firmC = firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r)
    firmD = firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r)
    mkt = market_clearing(c, d, ell, t_c, z_c, t_d, z_d, p_c, p_d, w, epsilon_c, epsilon_d, r)
    hh_budg = households_budget_array(c, d, ell, p_c, p_d, w, psi, tau_z, z_d, z_c)
    hh_budg = hh_budg[1:]
    
    return np.concatenate((hh_focs, firmC, firmD, mkt, hh_budg))

##############################################################################################################################################################
# 6) Main Solve Function (Modified to Return Convergence Info)
##############################################################################################################################################################
def main_solve_for_tau(tau_z, n):
    """
    Solve the full system for a given tau_z with n households.
    Returns a tuple:
      (utilities, aggregate_polluting, converged)
    where:
      - utilities is the utility vector (one per household).
      - aggregate_polluting is the sum of polluting good consumption.
      - converged is True if the root finder converged, False otherwise.
      
    Utility is computed as:
      U_i = α * ln(c_i) + β * ln(d_i - d0) + γ * ln(ell_i)
    """
    params = {
        'alpha':     0.7,
        'beta':      0.1,
        'gamma':     0.1,
        'd0':        0.01,
        'r':         0.5,
        'p_c':       1.0,
        'epsilon_c': 0.8,
        'epsilon_d': 0.6,
        'tau_z':     tau_z
    }
    psi = np.array([0.50, 1.50, 2.0, 0.50, 0.50])
    params['psi'] = psi

    u0 = np.zeros(26)
    u0[0:5] = np.log(1.0)
    u0[5:10] = np.log(1.0)
    u0[10:15] = 0.0
    u0[15:20] = 1.0
    u0[20] = 0.0
    u0[21] = 0.0
    u0[22] = 0.0
    u0[23] = 0.0
    u0[24] = np.log(1.0)
    u0[25] = np.log(1.0)

    sol = root(lambda x: full_system(x, params, n), u0, method='lm', tol=1e-15)
    converged = sol.success

    final_res = full_system(sol.x, params, n)
    resid_norm = np.linalg.norm(final_res)
    c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w = transform(sol.x, params['d0'], n)

    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = (params['alpha'] * np.log(c[i]) +
                        params['beta']  * np.log(d[i] - params['d0']) +
                        params['gamma'] * np.log(ell[i]))

    aggregate_polluting = np.sum(d)

    # Print details of the first (root-finder) optimizer
    print("\n=== Households: Utility and Polluting Good Consumption ===")
    for i in range(n):
        print(f"Household {i+1}: Utility = {utilities[i]:.4f}, Polluting Good Consumption = {d[i]:.4f}")
    print(f"\nAggregate Polluting Good Consumption = {aggregate_polluting:.4f}")
    print(f"First Optimizer Convergence: {converged}, Residual Norm: {resid_norm:.2e}")

    return utilities, aggregate_polluting, converged