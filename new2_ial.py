import numpy as np
import numba as nb
from scipy.optimize import root

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

################################################################################################
# 1) Modified Household FOCs (omitting c-FOC for household 1)
################################################################################################
@nb.njit
def households_focs_array_except_firstC(c, d, ell, p_c, p_d, w,
                                        alpha, beta, gamma, d0, mult, psi):
    """
    Returns 14 FOCs instead of 15, skipping the c-FOC for household 1.
    For i=0,...,n-1 => 5 HH total => we normally do:
      foc_c[i]   = alpha/c[i] - mult[i]*p_c
      foc_d[i]   = beta/(d[i]-d0) - mult[i]*p_d
      foc_ell[i] = gamma/ell[i]   - mult[i]*w*psi[i]
    We skip foc_c for i=0 (household 1).
    We'll keep all 3 for i=1..4, and keep foc_d/foc_ell for i=0.
    
    => total eq = 2 for i=0 + 3*(n-1) for i=1..n-1 = 2 + 3*4 = 14
    """
    n = c.shape[0]  # n=5
    # We'll store 14 residuals in a 1D array:
    res = np.empty(14)
    
    # i=0: skip c-FOC, keep d-FOC & ell-FOC
    idx = 0
    if (c[0] <= 0.0) or (d[0] <= d0) or (ell[0] <= 0.0) or (ell[0] >= 1.0):
        # Large penalty if out of bounds
        res[idx]   = 1e6
        res[idx+1] = 1e6
    else:
        foc_d   = beta  * (1.0/(d[0] - d0)) - mult[0] * p_d
        foc_ell = gamma * (1.0/ell[0])     - mult[0] * w * psi[0]
        res[idx]   = foc_d
        res[idx+1] = foc_ell
    idx += 2
    
    # i=1..4: keep all 3 FOCs
    for i in range(1, n):
        if (c[i] <= 0.0) or (d[i] <= d0) or (ell[i] <= 0.0) or (ell[i] >= 1.0):
            res[idx : idx+3] = np.array([1e6, 1e6, 1e6])
        else:
            foc_c   = alpha * (1.0/c[i]) - mult[i] * p_c
            foc_d   = beta  * (1.0/(d[i] - d0)) - mult[i] * p_d
            foc_ell = gamma * (1.0/ell[i])     - mult[i] * w * psi[i]
            res[idx : idx+3] = np.array([foc_c, foc_d, foc_ell])
        idx += 3
    
    return res

################################################################################################
# 2) Household Budget array (all 5 constraints)
################################################################################################
@nb.njit
def households_budget_array_all5(c, d, ell, p_c, p_d, w, psi, tau_z, z_d, z_c):
    """
    Now we do NOT drop any budget constraint => returns length n=5.
    Also we add a lumpsum transfer of 0.2*tau_z*(z_d+z_c) to each HH 
    just as in your code snippet. 
    (Or remove it if you want none).
    """
    n = c.shape[0]
    res = np.empty(n)
    lumpsum = tau_z*(z_d+z_c)*0.2  # 1/5 of total tax revenue for each HH
    for i in range(n):
        lhs = p_c*c[i] + p_d*d[i]
        rhs = w*(1.0 - ell[i])*psi[i] + lumpsum
        res[i] = lhs - rhs
    return res

################################################################################################
# 3) Firms, Production, FOCs (unchanged)
################################################################################################
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):
    inside = epsilon_c*(t_c**r) + (1.0 - epsilon_c)*(z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):
    inside = epsilon_d*(t_d**r) + (1.0 - epsilon_d)*(z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c * (t_c**(r - 1.0)) * (y_c**(1.0 - r))
    dy_dz_c = (1.0 - epsilon_c)*(z_c**(r - 1.0))*(y_c**(1.0 - r))
    foc_t_c = p_c*dy_dt_c - w
    foc_z_c = p_c*dy_dz_c - tau_z
    return np.array([foc_t_c, foc_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d*(t_d**(r - 1.0))*(y_d**(1.0 - r))
    dy_dz_d = (1.0 - epsilon_d)*(z_d**(r - 1.0))*(y_d**(1.0 - r))
    foc_t_d = p_d*dy_dt_d - w
    foc_z_d = p_d*dy_dz_d - tau_z
    return np.array([foc_t_d, foc_z_d])

################################################################################################
# 4) Market Clearing (3 eq)
################################################################################################
@nb.njit
def market_clearing(c, d, ell, t_c, z_c, t_d, z_d, p_c, p_d, w, eps_c, eps_d, r):
    y_c = firm_c_production(t_c, z_c, eps_c, r)
    y_d = firm_d_production(t_d, z_d, eps_d, r)
    eq_c = np.sum(c) - y_c
    eq_d = np.sum(d) - y_d
    eq_l = np.sum(ell) + t_c + t_d - c.shape[0]
    return np.array([eq_c, eq_d, eq_l])

################################################################################################
# 5) Transform Function (same as yours)
################################################################################################
def transform(u, d0, n=5):
    c   = np.exp(u[0:5])
    d   = d0 + np.exp(u[5:10])
    ell = 1.0/(1.0 + np.exp(-u[10:15]))
    mult= u[15:20]
    t_c = n*(1.0/(1.0 + np.exp(-u[20])))
    t_d = n*(1.0/(1.0 + np.exp(-u[21])))
    z_c = np.exp(u[22])
    z_d = np.exp(u[23])
    p_d = np.exp(u[24])
    w   = np.exp(u[25])
    return c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, p_c, w

################################################################################################
# 6) Full System: drop c-FOC for HH1, keep all budgets
################################################################################################
def full_system_skip_cFOC1(u, params, n=5):
    """
    We have:
      - (3n -1)=14 household FOCs (omitting c-FOC for HH1)
      - 2 firmC FOCs
      - 2 firmD FOCs
      - 3 market clearing
      - n=5 household budgets (no dropping)
    => total eq = 14 + 2 + 2 + 3 + 5 = 26
    """
    d0 = params['d0']
    c, d, ell, mult, t_c, t_d, z_c, z_d, p_c, p_d, w = transform(u, d0, n)
    
    #p_c   = params['p_c']
    alpha = params['alpha']
    beta  = params['beta']
    gamma = params['gamma']
    tau_z = params['tau_z']
    eps_c = params['epsilon_c']
    eps_d = params['epsilon_d']
    r     = params['r']
    psi   = params['psi']
    
    # 14 FOCs for households
    hh_focs14 = households_focs_array_except_firstC(
        c, d, ell, p_c, p_d, w, alpha, beta, gamma, d0, mult, psi
    )
    
    # 2 FOCs for firm C
    firmC = firm_c_focs(t_c, z_c, p_c, w, tau_z, eps_c, r)
    
    # 2 FOCs for firm D
    firmD = firm_d_focs(t_d, z_d, p_d, w, tau_z, eps_d, r)
    
    # 3 market clearing eq
    mc = market_clearing(c, d, ell, t_c, z_c, t_d, z_d, p_c, p_d, w, eps_c, eps_d, r)
    
    # 5 household budgets (all)
    hh_budg5 = households_budget_array_all5(c, d, ell, p_c, p_d, w, psi, tau_z, z_d, z_c)
    
    # Combine into one 26-vector
    return np.concatenate((hh_focs14, firmC, firmD, mc, hh_budg5))

################################################################################################
# 7) Solve & Print
################################################################################################
def main_solve_skip_cFOC1(tau_z=3, n=5):
    # Set params
    params = {
        'alpha': 0.7,
        'beta': 0.1,
        'gamma': 0.1,
        'd0': 0.01,
        'r': 0.5,
        'p_c': 1.0,
        'epsilon_c': 0.8,
        'epsilon_d': 0.6,
        'tau_z': tau_z
    }
    psi = np.array([0.5, 1.5, 2.0, 0.5, 0.5])
    params['psi'] = psi
    
    # 26 unknowns
    u0 = np.zeros(26)
    # typical guess
    u0[0:5]   = np.log(1.0)
    u0[5:10]  = np.log(1.0)
    u0[10:15] = 0.0
    u0[15:20] = 1.0
    u0[20]    = 0.0
    u0[21]    = 0.0
    u0[22]    = 0.0
    u0[23]    = 0.0
    u0[24]    = 0.0
    u0[25]    = 0.0
    
    # Solve
    sol = root(lambda x: full_system_skip_cFOC1(x, params, n), u0, method='lm', tol=1e-12)
    res = full_system_skip_cFOC1(sol.x, params, n)
    resid_norm = np.linalg.norm(res)
    
    # Transform final
    c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, p_c, w = transform(sol.x, params['d0'], n)
    
    # Compute omitted c-FOC for HH1
    # c-FOC: alpha*(1/c[0]) - mult[0]*p_c
    cFOC_HH1 = params['alpha']*(1.0/c[0]) - mult[0]*params['p_c']
    
    # Print results
    print(f"=== Solve with all 5 budgets, skipping c-FOC for HH1. tau_z={tau_z} ===")
    print("Success:", sol.success)
    print("Message:", sol.message)
    print(f"Final residual norm: {resid_norm:.2e}")
    
    # Print omitted c-FOC for HH1
    print(f"\nOMITTED c-FOC for HH1 => {cFOC_HH1:.8f}\n(This is not forced to zero!)")
    
    print("Residual vector (26 dims):", res)
    print()
    print("=== Household Stats ===")
    for i in range(n):
        print(f"HH {i+1}: c={c[i]:.4f}, d={d[i]:.4f}, ell={ell[i]:.4f}, mult={mult[i]:.4f}")
    print()
    print("=== Market / Firms ===")
    print(f"t_c={t_c:.4f}, t_d={t_d:.4f}, z_c={z_c:.4f}, z_d={z_d:.4f}")
    print(f"p_d={p_d:.4f}, w={w:.4f}")

if __name__=="__main__":
    main_solve_skip_cFOC1(tau_z=2, n=5)