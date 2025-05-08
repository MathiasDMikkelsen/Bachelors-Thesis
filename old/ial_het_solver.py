import numpy as np
import numba as nb
from scipy.optimize import root

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

###############################################################################
# 1) Households: FOCs + Budget Constraint
###############################################################################
@nb.njit
def hh1_focs(c1, d1, ell1, p_c, w, alpha, beta, gamma, d0, mult1, psi1):
    """
    Household 1 FOCs:
      alpha * c1^(alpha-1) = mult1 * p_c
      beta * (d1 - d0)^(beta-1) = mult1 * p_c
      gamma * ell1^(gamma-1) = mult1 * w * psi1
    Returns array of length 3.
    If infeasible => large residuals.
    """
    if (c1 <= 0.0) or (d1 <= d0) or (ell1 <= 0.0) or (ell1 >= 1.0):
        return np.array([1e6, 1e6, 1e6])
    foc_c1   = alpha * (c1**(alpha - 1.0)) - mult1 * p_c
    foc_d1   = beta  * ((d1 - d0)**(beta  - 1.0)) - mult1 * p_c
    foc_ell1 = gamma * (ell1**(gamma - 1.0)) - mult1 * w * psi1
    return np.array([foc_c1, foc_d1, foc_ell1])

@nb.njit
def hh2_focs(c2, d2, ell2, p_c, w, alpha, beta, gamma, d0, mult2, psi2):
    """
    Household 2 FOCs:
      alpha * c2^(alpha-1) = mult2 * p_c
      beta * (d2 - d0)^(beta-1) = mult2 * p_c
      gamma * ell2^(gamma-1) = mult2 * w * psi2
    Returns array of length 3.
    If infeasible => large residuals.
    """
    if (c2 <= 0.0) or (d2 <= d0) or (ell2 <= 0.0) or (ell2 >= 1.0):
        return np.array([1e6, 1e6, 1e6])
    foc_c2   = alpha * (c2**(alpha - 1.0)) - mult2 * p_c
    foc_d2   = beta  * ((d2 - d0)**(beta  - 1.0)) - mult2 * p_c
    foc_ell2 = gamma * (ell2**(gamma - 1.0)) - mult2 * w * psi2
    return np.array([foc_c2, foc_d2, foc_ell2])

@nb.njit
def hh1_budget_constraint(c1, d1, ell1, p_c, p_d, w):
    """
    Household 1 Budget (14th equation):
      p_c*c1 + p_d*d1 = w*(1 - ell1)
    => p_c=1 => c1 + p_d*d1 - w*(1-ell1) = 0
    """
    return p_c*c1 + p_d*d1 - w*(1.0 - ell1)

###############################################################################
# 2) Firms: Production & FOCs
###############################################################################
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):
    inside = epsilon_c*(t_c**r) + (1.0 - epsilon_c)*(z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):
    inside = epsilon_d*(t_d**r) + (1.0 - epsilon_d)*(z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    """
    2 eq:
      p_c * dy_c/dt_c - w = 0
      p_c * dy_c/dz_c - tau_z = 0
    """
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c*(t_c**(r - 1.0))*(y_c**(1.0 - r))
    dy_dz_c = (1.0 - epsilon_c)*(z_c**(r - 1.0))*(y_c**(1.0 - r))
    foc_t_c = p_c*dy_dt_c - w
    foc_z_c = p_c*dy_dz_c - tau_z
    return np.array([foc_t_c, foc_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    """
    2 eq:
      p_d * dy_d/dt_d - w = 0
      p_d * dy_d/dz_d - tau_z = 0
    """
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d*(t_d**(r - 1.0))*(y_d**(1.0 - r))
    dy_dz_d = (1.0 - epsilon_d)*(z_d**(r - 1.0))*(y_d**(1.0 - r))
    foc_t_d = p_d*dy_dt_d - w
    foc_z_d = p_d*dy_dz_d - tau_z
    return np.array([foc_t_d, foc_z_d])

###############################################################################
# 3) Market Clearing (3 eq)
###############################################################################
@nb.njit
def market_clearing(c1, d1, ell1,
                    c2, d2, ell2,
                    t_c, z_c,
                    t_d, z_d,
                    p_c, p_d, w,
                    epsilon_c, epsilon_d, r):
    """
    1) c1 + c2 = y_c
    2) d1 + d2 = y_d
    3) (ell1 + ell2) + (t_c + t_d) = 2  (2 households => total time=2)
    """
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    eq_c = (c1 + c2) - y_c
    eq_d = (d1 + d2) - y_d
    eq_l = (ell1 + ell2) + (t_c + t_d) - 2.0
    return np.array([eq_c, eq_d, eq_l])

###############################################################################
# 4) Transform: 14 Unknowns
###############################################################################
def transform(u, d0):
    """
    Indices in 'u':
      0) ln(c1)
      1) ln(c2)
      2) ln(d1 - d0)
      3) ln(d2 - d0)
      4) logistic => ell1 in (0,1)
      5) logistic => ell2 in (0,1)
      6) logistic => t_c in (0,1)
      7) logistic => t_d in (0,1)
      8) param => z_c>0
      9) param => z_d>0
      10) ln(p_d)
      11) ln(w)
      12) mult1
      13) mult2
    """
    c1   = np.exp(u[0])
    c2   = np.exp(u[1])
    d1   = d0 + np.exp(u[2])
    d2   = d0 + np.exp(u[3])
    ell1 = 1.0 / (1.0 + np.exp(-u[4]))
    ell2 = 1.0 / (1.0 + np.exp(-u[5]))
    t_c  = 1.0 / (1.0 + np.exp(-u[6]))
    t_d  = 1.0 / (1.0 + np.exp(-u[7]))
    z_c  = 100.0 / (1.0 + np.exp(-u[8]))
    z_d  = 100.0 / (1.0 + np.exp(-u[9]))
    p_d  = np.exp(u[10])
    w    = np.exp(u[11])
    mult1= u[12]
    mult2= u[13]

    return (c1, c2, d1, d2, ell1, ell2, t_c, t_d, z_c, z_d, p_d, w, mult1, mult2)

###############################################################################
# 5) Full System: 14 eq => no dimension mismatch
###############################################################################
def full_system_transformed(u, params):
    """
    Builds 14 equations:
      - 3 HH1 FOCs
      - 3 HH2 FOCs
      - 2 Firm C FOCs
      - 2 Firm D FOCs
      - 3 Market clearing
      - 1 HH1 budget
    """
    (c1, c2, d1, d2, ell1, ell2,
     t_c, t_d, z_c, z_d, p_d, w,
     mult1, mult2) = transform(u, params['d0'])

    # Unpack
    p_c       = params['p_c']
    alpha     = params['alpha']
    beta      = params['beta']
    gamma     = params['gamma']
    tau_z     = params['tau_z']
    epsilon_c = params['epsilon_c']
    epsilon_d = params['epsilon_d']
    r         = params['r']
    psi1      = params['psi1']
    psi2      = params['psi2']

    # 1) Households
    hh1_res = hh1_focs(c1, d1, ell1, p_c, w, alpha, beta, gamma, params['d0'], mult1, psi1)
    hh2_res = hh2_focs(c2, d2, ell2, p_c, w, alpha, beta, gamma, params['d0'], mult2, psi2)

    # 2) Firms
    firm_c_res = firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r)
    firm_d_res = firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r)

    # 3) Market clearing
    mkt_res = market_clearing(
        c1, d1, ell1,
        c2, d2, ell2,
        t_c, z_c,
        t_d, z_d,
        p_c, p_d, w,
        epsilon_c, epsilon_d, r
    )

    # 4) Budget for HH1
    bgt1 = hh1_budget_constraint(c1, d1, ell1, p_c, p_d, w)

    # Return 14 equations total
    return np.concatenate((hh1_res, hh2_res, firm_c_res, firm_d_res, mkt_res, [bgt1]))

###############################################################################
# 6) Main Solver
###############################################################################
def find_feasible_tau_z():
    """
    We now have 14 unknowns and 14 eq => method='lm' can work.
    We vary tau_z to see if the system converges.
    """
    base_params = {
        'alpha':     0.7,
        'beta':      0.2,
        'gamma':     0.2,
        'd0':        0.1,
        'r':         0.5,
        'p_c':       1.0,  # numeraire
        'epsilon_c': 0.9,
        'epsilon_d': 0.9,
        'psi1':      0.5,
        'psi2':      0.5
    }

    # Range of tau_z to test
    tau_z_values = np.linspace(1, 20, 50)

    # 14 unknowns => improved initial guess
    # ln(c1)=ln(1.0)=0, ln(c2)=ln(2)=0.693..., ln(d1-d0)=ln(1.0)=0, ln(d2-d0)=ln(1.5)=~0.405
    # logistic param ~ 0 => ~0.5, logistic param ~1 => ~0.73, etc.
    # We do a mix so they're in the feasible interior
    u0 = np.array([
        np.log(1.0),    # ln(c1)
        np.log(2.0),    # ln(c2)
        np.log(1.0),    # ln(d1 - d0)=  ln(1.0)=0
        np.log(1.5),    # ln(d2 - d0)= ~0.405
        0.0,            # logistic => ell1 ~0.5
        1.0,            # logistic => ell2 ~0.73
        0.0,            # logistic => t_c ~0.5
        0.0,            # logistic => t_d ~0.5
        1.0,            # param => z_c => ~ 100/(1+e^-1)= ~27
        2.0,            # param => z_d => ~ 100/(1+ e^-2)= ~ 12
        np.log(2.0),    # ln(p_d)= ~0.693 => p_d=2
        np.log(1.0),    # ln(w)=0 => w=1
        1.0,            # mult1
        1.0             # mult2
    ])

    feasible_solutions = []

    for tau_z in tau_z_values:
        params = base_params.copy()
        params['tau_z'] = tau_z

        sol = root(lambda x: full_system_transformed(x, params),
                   u0, method='lm', tol=1e-8)

        res_vec = full_system_transformed(sol.x, params)
        resid_norm = np.linalg.norm(res_vec)

        if sol.success and resid_norm < 1e-6:
            (c1, c2, d1, d2,
             ell1, ell2, t_c, t_d,
             z_c, z_d, p_d, w,
             mult1, mult2) = transform(sol.x, params['d0'])

            feasible_solutions.append({
                'tau_z': tau_z,
                'c1': c1, 'c2': c2,
                'd1': d1, 'd2': d2,
                'ell1': ell1, 'ell2': ell2,
                't_c': t_c, 't_d': t_d,
                'z_c': z_c, 'z_d': z_d,
                'p_d': p_d, 'w': w,
                'mult1': mult1, 'mult2': mult2,
                'resid': resid_norm
            })
            print(f"** Converged at tau_z={tau_z:.2f}, residual={resid_norm:.2e}")
        else:
            print(f"No converge at tau_z={tau_z:.2f}, resid={resid_norm:.2e}, success={sol.success}")

    if feasible_solutions:
        print("\n=== Feasible Solutions ===")
        for fs in feasible_solutions:
            print(fs)
    else:
        print("No feasible solutions in tested range.")

if __name__ == '__main__':
    find_feasible_tau_z()