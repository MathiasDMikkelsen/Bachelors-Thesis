import numpy as np
import numba as nb
from scipy.optimize import root

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

###############################################################################
# 1) Households: FOCs + Budget Constraint
###############################################################################
@nb.njit
def hh1_focs(c1, d1, ell1, p_c, w, alpha, beta, gamma, d0, mult1, psi1):
    if (c1 <= 0.0) or (d1 <= d0) or (ell1 <= 0.0) or (ell1 >= 1.0):
        return np.array([1e6, 1e6, 1e6])
    foc_c1   = alpha * (c1**(alpha - 1.0)) - mult1 * p_c
    foc_d1   = beta  * ((d1 - d0)**(beta  - 1.0)) - mult1 * p_c
    foc_ell1 = gamma * (ell1**(gamma - 1.0)) - mult1 * w * psi1
    return np.array([foc_c1, foc_d1, foc_ell1])

@nb.njit
def hh2_focs(c2, d2, ell2, p_c, w, alpha, beta, gamma, d0, mult2, psi2):
    if (c2 <= 0.0) or (d2 <= d0) or (ell2 <= 0.0) or (ell2 >= 1.0):
        return np.array([1e6, 1e6, 1e6])
    foc_c2   = alpha * (c2**(alpha - 1.0)) - mult2 * p_c
    foc_d2   = beta  * ((d2 - d0)**(beta  - 1.0)) - mult2 * p_c
    foc_ell2 = gamma * (ell2**(gamma - 1.0)) - mult2 * w * psi2
    return np.array([foc_c2, foc_d2, foc_ell2])

@nb.njit
def hh1_budget_constraint(c1, d1, ell1, p_c, p_d, w):
    # p_c*c1 + p_d*d1 = w*(1 - ell1)
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
      4) logistic => ell1
      5) logistic => ell2
      6) logistic => t_c
      7) logistic => t_d
      8) param => z_c>0  (you can set exp(u[8]) or logistic, whichever you prefer)
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
    # For unbounded pollution, you might prefer z_c = exp(u[8]), etc.
    # Currently set to 100/(1+exp(-u[8])) => bounded
    z_c = np.exp(u[8])
    z_d = np.exp(u[9])
    p_d  = np.exp(u[10])
    w    = np.exp(u[11])
    mult1= u[12]
    mult2= u[13]

    return (c1, c2, d1, d2, ell1, ell2, t_c, t_d, z_c, z_d, p_d, w, mult1, mult2)

###############################################################################
# 5) Full System
###############################################################################
def full_system_transformed(u, params):
    (c1, c2, d1, d2, ell1, ell2,
     t_c, t_d, z_c, z_d, p_d, w,
     mult1, mult2) = transform(u, params['d0'])

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

    hh1_res = hh1_focs(c1, d1, ell1, p_c, w, alpha, beta, gamma, params['d0'], mult1, psi1)
    hh2_res = hh2_focs(c2, d2, ell2, p_c, w, alpha, beta, gamma, params['d0'], mult2, psi2)
    firm_c_res = firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r)
    firm_d_res = firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r)
    mkt_res = market_clearing(c1, d1, ell1, c2, d2, ell2, t_c, z_c,
                              t_d, z_d, p_c, p_d, w, epsilon_c, epsilon_d, r)
    bgt1 = hh1_budget_constraint(c1, d1, ell1, p_c, p_d, w)

    return np.concatenate((hh1_res, hh2_res, firm_c_res, firm_d_res, mkt_res, [bgt1]))

###############################################################################
# 6) Single-Solve Main
###############################################################################
def main_solve_for_tau(tau_z=5.0):
    """
    Solve the system for a given tau_z, print:
      - success, message
      - final residual norm
      - solution (c1, c2, ..., mult2)
      - FOC errors (household, firm, market clearing, budget)
      - profits
    """
    # Base parameters
    params = {
        'alpha':     0.7,
        'beta':      0.2,
        'gamma':     0.2,
        'd0':        0.1,
        'r':         0.5,
        'p_c':       1.0,   # numeraire
        'epsilon_c': 0.995,
        'epsilon_d': 0.92,
        'psi1':      0.2,
        'psi2':      0.8,
        'tau_z':     tau_z
    }

    # 14-dimensional initial guess
    u0 = np.array([
        np.log(1.0),   # ln(c1)
        np.log(2.0),   # ln(c2)
        np.log(1.0),   # ln(d1 - d0)
        np.log(1.5),   # ln(d2 - d0)
        0.0,           # logistic => ell1
        1.0,           # logistic => ell2
        0.0,           # logistic => t_c
        0.0,           # logistic => t_d
        1.0,           # param => z_c
        2.0,           # param => z_d
        np.log(2.0),   # ln(p_d)
        np.log(1.0),   # ln(w)
        1.0,           # mult1
        1.0            # mult2
    ])

    # Solve once with method='lm'
    sol = root(lambda x: full_system_transformed(x, params),
               u0, method='lm', tol=1e-8)

    # Evaluate final residual vector
    final_res = full_system_transformed(sol.x, params)
    resid_norm = np.linalg.norm(final_res)

    # Print solver outcome
    print(f"=== Solving for tau_z={tau_z:.4f} ===")
    print("Success:", sol.success)
    print("Message:", sol.message)
    print(f"Final residual norm: {resid_norm:.2e}")

    # Transform solution
    (c1, c2, d1, d2,
     ell1, ell2, t_c, t_d,
     z_c, z_d, p_d, w,
     mult1, mult2) = transform(sol.x, params['d0'])

    print("\n--- Endogenous Variables ---")
    print(f"c1={c1:.4f}, c2={c2:.4f}, d1={d1:.4f}, d2={d2:.4f}")
    print(f"ell1={ell1:.4f}, ell2={ell2:.4f}")
    print(f"t_c={t_c:.4f}, t_d={t_d:.4f}")
    print(f"z_c={z_c:.4f}, z_d={z_d:.4f}")
    print(f"p_d={p_d:.4f}, w={w:.4f}")
    print(f"mult1={mult1:.4f}, mult2={mult2:.4f}")

    # Print FOC errors block by block
    print("\n--- FOC/Constraint Residuals ---")
    # The final_res has length 14 => 0..2=HH1(3), 3..5=HH2(3), 6..7=firmC(2), 8..9=firmD(2), 10..12=mkt(3), 13=budget(1)
    hh1_res = final_res[0:3]
    hh2_res = final_res[3:6]
    firmC_res = final_res[6:8]
    firmD_res = final_res[8:10]
    mkt_res = final_res[10:13]
    bgt_res = final_res[13]

    print(f"HH1 FOCs: {hh1_res}")
    print(f"HH2 FOCs: {hh2_res}")
    print(f"Firm C FOCs: {firmC_res}")
    print(f"Firm D FOCs: {firmD_res}")
    print(f"Market clearing: {mkt_res}")
    print(f"HH1 budget: {bgt_res:.4e}")

    # Compute profits for each firm
    y_c = firm_c_production(t_c, z_c, params['epsilon_c'], params['r'])
    y_d = firm_d_production(t_d, z_d, params['epsilon_d'], params['r'])
    # Profit: revenue - labor cost - tau_z * price_of_that_input * z
    # For firm C: p_c=1 => profitC=1*y_c - w*t_c - 1*tau_z*z_c
    profit_c = params['p_c']*y_c - w*t_c - params['tau_z']*z_c
    # For firm D: profitD= p_d*y_d - w*t_d - p_d*tau_z*z_d
    profit_d = p_d*y_d - w*t_d - params['tau_z']*z_d

    print("\n--- Firm Profits ---")
    print(f"Firm C profit: {profit_c:.4f}")
    print(f"Firm D profit: {profit_d:.4f}")


if __name__ == "__main__":
    # Example usage: solve for tau_z=5
    main_solve_for_tau(tau_z=4)