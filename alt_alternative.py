import numpy as np
import numba as nb
from scipy.optimize import fsolve  # You can still use broyden1 if you wish.

@nb.njit
def hh_utility(c, d, ell, alpha, beta, gamma, d0):
    """
    Utility = alpha * log(c) + beta * log(d - d0) + gamma * log(ell).
    Must have c>0, d>d0, ell>0 for logs to be valid.
    """
    return alpha * np.log(c) + beta * np.log(d - d0) + gamma * np.log(ell)

@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, tau_w, mult):
    
    # feasibility implies c>0, d>d0, ell>0. if invalid return large residuals
    if c <= 0.0 or (d <= d0) or ell <= 0.0:
        return np.array([1e6, 1e6, 1e6])

    foc_c   = (alpha / c) - mult * p
    foc_d   = (beta / (d - d0)) - mult * p
    foc_ell = (gamma / ell) - mult * (1.0 - tau_w) * w
    return np.array([foc_c, foc_d, foc_ell])

@nb.njit
def firm_production(t, z, epsilon, r):
    """
    CES production function: y = [ε * t^r + (1-ε) * z^r]^(1/r)
    """
    inside = epsilon * (t ** r) + (1.0 - epsilon) * (z ** r)
    if inside <= 0:
        return 0.0
    return inside ** (1.0 / r)

@nb.njit
def firm_focs(t, z, p, w, tau_z, epsilon, r):
    """
    Returns 2 firm FOCs:
      1) p*(∂y/∂t) - w = 0
      2) p*(∂y/∂z) - tau_z = 0
    """
    y = firm_production(t, z, epsilon, r)
    if y <= 0:
        return np.array([1e6, 1e6])
    
    # Derivatives for CES
    # y = [inside]^(1/r), inside = ε t^r + (1−ε) z^r
    # dy/dt = ε * t^(r−1) * y^(1−r)
    dy_dt = epsilon * (t ** (r - 1)) * (y ** (1.0 - r))
    dy_dz = (1.0 - epsilon) * (z ** (r - 1)) * (y ** (1.0 - r))
    
    foc_t = p * dy_dt - w
    foc_z = p * dy_dz - tau_z
    return np.array([foc_t, foc_z])

@nb.njit
def market_clearing(c, d, ell, t, z, p, w, epsilon, r):
    """
    Two conditions:
      1) Goods market: (c + d) - y(t,z) = 0
      2) Labor market: (1 - ell) - t = 0
    """
    y = firm_production(t, z, epsilon, r)
    f_goods = (c + d) - y
    f_labor = (20.0 - ell) - t
    return np.array([f_goods, f_labor])

@nb.njit
def budget_constraint(c, d, ell, w):
    """
    Single condition: c + d - w*(1 - ell) = 0
    """
    return c + d - w * (20.0 - ell)

#########################
# The full system (8 eq) #
#########################

def full_system(x, params):
    """
    x = [c, d, ell, t, z, p, w, mult]

    We have:
      - 3 Household FOCs
      - 2 Firm FOCs
      - 2 Market clearing eqs
      - 1 Budget constraint

    Return a vector of 8 residuals.
    """
    c, d, ell, t, z, p, w, mult = x
    
    alpha   = params['alpha']
    beta    = params['beta']
    gamma   = params['gamma']
    d0      = params['d0']
    tau_w   = params['tau_w']
    tau_z   = params['tau_z']
    epsilon = params['epsilon']
    r       = params['r']
    
    # Household FOCs
    hh_res = hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, tau_w, mult)
    
    # Firm FOCs
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)
    
    # Market clearing
    mkt_res = market_clearing(c, d, ell, t, z, p, w, epsilon, r)  # 2 eq
    # Budget constraint
    budg_res = np.array([budget_constraint(c, d, ell, w)])        # 1 eq
    
    # Stack all 8 residuals
    return np.concatenate((hh_res, firm_res, mkt_res, budg_res))

#######################
# Testing the system #
#######################

def test_full_system():
    """
    Use a better solver approach. We can try either:
      1) from scipy.optimize import fsolve
      2) from scipy.optimize import broyden1

    We'll show fsolve here for more typical usage.
    """
    # initial guess for x = [c, d, ell, t, z, p, w, mult]
    x0 = np.array([10.0,   # c
                   5.0,    # d
                   0.5,    # ell
                   0.2,    # t
                   10.0,   # z
                   2.0,    # p
                   3.0,    # w
                   1.0])   # mult
    
    # parameter dictionary
    params = {
        'alpha':   0.7,     # preference for clean good
        'beta':    0.4,     # preference for polluting good
        'gamma':   0.3,     # preference for leisure
        'd0':      1.0,     # subsistence for d
        'tau_w':   0.5,     # income tax
        'tau_z':   3.1,       # cost for z
        'epsilon': 0.5,     # CES parameter
        'r':       0.5
    }
    
    # Define the function to pass to the solver
    def F_for_solver(x):
        return full_system(x, params)
    
    # Solve
    sol = fsolve(F_for_solver, x0, xtol=1e-8, maxfev=5000)
    
    print("solution x = [c, d, ell, t, z, p, w, mult]:", sol)
    print("residuals at solution are", F_for_solver(sol))

if __name__ == "__main__":
    test_full_system()