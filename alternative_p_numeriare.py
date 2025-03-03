import numpy as np
import numba as nb
from scipy.optimize import differential_evolution
np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

###############################################################################
# 1) MODEL FUNCTIONS (Power Utility, CES Production, Market Clearing)
###############################################################################

@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult):
    """
    Household with power utility: U = c^α + (d-d0)^β + ell^γ.
    FOCs:
      α·c^(α-1) = mult · p,
      β·(d-d0)^(β-1) = mult · p,
      γ·ell^(γ-1) = mult · w.
    Feasibility: c>0, d>d0, 0 < ell < 1.  
    Returns large residuals if any are violated.
    """
    if c <= 0.0 or d <= d0 or ell <= 0.0 or ell >= 1.0:
        return np.array([1e6, 1e6, 1e6])
    foc_c = alpha * (c**(alpha-1)) - mult * p
    foc_d = beta * ((d-d0)**(beta-1)) - mult * p
    foc_ell = gamma * (ell**(gamma-1)) - mult * w
    return np.array([foc_c, foc_d, foc_ell])

@nb.njit
def firm_production(t, z, epsilon, r):
    """
    CES production: y = [ε·t^r + (1-ε)·z^r]^(1/r).
    Returns 0 if the inside is nonpositive.
    """
    inside = epsilon*(t**r) + (1.0 - epsilon)*(z**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_focs(t, z, p, w, tau_z, epsilon, r):
    """
    Firm FOCs:
      p * (dy/dt) - w = 0,
      p * (dy/dz) - τ₍z₎ = 0.
    """
    y = firm_production(t, z, epsilon, r)
    if y <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt = epsilon*(t**(r-1))*(y**(1.0-r))
    dy_dz = (1.0 - epsilon)*(z**(r-1))*(y**(1.0-r))
    foc_t = p * dy_dt - w
    foc_z = p * dy_dz - tau_z
    return np.array([foc_t, foc_z])

@nb.njit
def market_clearing(c, d, ell, t, z, p, w, epsilon, r):
    """
    Market Clearing:
      Goods: (c + d) - y(t,z) = 0,
      Labor: (1 - ell) - t = 0  (i.e. t = 1 - ell).
    """
    y = firm_production(t, z, epsilon, r)
    f_goods = (c + d) - y
    f_labor = (1.0 - ell) - t
    return np.array([f_goods, f_labor])

###############################################################################
# 2) FULL SYSTEM OF EQUATIONS
###############################################################################
def full_system(x, params):
    """
    x = [c, d, ell, t, z, w, mult]
    The system consists of:
      3 household FOCs,
      2 firm FOCs,
      2 market clearing equations.
    p is now taken from params (i.e. the chosen numeriare).
    Returns a 7-dimensional residual vector.
    """
    c, d, ell, t, z, w, mult = x
    p = params['p']  # numeriare is now user-specified

    alpha   = params['alpha']
    beta    = params['beta']
    gamma   = params['gamma']
    d0      = params['d0']
    tau_z   = params['tau_z']
    epsilon = params['epsilon']
    r       = params['r']

    hh_res   = hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult)  # 3 eq.
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)                 # 2 eq.
    mkt_res  = market_clearing(c, d, ell, t, z, p, w, epsilon, r)         # 2 eq.
    # Budget constraint removed

    return np.concatenate((hh_res, firm_res, mkt_res))

###############################################################################
# 3) OBJECTIVE FUNCTION: Sum of Squared Residuals
###############################################################################
def objective(x, params):
    res = full_system(x, params)
    return np.sum(res**2)

###############################################################################
# 4) MAIN: Global Minimization via Differential Evolution and Equilibrium Profit
###############################################################################
def main():
    # You can choose the numeriare value here.
    numeriare_value = 3.0  # change this value as desired

    # Parameter dictionary
    params = {
        'alpha':   0.7,
        'beta':    0.2,
        'gamma':   0.2,
        'd0':      0.5,
        'tau_z':   1,    # tax parameter on z
        'epsilon': 0.6,
        'r':       0.5,
        'p':       numeriare_value  # numeriare value chosen by the user
    }

    # We'll solve for x = [c, d, ell, t, z, w, mult].
    # Set bounds that are generous but keep variables in reasonable ranges.
    bounds = [
        (1e-6, 100.0),   # c
        (0.1, 100.0),    # d (d must be > d0=0.5)
        (1e-6, 0.9999),  # ell in (0,1)
        (1e-6, 1.0),     # t in (0,1) because t should equal 1-ell
        (1e-6, 100.0),   # z
        (1e-6, 100.0),   # w
        (1e-6, 20.0)     # mult
    ]

    # Define the objective function for differential evolution
    def f_obj(x):
        return objective(x, params)

    # Run the global optimizer
    result = differential_evolution(
        f_obj,
        bounds,
        strategy='best1bin',
        maxiter=2000,
        popsize=30,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True
    )
    print("\n=== Differential Evolution Result ===")
    print("Converged:", result.success)
    print("Message: ", result.message)
    print("Minimum sum-of-squared errors:", result.fun)
    x_sol = result.x
    print("Solution x = [c, d, ell, t, z, w, mult]:\n", x_sol)
    res_final = full_system(x_sol, params)
    print("Final residuals:\n", res_final)

    # Compute equilibrium output and profit from the firm
    c, d, ell, t, z, w, mult = x_sol
    p = params['p']
    epsilon = params['epsilon']
    r = params['r']
    tau_z = params['tau_z']

    y = firm_production(t, z, epsilon, r)
    # Equilibrium profit: Revenue minus cost of labor and z
    profit = p * y - w * t - tau_z * z
    print("Equilibrium output (y): {:.8f}".format(y))
    print("Equilibrium profit: {:.8f}".format(profit))

if __name__=="__main__":
    main()