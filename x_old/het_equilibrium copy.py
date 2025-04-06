import numpy as np
import numba as nb
from scipy.optimize import differential_evolution
np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

##############################################################################################################################################################
# 1) DEFINE THE MODEL FUNCTIONS (Household FOCs, Firm Production and FOCs, Market Clearing of Goods and Labor)
##############################################################################################################################################################

@nb.njit
def hh1_focs(c1, d1, ell1, p_c, p_d, w, alpha, beta, gamma, d0, mult1, psi1):
    """
    Household 1 FOCs.
    """
    if c1 <= 0.0 or d1 <= d0 or ell1 <= 0.0 or ell1 >= 1.0:
        return np.array([1e6, 1e6, 1e6])
    foc_c1 = alpha * (c1**(alpha-1)) - mult1 * p_c
    foc_d1 = beta * ((d1-d0)**(beta-1)) - mult1 * p_d
    foc_ell1 = gamma * (ell1**(gamma-1)) - mult1 * w * psi1
    return np.array([foc_c1, foc_d1, foc_ell1])

@nb.njit
def hh2_focs(c2, d2, ell2, p_c, p_d, w, alpha, beta, gamma, d0, mult2, psi2):
    """
    Household 2 FOCs.
    """
    if c2 <= 0.0 or d2 <= d0 or ell2 <= 0.0 or ell2 >= 1.0:
        return np.array([1e6, 1e6, 1e6])
    foc_c2 = alpha * (c2**(alpha-1)) - mult2 * p_c
    foc_d2 = beta * ((d2-d0)**(beta-1)) - mult2 * p_d
    foc_ell2 = gamma * (ell2**(gamma-1)) - mult2 * w * psi2
    return np.array([foc_c2, foc_d2, foc_ell2])

@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):
    """
    Firm C production function.
    """
    inside = epsilon_c*(t_c**r) + (1.0 - epsilon_c)*(z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):
    """
    Firm D production function.
    """
    inside = epsilon_d*(t_d**r) + (1.0 - epsilon_d)*(z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    """
    Firm C FOCs.
    """
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c*(t_c**(r-1))*(y_c**(1.0-r))
    dy_dz_c = (1.0 - epsilon_c)*(z_c**(r-1))*(y_c**(1.0-r))
    foc_t_c = p_c * dy_dt_c - w
    foc_z_c = p_c * dy_dz_c - tau_z
    return np.array([foc_t_c, foc_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    """
    Firm D FOCs.
    """
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d*(t_d**(r-1))*(y_d**(1.0-r))
    dy_dz_d = (1.0 - epsilon_d)*(z_d**(r-1))*(y_d**(1.0-r))
    foc_t_d = p_d * dy_dt_d - w
    foc_z_d = p_d * dy_dz_d - tau_z
    return np.array([foc_t_d, foc_z_d])

@nb.njit
def market_clearing(c1, d1, ell1, c2, d2, ell2, t_c, z_c, t_d, z_d, p_c, p_d, w, epsilon_c, epsilon_d, r):
    """
    Market clearing conditions.
    """
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    f_goods = (c1 + d1 + c2 + d2) - (y_c + y_d)
    f_labor = (2.0 - ell1 - ell2) - (t_c + t_d)
    return np.array([f_goods, f_labor])

##############################################################################################################################################################
# 2) CREATE THE SYSTEM OF EQUATIONS
##############################################################################################################################################################
def full_system(x, params):
    """
    System of equations with two households and two firms.
    """
    c1, d1, ell1, mult1, c2, d2, ell2, mult2, t_c, z_c, t_d, z_d, p_d, w = x
    p_c = params['p_c']

    alpha   = params['alpha']
    beta    = params['beta']
    gamma   = params['gamma']
    d0      = params['d0']
    tau_z   = params['tau_z']
    epsilon_c = params['epsilon_c']
    epsilon_d = params['epsilon_d']
    r       = params['r']
    psi1     = params['psi1']
    psi2     = params['psi2']

    hh1_res   = hh1_focs(c1, d1, ell1, p_c, w, alpha, beta, gamma, d0, mult1, psi1)
    hh2_res   = hh2_focs(c2, d2, ell2, p_c, w, alpha, beta, gamma, d0, mult2, psi2)
    firm_c_res = firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r)
    firm_d_res = firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r)
    mkt_res  = market_clearing(c1, d1, ell1, c2, d2, ell2, t_c, z_c, t_d, z_d, p_c, p_d, w, epsilon_c, epsilon_d, r)

    return np.concatenate((hh1_res, hh2_res, firm_c_res, firm_d_res, mkt_res))

##############################################################################################################################################################
# 3) DEFINE OBJECTIVE FUNCTION (Sum of Squared Residuals)
##############################################################################################################################################################
def objective(x, params):
    """
    Objective function.
    """
    res = full_system(x, params)
    return np.sum(res**2)

##############################################################################################################################################################
# 4) DEFINE MAIN FUNCTION THAT FINDS THE EQUILIBRIUM USING DIFFERENTIAL EVOLUTION
##############################################################################################################################################################
def main():
    """
    Main function.
    """
    numeriare_value = 1

    params = {
        'alpha':   0.5,
        'beta':    0.5,
        'gamma':   0.5,
        'd0':      0.1,
        'tau_z':   0.2,
        'epsilon_c': 0.65,
        'epsilon_d': 0.35,
        'r':       0.5,
        'p_c':     numeriare_value,
        'psi1':    0.2,
        'psi2':    0.8
    }

    bounds = [
        (1e-6, 100.0),   # c1
        (params['d0'], 100.0),  # d1
        (1e-6, 0.9999),  # ell1
        (1e-6, 20.0),    # mult1
        (1e-6, 100.0),   # c2
        (params['d0'], 100.0),  # d2
        (1e-6, 0.9999),  # ell2
        (1e-6, 20.0),    # mult2
        (1e-6, 2.0),     # t_c
        (1e-6, 100.0),   # z_c
        (1e-6, 2.0),     # t_d
        (1e-6, 100.0),   # z_d
        (1e-6, 2.0),     # p_d
        (1e-6, 100.0)    # w
    ]

    def f_obj(x):
        return objective(x, params)

    result = differential_evolution(
        f_obj,
        bounds,
        strategy='best1bin',
        maxiter=10000,
        popsize=20,
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
    c1, d1, ell1, mult1, c2, d2, ell2, mult2, t_c, z_c, t_d, z_d, p_d, w = x_sol

    print("\n--- Endogenous Variables ---")
    print(f"c1: {c1:.8f}")
    print(f"d1: {d1:.8f}")
    print(f"ell1: {ell1:.8f}")
    print(f"mult1: {mult1:.8f}")
    print(f"c2: {c2:.8f}")
    print(f"d2: {d2:.8f}")
    print(f"ell2: {ell2:.8f}")
    print(f"mult2: {mult2:.8f}")
    print(f"t_c: {t_c:.8f}")
    print(f"z_c: {z_c:.8f}")
    print(f"t_d: {t_d:.8f}")
    print(f"z_d: {z_d:.8f}")
    print(f"p_d: {p_d:.8f}")
    print(f"w: {w:.8f}")

    p_c = params['p_c']
    epsilon_c = params['epsilon_c']
    epsilon_d = params['epsilon_d']
    r = params['r']
    tau_z = params['tau_z']

    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    profit_c = p_c * y_c - w * t_c - p_c * tau_z * z_c
    profit_d = p_d * y_d - w * t_d - p_d * tau_z * z_d

    print("\n--- Firms ---")
    print(f"y_c: {y_c:.8f}")
    print(f"y_d: {y_d:.8f}")
    print(f"Profit_c: {profit_c:.8f}")
    print(f"Profit_d: {profit_d:.8f}")

    hh1_res   = hh1_focs(c1, d1, ell1, p_c, w, params['alpha'], params['beta'], params['gamma'], params['d0'], mult1, params['psi1'])
    hh2_res   = hh2_focs(c2, d2, ell2, p_c, w, params['alpha'], params['beta'], params['gamma'], params['d0'], mult2, params['psi2'])
    firm_c_res = firm_c_focs(t_c, z_c, p_c, w, tau_z, params['epsilon_c'], params['r'])
    firm_d_res = firm_d_focs(t_d, z_d, p_d, w, tau_z, params['epsilon_d'], params['r'])
    mkt_res  = market_clearing(c1, d1, ell1, c2, d2, ell2, t_c, z_c, t_d, z_d, p_c, p_d, w, params['epsilon_c'], params['epsilon_d'], params['r'])

    print("\n--- Residuals ---")
    print(f"Household 1 FOC c: {hh1_res[0]:.3e}")
    print(f"Household 1 FOC d: {hh1_res[1]:.3e}")
    print(f"Household 1 FOC ell: {hh1_res[2]:.3e}")
    print(f"Household 2 FOC c: {hh2_res[0]:.3e}")
    print(f"Household 2 FOC d: {hh2_res[1]:.3e}")
    print(f"Household 2 FOC ell: {hh2_res[2]:.3e}")
    print(f"Firm C FOC t: {firm_c_res[0]:.3e}")
    print(f"Firm C FOC z: {firm_c_res[1]:.3e}")
    print(f"Firm D FOC t: {firm_d_res[0]:.3e}")
    print(f"Firm D FOC z: {firm_d_res[1]:.3e}")
    print(f"Market Clearing Goods: {mkt_res[0]:.3e}")
    print(f"Market Clearing Labor: {mkt_res[1]:.3e}")

if __name__=="__main__":
    main()