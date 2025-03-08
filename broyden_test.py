import numpy as np
import numba as nb
from scipy.optimize import root
np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

#------------------------------------------------------------------------------#
# Model Functions (unchanged except for removal of feasibility penalties)
#------------------------------------------------------------------------------#
@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult):
    # Household first-order conditions (for consumption, dirty good consumption, and leisure)
    foc_c   = alpha * (c**(alpha-1)) - mult * p
    foc_d   = beta * ((d-d0)**(beta-1)) - mult * p
    foc_ell = gamma * (ell**(gamma-1)) - mult * w
    return np.array([foc_c, foc_d, foc_ell])

@nb.njit
def firm_production(t, z, epsilon, r):
    inside = epsilon*(t**r) + (1.0 - epsilon)*(z**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_focs(t, z, p, w, tau_z, epsilon, r):
    y = firm_production(t, z, epsilon, r)
    dy_dt = epsilon*(t**(r-1))*(y**(1.0-r))
    dy_dz = (1.0 - epsilon)*(z**(r-1))*(y**(1.0-r))
    foc_t = p * dy_dt - w
    foc_z = p * dy_dz - p * tau_z
    return np.array([foc_t, foc_z])

@nb.njit
def market_clearing(c, d, ell, t, z, p, epsilon, r):
    # Only the goods market clearing condition remains,
    # since labor clearing is automatic (t = T - ell).
    y = firm_production(t, z, epsilon, r)
    f_goods = (c + d) - y
    return np.array([f_goods])

#------------------------------------------------------------------------------#
# Transformation Functions
#------------------------------------------------------------------------------#
def transform(u, d0, T):
    """
    Transforms the unconstrained variables u into the original variables.
    u[0]: ln(c), so c = exp(u[0])
    u[1]: ln(d-d0), so d = d0 + exp(u[1])
    u[2]: used for leisure/labor split; sigma = logistic(u[2])
          then ell = T*sigma and t = T*(1-sigma)
    u[3]: used to parameterize z in (0,100): z = 100/(1+exp(-u[3]))
    u[4]: w (wage rate)
    u[5]: mult (Lagrange multiplier)
    """
    c = np.exp(u[0])
    d = d0 + np.exp(u[1])
    sigma = 1.0 / (1.0 + np.exp(-u[2]))
    ell = T * sigma
    t = T * (1.0 - sigma)
    z = 100.0 / (1.0 + np.exp(-u[3]))
    w = u[4]
    mult = u[5]
    return c, d, ell, t, z, w, mult

def full_system_transformed(u, params):
    """
    Computes the 6-dimensional system of equations in the transformed variables.
    """
    c, d, ell, t, z, w, mult = transform(u, params['d0'], params['T'])
    p      = params['p']
    alpha  = params['alpha']
    beta   = params['beta']
    gamma  = params['gamma']
    tau_z  = params['tau_z']
    epsilon= params['epsilon']
    r      = params['r']
    
    hh_res   = hh_focs(c, d, ell, p, w, alpha, beta, gamma, params['d0'], mult)
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)
    mkt_res  = market_clearing(c, d, ell, t, z, p, epsilon, r)
    # Total system: 3 (household) + 2 (firm) + 1 (goods market) = 6 equations.
    return np.concatenate((hh_res, firm_res, mkt_res))

#------------------------------------------------------------------------------#
# Main Solver Function
#------------------------------------------------------------------------------#
def main_broyden_transformed():
    # Set parameters, including a changeable total time endowment T.
    numeriare_value = 1  # p as numeraire
    params = {
        'alpha':   0.7,
        'beta':    0.2,
        'gamma':   0.2,
        'd0':      0.5,
        'tau_z':   1.25, # 1.25- 1.31  yield feasible solutions
        'epsilon': 0.92,
        'r':       0.5,
        'p':       numeriare_value,
        'T':       1.0   # Total time endowment; change this value as needed.
    }
    
    # Provide an initial guess for u (6 variables):
    # u[0] = ln(c), u[1] = ln(d-d0), u[2] = for leisure split, u[3] = for z, u[4] = w, u[5] = mult.
    u0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    
    sol = root(lambda u: full_system_transformed(u, params), u0, method='broyden1', tol=1e-10)
    
    print("\n=== Broyden Solver (Transformed) Result ===")
    print("Success:", sol.success)
    print("Message: ", sol.message)
    residual_norm = np.linalg.norm(full_system_transformed(sol.x, params))
    print("Norm of residuals: {:.3e}".format(residual_norm))
    
    # Transform back to the original variables.
    c, d, ell, t, z, w, mult = transform(sol.x, params['d0'], params['T'])
    print("\n--- Endogenous Variables ---")
    print(f"c (Clean Goods Consumption): {c:.8f}")
    print(f"d (Dirty Goods Consumption): {d:.8f}")
    print(f"ell (Leisure): {ell:.8f}")
    print(f"t (Labor Input): {t:.8f}")
    print(f"z (Pollution Input): {z:.8f}")
    print(f"w (Wage Rate): {w:.8f}")
    print(f"mult (Lagrange Multiplier): {mult:.8f}")
    
    # Compute firm output and profit.
    p      = params['p']
    epsilon= params['epsilon']
    r      = params['r']
    tau_z  = params['tau_z']
    y = firm_production(t, z, epsilon, r)
    profit = p * y - w * t - p * tau_z * z
    print("\n--- Firm ---")
    print(f"y (Output): {y:.8f}")
    print(f"Profit: {profit:.8f}")
    
    # Diagnostics: compute residuals for each block.
    hh_res   = hh_focs(c, d, ell, p, w, params['alpha'], params['beta'], params['gamma'], params['d0'], mult)
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)
    mkt_res  = market_clearing(c, d, ell, t, z, p, epsilon, r)
    budget_res = p * c + p * d - w * t - p * tau_z * z  # Note: labor market clearing holds automatically.
    
    print("\n--- Residuals ---")
    print(f"Household FOC c: {hh_res[0]:.3f}")
    print(f"Household FOC d: {hh_res[1]:.3f}")
    print(f"Household FOC ell: {hh_res[2]:.3f}")
    print(f"Firm FOC t: {firm_res[0]:.3f}")
    print(f"Firm FOC z: {firm_res[1]:.3f}")
    print(f"Market Clearing Goods: {mkt_res[0]:.3f}")
    print(f"Budget Constraint: {budget_res:.3f}")

if __name__=="__main__":
    main_broyden_transformed()