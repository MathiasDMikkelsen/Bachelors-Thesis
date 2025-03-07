import numpy as np
import numba as nb
from scipy.optimize import root
np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

#------------------------------------------------------------------------------#
# Model Functions (unchanged)
#------------------------------------------------------------------------------#
@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult):
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
    y = firm_production(t, z, epsilon, r)
    f_goods = (c + d) - y
    return np.array([f_goods])

#------------------------------------------------------------------------------#
# Transformation Functions
#------------------------------------------------------------------------------#
def transform(u, d0, T):
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
    return np.concatenate((hh_res, firm_res, mkt_res))

#------------------------------------------------------------------------------#
# Solver function trying multiple tau_z values
#------------------------------------------------------------------------------#
def find_feasible_tau_z():
    numeriare_value = 1
    base_params = {
        'alpha':   0.7,
        'beta':    0.2,
        'gamma':   0.2,
        'd0':      0.5,
        'epsilon': 0.92,
        'r':       0.5,
        'p':       numeriare_value,
        'T':       1.0
    }
    
    u0 = np.array([np.log(1.0), np.log(0.6), 0.0, 1.0, 1.0, 1.0])
    tau_z_values = np.linspace(0.1, 10.0, 100)

    feasible_solutions = []

    for tau_z in tau_z_values:
        params = base_params.copy()
        params['tau_z'] = tau_z

        sol = root(lambda u: full_system_transformed(u, params), u0, method='lm', tol=1e-8)

        residual_norm = np.linalg.norm(full_system_transformed(sol.x, params))
        if sol.success and residual_norm < 1e-6:
            c, d, ell, t, z, w, mult = transform(sol.x, params['d0'], params['T'])
            feasible_solutions.append({
                'tau_z': tau_z,
                'c': c,
                'd': d,
                'ell': ell,
                't': t,
                'z': z,
                'w': w,
                'mult': mult,
                'residual_norm': residual_norm
            })
            print(f"Converged at tau_z = {tau_z:.4f}, residual norm = {residual_norm:.3e}")

    if feasible_solutions:
        print("\n=== Feasible solutions found: ===")
        for sol in feasible_solutions:
            print(f"tau_z = {sol['tau_z']:.4f}, c = {sol['c']:.4f}, d = {sol['d']:.4f}, ell = {sol['ell']:.4f}, t = {sol['t']:.4f}, z = {sol['z']:.4f}, w = {sol['w']:.4f}")
    else:
        print("No feasible tau_z found.")

if __name__=="__main__":
    numeriare_value = 1
    find_feasible_tau_z()
