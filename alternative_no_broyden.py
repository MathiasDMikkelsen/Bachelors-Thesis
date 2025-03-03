import numpy as np
import numba as nb
from scipy.optimize import minimize

###############################################################################
# 1) Household, Firm, Market, Budget with no taxes
###############################################################################

@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult):
    """
    U = c^alpha + (d-d0)^beta + ell^gamma
    FOCs:
      alpha*c^(alpha-1) = mult*p
      beta*(d-d0)^(beta-1) = mult*p
      gamma*ell^(gamma-1) = mult*w
    c>0, d>d0, 0<ell<1 => if invalid => large residuals
    """
    if c<=0.0 or d<=d0 or ell<=0.0 or ell>=1.0:
        return np.array([1e6,1e6,1e6])
    foc_c   = alpha*(c**(alpha-1)) - mult*p
    foc_d   = beta*((d - d0)**(beta - 1)) - mult*p
    foc_ell = gamma*(ell**(gamma-1)) - mult*w
    return np.array([foc_c, foc_d, foc_ell])

@nb.njit
def firm_production(t,z,epsilon,r):
    inside = epsilon*(t**r) + (1.0 - epsilon)*(z**r)
    if inside<=0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_focs(t,z,p,w,epsilon,r):
    """
    2 eqs:
      p*dy/dt - w=0
      p*dy/dz - 0=0 (since tau_z=0)
    """
    y = firm_production(t,z,epsilon,r)
    if y<=0.0:
        return np.array([1e6,1e6])
    dy_dt = epsilon*(t**(r-1))*(y**(1-r))
    dy_dz = (1.0 - epsilon)*(z**(r-1))*(y**(1-r))
    return np.array([p*dy_dt - w, p*dy_dz - 0.0])

@nb.njit
def market_clearing(c,d,ell,t,z,p,w,epsilon,r):
    """
    (c + d)-y(t,z)=0,
    (1-ell)-t=0 => t=1-ell
    """
    y = firm_production(t,z,epsilon,r)
    return np.array([(c + d)-y, (1.0 - ell)-t])

@nb.njit
def budget_constraint(c,d,ell,w,z,p):
    """
    (c + d)*p - w*(1-ell) =0
    c+d>0 => else big penalty
    """
    if (c+d)<=0.0:
        return 1e6
    lhs = (c + d)*p
    rhs = w*(1.0 - ell)
    return lhs - rhs

def full_system(x, params):
    """
    x= [c,d,ell,t,z,p,w,mult], => 8 eq:
      - 3 from HH FOCs
      - 2 from firm FOCs
      - 2 from market eq
      - 1 from budget
    """
    c,d,ell,t,z,p,w,mult= x
    alpha= params['alpha']
    beta=  params['beta']
    gamma= params['gamma']
    d0=    params['d0']
    epsilon= params['epsilon']
    r= params['r']

    hh_res= hh_focs(c,d,ell,p,w,alpha,beta,gamma,d0,mult)  # length 3
    firm_res= firm_focs(t,z,p,w,epsilon,r)                 # length 2
    mkt_res= market_clearing(c,d,ell,t,z,p,w,epsilon,r)    # length 2
    budg_val= budget_constraint(c,d,ell,w,z,p)             # scalar => array
    return np.concatenate((hh_res, firm_res, mkt_res, [budg_val]))

def objective(x, params):
    """ sum of squares of the 8 residuals """
    res = full_system(x, params)
    return np.sum(res**2)

###############################################################################
# 2) Hybrid approach: random search + local refinement
###############################################################################

def random_search_plus_local_minimize():
    """
    1) random search over a bounding region to find a best candidate
    2) local minimize (L-BFGS-B) from that candidate
    """
    # set parameter dictionary
    params = {
        'alpha': 0.7,
        'beta':  0.2,
        'gamma': 0.1,
        'd0':    0.5,
        'epsilon': 0.5,
        'r':     0.8
    }

    # define bounding region for x= [c,d,ell,t,z,p,w,mult]
    # c>0, d> d0=0.5, ell in(0,1), t in(0,1), z>0, p>0, w>0, mult>0
    # we pick some upper bounds to keep the region finite
    bounds = [
        (1e-6, 10.0),   # c
        (0.51, 10.0),   # d
        (1e-6, 0.9999), # ell
        (1e-6, 0.9999), # t
        (1e-6, 10.0),   # z
        (1e-6, 10.0),   # p
        (1e-6, 10.0),   # w
        (1e-6, 10.0)    # mult
    ]

    # 1) random search
    np.random.seed(123)  # for reproducibility
    def random_feasible(bounds):
        return np.array([
            np.random.uniform(low=b[0], high=b[1]) for b in bounds
        ])

    best_val = 1e15
    best_x = None
    n_random = 2000
    for i in range(n_random):
        x_trial = random_feasible(bounds)
        val = objective(x_trial, params)
        if val< best_val:
            best_val= val
            best_x= x_trial

    print(f"Random search best so far: sum of squares= {best_val:.6e}, x= {best_x}")

    # 2) local refine from best_x using L-BFGS-B
    from scipy.optimize import minimize
    def f_obj(x):
        return objective(x, params)

    # convert 'bounds' to 'minimize' style
    lbfgs_bounds= bounds
    res = minimize(
        f_obj,
        best_x,
        method='L-BFGS-B',
        bounds=lbfgs_bounds,
        options={'ftol':1e-12, 'maxiter':2000}
    )
    if not res.success:
        print("Local minimization did not converge:", res.message)
    else:
        print("Local minimization successful!")
    x_sol = res.x
    val_sol= res.fun
    print("Final sum of squares:", val_sol)
    print("Solution x= [c,d,ell,t,z,p,w,mult]:\n", x_sol)
    final_res = full_system(x_sol, params)
    print("Final residuals:\n", final_res)

###############################################################################
# 3) Main
###############################################################################

if __name__=="__main__":
    random_search_plus_local_minimize()