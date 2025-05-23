import numpy as np
import numba as nb
from scipy.optimize import root

# 1. blocks
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):
    inside = epsilon_c * (t_c**r) + (1.0 - epsilon_c) * (z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):
    inside = epsilon_d * (t_d**r) + (1.0 - epsilon_d) * (z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c * (t_c**(r-1.0)) * (y_c**(1.0-r))
    dy_dz_c = (1.0 - epsilon_c) * (z_c**(r-1.0)) * (y_c**(1.0-r))
    eq_t_c  = p_c * dy_dt_c - w
    eq_z_c  = p_c * dy_dz_c - tau_z
    return np.array([eq_t_c, eq_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d * (t_d**(r-1.0)) * (y_d**(1.0-r))
    dy_dz_d = (1.0 - epsilon_d) * (z_d**(r-1.0)) * (y_d**(1.0-r))
    eq_t_d  = p_d * dy_dt_d - w
    eq_z_d  = p_d * dy_dz_d - tau_z
    return np.array([eq_t_d, eq_z_d])
# end blocks

# 2. transform func
def transform(u, params, n=5):

    d0 = params['d0']
    
    # Households:
    d_raw   = u[0:5]
    ell_raw = u[5:10]
    lam     = u[10:15]
    d       = d0 + np.exp(d_raw)
    ell     = 1.0/(1.0 + np.exp(-ell_raw))
    
    # Market:
    t_c  = n/(1.0 + np.exp(-u[15]))
    t_d  = n/(1.0 + np.exp(-u[16]))
    z_c  = np.exp(u[17])
    z_d  = np.exp(u[18])
    p_d  = np.exp(u[19])
    w    = np.exp(u[20])
    
    return d, ell, lam, t_c, t_d, z_c, z_d, p_d, w
# end transform

# 3. construct full system
def full_system(u, params, n=5):
    
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(u, params, n)
    
    alpha = params['alpha']
    beta  = params['beta']
    gamma = params['gamma']
    d0    = params['d0']
    p_c   = params['p_c']
    tau_z = params['tau_z']
    tau_w = params['tau_w']
    phi   = params['phi']
    l_vec = params['l']
    t_total = params['t_total']
    eps_c = params['epsilon_c']
    eps_d = params['epsilon_d']
    r     = params['r']
    
    pollution_tax_rev = (z_c + z_d)*tau_z
    
    hh_eq = np.empty(3*n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*phi[i]*w*(t_total - ell[i]) + l_vec[i]*pollution_tax_rev
        c_i = (inc_i - p_d*d[i]) / p_c  
        
        eq_c   = alpha/c_i - lam[i]*p_c
        eq_d   = beta/(d[i]-d0) - lam[i]*p_d
        eq_ell = gamma/ell[i] - lam[i]*(1.0-tau_w[i])*phi[i]*w
        
        hh_eq[3*i]   = eq_c
        hh_eq[3*i+1] = eq_d
        hh_eq[3*i+2] = eq_ell
    
    # Firm FOCs:
    fc = firm_c_focs(t_c, z_c, p_c, w, tau_z, eps_c, r)
    fd = firm_d_focs(t_d, z_d, p_d, w, tau_z, eps_d, r)
    
    # Market clearing:
    y_d = firm_d_production(t_d, z_d, eps_d, r)
    eq_d_mkt = np.sum(d) + 0.5 - y_d
    eq_l_mkt = np.sum(ell) + t_c + t_d - n*t_total
    
    return np.concatenate((hh_eq, fc, fd, [eq_d_mkt, eq_l_mkt]))
# end full system

# solve system
def solve(tau_w, tau_z, l_vec, n=5):
    
    params = {
        'alpha':     0.7,
        'beta':      0.2,
        'gamma':     0.2,
        'd0':        0.2,
        'p_c':       1.0, # normalized price of clean good
        'epsilon_c': 0.995,
        'epsilon_d': 0.92,
        'r':         0.5,
        'tau_z':     tau_z,
        'tau_w':     tau_w, 
        'phi':       np.array([0.1*5, 0.1*5, 0.2*5, 0.3*5, 0.511*5]),
        'l':         l_vec, 
        't_total':   1.0, 
    }
    
    u0 = np.zeros(21)
    u0[0:5]   = np.log(0.99)       # ln(d_i-d0)
    u0[5:10]  = 0.0                # ell_i = 0.5
    u0[10:15] = 1.0                # lambda_i = 1
    u0[15]    = 0.0                # t_c => ~ n/2
    u0[16]    = 0.0                # t_d => ~ n/2
    u0[17]    = 0.0                # ln(z_c)=0 => z_c=1
    u0[18]    = 0.0                # ln(z_d)=0 => z_d=1
    u0[19]    = 0.0                # ln(p_d)=0 => p_d=1
    u0[20]    = np.log(5.0)        # ln(w)=ln(5) => w=5
    
    # Solve using LM
    sol = root(lambda x: full_system(x, params, n), u0, method='lm', tol=1e-12)
    final_res = full_system(sol.x, params, n)
    resid_norm = np.linalg.norm(final_res)
    
    # Transform solution to get variables
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(sol.x, params, n)
    
    tax_rev = 0.0
    for i in range(n):
        tax_rev += tau_w[i]*w*(1.0 - ell[i])*params['phi'][i]
    tax_rev += tau_z*(z_c + z_d)
    
    # Back out c_i for each household
    c = np.empty(n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i]*tax_rev
        c[i] = (inc_i - p_d*d[i]) / params['p_c']
    
    # Compute budget errors for each household
    budget_errors = np.empty(n)
    for i in range(n):
        budget_errors[i] = params['p_c']*c[i] + p_d*d[i] - ((1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i]*tax_rev)
    
    # Compute household utilities: u_i = alpha*ln(c_i) + beta*ln(d_i-d0) + gamma*ln(ell_i)
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = params['alpha']*np.log(c[i]) + params['beta']*np.log(d[i]-params['d0']) + params['gamma']*np.log(ell[i])
    
    # Aggregate pollution:
    aggregate_polluting = z_d + z_c
    
    # Compute firm outputs and profits
    y_c = firm_c_production(t_c, z_c, params['epsilon_c'], params['r'])
    y_d = firm_d_production(t_d, z_d, params['epsilon_d'], params['r'])
    profit_c = params['p_c']*y_c - (w*t_c + tau_z*z_c)
    profit_d = p_d*y_d - (w*t_d + tau_z*z_d)
    
    # Print equilibrium results
    print("equilibirum:")
    print("convergence ", sol.success)
    print("solution message ", sol.message)
    print("residual norm", resid_norm)
    
    print("hh solution:")
    for i in range(n):
        print(f"hh {i+1}:")
        print(f"d: {d[i]:.4f}")
        print(f"ell: {ell[i]:.4f}")
        print(f"lambda: {lam[i]:.4f}")
        print(f"c (backed out): {c[i]:.4f}")
        print(f"budget error: {budget_errors[i]:.2e}")
        print("")
    
    print("firm c:")
    print(f"t: {t_c:.4f}")
    print(f"z: {z_c:.4f}")
    print(f"y: {y_c:.4f}")
    print(f"pi: {profit_c:.4f}")
    
    print("firm d:")
    print(f"t_d: {t_d:.4f}")
    print(f"z_d: {z_d:.4f}")
    print(f"y_d: {y_d:.4f}")
    print(f"Profit_D: {profit_d:.4f}")
    
    print("prices:")
    print(f"p_d: {p_d:.4f}")
    print(f"w:   {w:.4f}")
    
    print("final residuals:")
    print(final_res)
    
    return utilities, aggregate_polluting, sol.success, c, d, ell

# test
n = 5
tau_w_arr = np.array([0.10, 0.12, 0.15, 0.18, 0.20])
l_arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
tau_z = 1.0
utilities, aggregate_polluting, first_converged, c, d, ell = solve(tau_w_arr, tau_z, l_arr, n=n)
    
print("returned Values:")
print("utilities:", utilities)
print("aggregate pollution:", aggregate_polluting)
print("convergence:", first_converged)
