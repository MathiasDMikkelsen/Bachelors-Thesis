import numpy as np
import numba as nb
from scipy.optimize import root
import clean.blocks as blocks

np.set_printoptions(suppress=True, precision=8)

# 1. transform 
def transform(u, params, n=5):

    d0 = params['d0']
    
    d_raw   = u[0:5]
    ell_raw = u[5:10]
    lam     = u[10:15]
    d       = d0 + np.exp(d_raw)
    ell     = 1.0/(1.0 + np.exp(-ell_raw))
    
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
    x = params['x']
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
    G     = params['G']
    
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
    
    fc = blocks.firm_c_focs(t_c, z_c, p_c, w, tau_z, eps_c, r, x)
    fd = blocks.firm_d_focs(t_d, z_d, p_d, w, tau_z, eps_d, r, x)
    
    y_d = blocks.firm_d_production(t_d, z_d, eps_d, r, x)
    eq_d_mkt = np.sum(d) + 0.5 - y_d + 0.5 * G/p_d
    eq_l_mkt = np.sum(ell) + t_c + t_d - n*t_total
    
    return np.concatenate((hh_eq, fc, fd, [eq_d_mkt, eq_l_mkt]))
# end constructing full system

# 3. solve system
def solve(tau_w, tau_z, l_vec, G, n=5):
    
    params = {
        'alpha':     0.7,
        'beta':      0.2,
        'gamma':     0.2,
        'd0':        0.02, # klenert calibrates to 0.5
        'x':         100.0,
        'p_c':       1.0, 
        'epsilon_c': 0.995,
        'epsilon_d': 0.92,
        'r':         -1.0, # wrong before
        'tau_z':     tau_z,
        'tau_w':     tau_w, 
        'phi':       np.array([0.03, 0.0825, 0.141, 0.229, 0.511]),
        'l':         l_vec, 
        't_total':   1.0, 
        'G':         G
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
    
    sol = root(lambda x: full_system(x, params, n), u0, method='lm', tol=1e-12)
    final_res = full_system(sol.x, params, n)
    resid_norm = np.linalg.norm(final_res)
    
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(sol.x, params, n)
    
    pollution_tax_rev = tau_z*(z_c + z_d)
    
    # Back out c_i for each household
    c = np.empty(n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i]*pollution_tax_rev
        c[i] = (inc_i - p_d*d[i]) / params['p_c']
    
    # Compute budget errors for each household
    budget_errors = np.empty(n)
    for i in range(n):
        budget_errors[i] = params['p_c']*c[i] + p_d*d[i] - ((1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i]*pollution_tax_rev)
    
    # Compute household utilities: u_i = alpha*ln(c_i) + beta*ln(d_i-d0) + gamma*ln(ell_i)
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = params['alpha']*np.log(c[i]) + params['beta']*np.log(d[i]-params['d0']) + params['gamma']*np.log(ell[i])
    
    # Aggregate pollution:
    aggregate_polluting = z_d + z_c
    
    # Compute firm outputs and profits
    y_c = blocks.firm_c_production(t_c, z_c, params['epsilon_c'], params['r'], params['x'])
    y_d = blocks.firm_d_production(t_d, z_d, params['epsilon_d'], params['r'], params['x'])
    profit_c = params['p_c']*y_c - (w*t_c + tau_z*z_c)
    profit_d = p_d*y_d - (w*t_d + tau_z*z_d)
    
    print("equilibirum:")
    print("convergence ", sol.success)
    print("solution message ", sol.message)
    print("residual norm", resid_norm)
    print("")
    
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
    print("")
    
    print("firm d:")
    print(f"t_d: {t_d:.4f}")
    print(f"z_d: {z_d:.4f}")
    print(f"y_d: {y_d:.4f}")
    print(f"pi: {profit_d:.4f}")
    print("")
    
    print("prices:")
    print(f"p_d: {p_d:.4f}")
    print(f"w:   {w:.4f}")
    print("")
    
    print("final residuals:")
    print(final_res)
    
    return utilities, aggregate_polluting, sol.success, c, d, ell, w, p_d
# end solve system

# 4. test
n = 5
tau_w_arr = np.array([0.10, 0.12, 0.15, 0.30, 0.60])
l_arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
G = 0.0
tau_z = 2.
utilities, aggregate_polluting, first_converged, c, d, ell, w, p_d = solve(tau_w_arr, tau_z, l_arr, G, n=n)
    
print("returned Values:")
print("utilities:", utilities)
print("aggregate pollution:", aggregate_polluting)
print("convergence:", first_converged)
print("consumption:", c)
print("dirty good consumption:", d)
print("labor supply:", ell)
# end test