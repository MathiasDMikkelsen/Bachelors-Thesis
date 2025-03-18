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
    t_total = params['t_total']
    eps_c = params['epsilon_c']
    eps_d = params['epsilon_d']
    r     = params['r']
    G     = params['G']
    
    tax_rev = 0
    for i in range(n):
        tax_rev += tau_w[i]*phi[i]*w*(t_total-ell[i])
    tax_rev += (z_c + z_d)*tau_z
    
    l_val = (tax_rev - G) / n # Lumpsum value
    
    hh_eq = np.empty(3*n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*phi[i]*w*(t_total - ell[i]) + l_val
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
    eq_l_mkt = np.sum(phi*(t_total-ell)) - (t_d+t_c)
    
    return np.concatenate((hh_eq, fc, fd, [eq_d_mkt, eq_l_mkt]))
# end constructing full system

# 3. solve system
def solve(tau_w, tau_z, G, n=5):
    params = {
        'alpha':     0.7,
        'beta':      0.2,
        'gamma':     0.2,
        'd0':        0.1,
        'x':         100.0,
        'p_c':       1.0, 
        'epsilon_c': 0.995,
        'epsilon_d': 0.92,
        'r':         -1.0,
        'tau_z':     tau_z,
        'tau_w':     tau_w, 
        'phi':       np.array([0.03, 0.0825, 0.141, 0.229, 0.5175]),
        't_total':   1.0, 
        'G':         G
    }
    
    u0 = np.zeros(21)
    u0[0:5]   = np.log(0.99)       # ln(d_i-d0)
    u0[5:10]  = 0.0                # ell_i ~ 0.5
    u0[10:15] = 1.0                # lambda_i = 1
    u0[15]    = 0.0                # t_c
    u0[16]    = 0.0                # t_d
    u0[17]    = 0.0                # ln(z_c)=0 => z_c=1
    u0[18]    = 0.0                # ln(z_d)=0 => z_d=1
    u0[19]    = 0.0                # ln(p_d)=0 => p_d=1
    u0[20]    = np.log(5.0)        # ln(w)=ln(5) => w=5
    
    sol = root(lambda x: full_system(x, params, n), u0, method='lm', tol=1e-12)
    final_res = full_system(sol.x, params, n)
    resid_norm = np.linalg.norm(final_res)
    
    d, ell, lam, t_c, t_d, z_c, z_d, p_d, w = transform(sol.x, params, n)
    
    tax_rev = 0
    for i in range(n):
        tax_rev += tau_w[i]*params['phi'][i]*w*(params['t_total']-ell[i])
    tax_rev += (z_c + z_d)*tau_z
    
    l_val = (tax_rev - G) / n
    
    # Back out c_i for each household
    c = np.empty(n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_val
        c[i] = (inc_i - p_d*d[i]) / params['p_c']
    
    # Compute household utilities: u_i = alpha*ln(c_i) + beta*ln(d_i-d0) + gamma*ln(ell_i)
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = params['alpha']*np.log(c[i]) + params['beta']*np.log(d[i]-params['d0']) + params['gamma']*np.log(ell[i])
    
    # Aggregate pollution:
    aggregate_polluting = z_d + z_c
    
    # (Firms' outputs and profits are computed below for information.)
    y_c = blocks.firm_c_production(t_c, z_c, params['epsilon_c'], params['r'], params['x'])
    y_d = blocks.firm_d_production(t_d, z_d, params['epsilon_d'], params['r'], params['x'])
    profit_c = params['p_c']*y_c - (w*t_c + tau_z*z_c)
    profit_d = p_d*y_d - (w*t_d + tau_z*z_d)
    
    # Print equilibrium details (optional)
    print("--------------------------------------------------")
    print(f"Solving for tau_z = {tau_z:.4f}, G = {G:.4f}")
    print("Convergence:", sol.success)
    print("Residual norm:", resid_norm)
    print("")
    
    return utilities, aggregate_polluting, sol.success, c, d, ell, w, p_d, l_val, params, resid_norm
# end solve system

# 4. loop over tau_z and G combinations
if __name__ == "__main__":
    n = 5
    # Fixed tau_w for households (using your chosen values)
    tau_w_arr = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
    
    # Define grids for tau_z and G. Adjust these ranges and number of points as needed.
    tau_z_values = np.linspace(0.01, 5.0, 20)
    G_values = np.linspace(0.0, 5.0, 20)
    
    tol = 1e-6  # Tolerance for considering residuals as "zero"
    valid_combinations = []
    
    for tau_z in tau_z_values:
        for G in G_values:
            # Solve the model for this combination.
            utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, params, resid_norm = solve(tau_w_arr, tau_z, G, n=n)
            
            # Check if solution converged and the residual norm is below tolerance.
            if converged and resid_norm < tol:
                valid_combinations.append((tau_z, G, resid_norm))
    
    print("==================================================")
    print("Found valid (tau_z, G) combinations with near-zero residuals:")
    for combo in valid_combinations:
        print(f"tau_z = {combo[0]:.4f}, G = {combo[1]:.4f}, Residual norm = {combo[2]:.2e}")