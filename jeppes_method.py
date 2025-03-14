import numpy as np
import numba as nb
from scipy.optimize import root

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})

###############################################################################
# 1) Firm Production and FOCs (unchanged)
###############################################################################
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):
    inside = epsilon_c*(t_c**r) + (1.0 - epsilon_c)*(z_c**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):
    inside = epsilon_d*(t_d**r) + (1.0 - epsilon_d)*(z_d**r)
    if inside <= 0.0:
        return 0.0
    return inside**(1.0/r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c*(t_c**(r-1.0))*(y_c**(1.0-r))
    dy_dz_c = (1.0 - epsilon_c)*(z_c**(r-1.0))*(y_c**(1.0-r))
    eq_t_c  = p_c*dy_dt_c - w
    eq_z_c  = p_c*dy_dz_c - tau_z
    return np.array([eq_t_c, eq_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d*(t_d**(r-1.0))*(y_d**(1.0-r))
    dy_dz_d = (1.0 - epsilon_d)*(z_d**(r-1.0))*(y_d**(1.0-r))
    eq_t_d  = p_d*dy_dt_d - w
    eq_z_d  = p_d*dy_dz_d - tau_z
    return np.array([eq_t_d, eq_z_d])

###############################################################################
# 2) Transformation Function (21 unknowns for n=5)
###############################################################################
def transform(u, params, n=5):
    """
    The unknown vector u has 21 entries:
    
      For each household i (i=0,...,4):
        u[i]       = ln(d_i - d0), so that d_i = d0 + exp(u[i])
        u[5+i]     = logistic transform for leisure: ell_i = 1/(1+exp(-u[5+i]))
        u[10+i]    = lambda_i (direct, no transform)
      
      Market variables:
        u[15]     = logistic transform for t_c: t_c = n/(1+exp(-u[15]))
        u[16]     = logistic transform for t_d: t_d = n/(1+exp(-u[16]))
        u[17]     = ln(z_c), so that z_c = exp(u[17])
        u[18]     = ln(z_d), so that z_d = exp(u[18])
        u[19]     = ln(p_d), so that p_d = exp(u[19])
        u[20]     = ln(w),   so that w   = exp(u[20])
    """
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

###############################################################################
# 3) Full System Function (21 equations)
###############################################################################
def full_system(u, params, n=5):
    """
    Constructs a system of 21 equations:
    
      Households: 3 FOCs per household with c substituted from budget:
        Budget: p_c*c_i + p_d*d_i = (1-tau_w_i)*phi_i*w*(t_total - ell_i) + l_i.
        => c_i = [ (1-tau_w_i)*phi_i*w*(t_total - ell_i) + l_i - p_d*d_i ] / p_c.
      
        FOC for c:   alpha/c_i - lambda_i*p_c = 0.
        FOC for d:   beta/(d_i-d0) - lambda_i*p_d = 0.
        FOC for ell: gamma/ell_i - lambda_i*(1-tau_w_i)*phi_i*w = 0.
      (Total: 15 eqs for 5 households.)
      
      Firms: 2 eq for firm C and 2 eq for firm D (4 eqs).
      
      Markets: 1 eq for polluting good market (sum d_i - y_d = 0)
               and 1 eq for labor market (sum ell_i + t_c + t_d - n*t_total = 0).
      (Total: 2 eqs)
      
      Grand total: 15 + 4 + 2 = 21 equations.
    """
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
    
    # Compute tax revenue:
    tax_rev = 0.0
    for i in range(n):
        tax_rev += tau_w[i]*w*(t_total - ell[i])*phi[i]
    tax_rev += tau_z*(z_c + z_d)
    
    # Household FOCs:
    hh_eq = np.empty(3*n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*phi[i]*w*(t_total - ell[i]) + l_vec[i]
        c_i = (inc_i - p_d*d[i]) / p_c  # c substituted from budget
        
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
    eq_d_mkt = np.sum(d) - y_d
    eq_l_mkt = np.sum(ell) + t_c + t_d - n*t_total
    
    return np.concatenate((hh_eq, fc, fd, [eq_d_mkt, eq_l_mkt]))

###############################################################################
# 4) Main Solve and Print Equilibrium with Budget Errors
###############################################################################
def main_solve_print(tau_w, tau_z, l_vec, n=5):
    """
    Solves the 21-equation system using LM, prints the equilibrium solution along with
    the budget errors for each household, and returns:
      - utilities: array of household utilities,
      - aggregate_polluting: sum of d_i across households,
      - first_converged: Boolean flag indicating if the solver converged.
    """
    # Set parameters
    params = {
        'alpha':     0.7,
        'beta':      0.1,
        'gamma':     0.1,
        'd0':        0.01,
        'p_c':       1.0,  # normalized price of clean good
        'epsilon_c': 0.8,
        'epsilon_d': 0.6,
        'r':         0.5,
        'tau_z':     tau_z,
        'tau_w':     tau_w,    # array of length n
        'phi':       np.array([0.7, 0.8, 0.9, 1.2, 1.4]),
        'l':         l_vec,    # lumpsum offsets per household
        't_total':   1.0,      # total time endowment
    }
    
    # Initial guess: 21 variables
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
    
    # Back out c_i for each household
    c = np.empty(n)
    for i in range(n):
        inc_i = (1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i]
        c[i] = (inc_i - p_d*d[i]) / params['p_c']
    
    # Compute budget errors for each household
    budget_errors = np.empty(n)
    for i in range(n):
        budget_errors[i] = params['p_c']*c[i] + p_d*d[i] - ((1.0 - tau_w[i])*params['phi'][i]*w*(params['t_total'] - ell[i]) + l_vec[i])
    
    # Compute household utilities: u_i = alpha*ln(c_i) + beta*ln(d_i-d0) + gamma*ln(ell_i)
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = params['alpha']*np.log(c[i]) + params['beta']*np.log(d[i]-params['d0']) + params['gamma']*np.log(ell[i])
    
    # Aggregate pollution:
    aggregate_polluting = np.sum(d)
    
    # Compute firm outputs and profits
    y_c = firm_c_production(t_c, z_c, params['epsilon_c'], params['r'])
    y_d = firm_d_production(t_d, z_d, params['epsilon_d'], params['r'])
    profit_c = params['p_c']*y_c - (w*t_c + tau_z*z_c)
    profit_d = p_d*y_d - (w*t_d + tau_z*z_d)
    
    # Print equilibrium results
    print("\n================== EQUILIBRIUM RESULTS ==================")
    print("Converged?    ", sol.success)
    print("Message:      ", sol.message)
    print("Residual Norm:", resid_norm)
    
    print("\n--- Household Results ---")
    for i in range(n):
        print(f"Household {i+1}:")
        print(f"  d_i: {d[i]:.4f}")
        print(f"  ell_i: {ell[i]:.4f}")
        print(f"  lambda_i: {lam[i]:.4f}")
        print(f"  c_i (backed out): {c[i]:.4f}")
        print(f"  Budget Error: {budget_errors[i]:.2e}")
        print("")
    
    print("=== FIRM C (clean good) ===")
    print(f"t_c: {t_c:.4f}")
    print(f"z_c: {z_c:.4f}")
    print(f"y_c: {y_c:.4f}")
    print(f"Profit_C: {profit_c:.4f}")
    
    print("\n=== FIRM D (polluting good) ===")
    print(f"t_d: {t_d:.4f}")
    print(f"z_d: {z_d:.4f}")
    print(f"y_d: {y_d:.4f}")
    print(f"Profit_D: {profit_d:.4f}")
    
    print("\n=== MARKET PRICES ===")
    print(f"p_d: {p_d:.4f}")
    print(f"w:   {w:.4f}")
    
    print("\n=== FINAL RESIDUALS (21 eq) ===")
    print(final_res)
    print("=======================================================")
    
    # Return utilities, aggregate_polluting, and convergence flag
    return utilities, aggregate_polluting, sol.success

###############################################################################
# Run the Equilibrium Solver and Print Results
###############################################################################
if __name__=="__main__":
    n = 5
    tau_w_arr = np.array([0.10, 0.12, 0.15, 0.18, 0.20])
    l_arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    tau_z = 4.0
    utilities, aggregate_polluting, first_converged = main_solve_print(tau_w_arr, tau_z, l_arr, n=n)
    
    print("\nReturned Values:")
    print("Utilities:", utilities)
    print("Aggregate Polluting:", aggregate_polluting)
    print("Convergence Flag (first_converged):", first_converged)