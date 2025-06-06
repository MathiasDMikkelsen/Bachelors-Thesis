import numpy as np                                                            # Imports the NumPy library for numerical computing
import numba as nb                                                            # Imports Numba to speed up Python functions using JIT compilation
from scipy.optimize import root                                               # Imports the 'root' function from SciPy for solving systems of equations

np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})     # Sets NumPy print options for floating-point formatting

##############################################################################################################################################################
# 1) Household Functions (vectorized over n households)
##############################################################################################################################################################
@nb.njit
def households_focs_array(c, d, ell, p_c, p_d, w,
                           alpha, beta, gamma, d0, mult, phi):
    """
    For each household i (i=0,...,n-1):
      FOC for c:  alpha * c_i^(alpha-1)   - mult_i * p_c = 0
      FOC for d:  beta  * (d_i - d0)^(beta-1) - mult_i * p_d = 0
      FOC for ell: gamma * ell_i^(gamma-1)   - mult_i * w * phi_i = 0
    If any variable is out of bounds, we return a high penalty.
    Returns a 3*n vector.
    """
    n = c.shape[0]                                                            # Number of households
    res = np.empty(3 * n)                                                     # Pre-allocates array for residuals
    for i in range(n):                                                        # Loops over each household
        if (c[i] <= 0.0) or (d[i] <= d0) or (ell[i] <= 0.0) or (ell[i] >= 1.0):
            res[3*i : 3*i+3] = np.array([1e6, 1e6, 1e6])                      # Penalty if out of bounds
        else:
            foc_c   = alpha * 1/c[i] - mult[i] * p_c                          # FOC for clean good
            foc_d   = beta  * 1/(d[i] - d0) - mult[i] * p_d                   # FOC for dirty good
            foc_ell = gamma * 1/ell[i] - mult[i] * w * phi[i]                 # FOC for leisure
            res[3*i : 3*i+3] = np.array([foc_c, foc_d, foc_ell])
    return res

@nb.njit
def households_budget_array(c, d, ell, p_c, p_d, w, phi, tau_w, l, tax_revenue):
    """
    For each household, the budget constraint is:
      p_c * c_i + p_d * d_i = w * (1 - ell_i) * phi_i
    Returns a vector of length n.
    """
    n = c.shape[0]
    res = np.empty(n)
    for i in range(n):                                                        # Loops over each household
        res[i] = p_c * c[i] + p_d * d[i] - (1-tau_w[i]) * w * (1.0 - ell[i]) * phi[i] - l[i] * tax_revenue 
    return res                                                                # Returns the array of budget constraints

##############################################################################################################################################################
# 2) Firms: Production & FOCs
##############################################################################################################################################################
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r):                                # Numba-jitted function for firm C production
    inside = epsilon_c * (t_c**r) + (1.0 - epsilon_c) * (z_c**r)              # CES-style inside term
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)                                                  # Returns the production output of good c

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r):                                # Numba-jitted function for firm D production
    inside = epsilon_d * (t_d**r) + (1.0 - epsilon_d) * (z_d**r)              # CES-style inside term
    if inside <= 0.0:
        return 0.0
    return inside**(1.0 / r)                                                  # Returns the production output of good d

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):                       # Numba-jitted function for firm C's FOCs
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)                           # Computes output of firm C
    if y_c <= 0.0:
        return np.array([1e6, 1e6])                                           # Penalty if output is invalid
    dy_dt_c = epsilon_c * (t_c**(r - 1.0)) * (y_c**(1.0 - r))                 # Derivative of y_c w.r.t t_c
    dy_dz_c = (1.0 - epsilon_c) * (z_c**(r - 1.0)) * (y_c**(1.0 - r))         # Derivative of y_c w.r.t z_c
    foc_t_c = p_c * dy_dt_c - w                                               # FOC for labor
    foc_z_c = p_c * dy_dz_c - tau_z                                           # FOC for pollution
    return np.array([foc_t_c, foc_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):                       # Numba-jitted function for firm D's FOCs
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)                           # Computes output of firm D
    if y_d <= 0.0:
        return np.array([1e6, 1e6])                                           # Penalty if output is invalid
    dy_dt_d = epsilon_d * (t_d**(r - 1.0)) * (y_d**(1.0 - r))                 # Derivative of y_d w.r.t t_d
    dy_dz_d = (1.0 - epsilon_d) * (z_d**(r - 1.0)) * (y_d**(1.0 - r))         # Derivative of y_d w.r.t z_d
    foc_t_d = p_d * dy_dt_d - w                                               # FOC for labor
    foc_z_d = p_d * dy_dz_d - tau_z                                           # FOC for pollution
    return np.array([foc_t_d, foc_z_d])

##############################################################################################################################################################
# 3) Market Clearing (3 Equations)
##############################################################################################################################################################
@nb.njit
def market_clearing(c, d, ell, t_c, z_c, t_d, z_d, epsilon_c, epsilon_d, r):
    """
    Market clearing requires that:
      - The sum of household consumption of good c equals firm C's output.
      - The sum of household consumption of good d equals firm D's output.
      - The labor market clears: sum(ell) + t_c + t_d = number of households.
    Here, output is computed using the full production functions.
    """
    y_c = firm_c_production(t_c, z_c, epsilon_c, r)                           # Firm C output
    y_d = firm_d_production(t_d, z_d, epsilon_d, r)                           # Firm D output
    eq_c = np.sum(c) - y_c                                                    # Supply-demand gap for c
    eq_d = np.sum(d) - y_d                                                    # Supply-demand gap for d
    eq_l = np.sum(ell) + t_c + t_d - c.shape[0]                               # Labor market clearing
    return np.array([eq_c, eq_d, eq_l])

##############################################################################################################################################################
# 4) Transformation Function: 26 Unknowns for 5 Households
##############################################################################################################################################################
def transform(u, d0, n=5):                                                    # Transforms the 26-dimensional vector u
    """
    The unknown vector u is 26-dimensional (for n=5 households):
      u[0:5]   : ln(c_i)       for i=1,...,5.
      u[5:10]  : ln(d_i - d0)  for i=1,...,5.
      u[10:15] : real numbers, logistic transform => ell_i = 1/(1+exp(-x))
      u[15:20] : multipliers for each household.
      u[20]   : real number, logistic transform => t_c (scaled by n)
      u[21]   : real number, logistic transform => t_d (scaled by n)
      u[22]   : ln(z_c)  so that z_c = exp(u[22])
      u[23]   : ln(z_d)  so that z_d = exp(u[23])
      u[24]   : ln(p_d)  so that p_d = exp(u[24])
      u[25]   : ln(w)    so that w = exp(u[25])
    """
    # Households:
    c    = np.exp(u[0:5])                                                     # Clean good consumption c_i
    d    = d0 + np.exp(u[5:10])                                               # Polluting good consumption d_i
    ell  = 1.0 / (1.0 + np.exp(-u[10:15]))                                    # Leisure as logistic transform
    mult = u[15:20]                                                           # Multipliers for each household
    
    # Market/Firms variables:
    t_c  = n * (1.0 / (1.0 + np.exp(-u[20])))                                 # Labor for firm C
    t_d  = n * (1.0 / (1.0 + np.exp(-u[21])))                                 # Labor for firm D
    z_c  = np.exp(u[22])                                                      # Pollution for firm C
    z_d  = np.exp(u[23])                                                      # Pollution for firm D
    p_d  = np.exp(u[24])                                                      # Price of the polluting good
    w    = np.exp(u[25])                                                      # Wage
    return c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w

##############################################################################################################################################################
# 5) Full System Function: 26 Equations
##############################################################################################################################################################
def full_system(u, params, n=5):                                              # Defines the function computing the residuals
    """
    The system comprises:
      - 3n household FOCs
      - 2 firm FOCs for firm C
      - 2 firm FOCs for firm D
      - 3 market clearing equations
      - n household budgets, dropping one
    Total equations: 15 + 2 + 2 + 3 + (5-1) = 26.
    """
    d0 = params['d0']                                                         # Grabs d0 from params
    (c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w) = transform(u, d0, n)       # Transforms unknowns into model variables
    p_c = params['p_c']
    alpha     = params['alpha']
    beta      = params['beta']
    gamma     = params['gamma']
    tau_z     = params['tau_z']
    tau_w     = params['tau_w']
    epsilon_c = params['epsilon_c']
    epsilon_d = params['epsilon_d']
    r         = params['r']
    phi       = params['phi']
    l         = params['l']

    # Household FOCs
    hh_focs = households_focs_array(c, d, ell, p_c, p_d, w,
                                    alpha, beta, gamma, d0, mult, phi)
    # Firm FOCs
    firmC = firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r)
    firmD = firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r)
    
    # Market clearing
    mkt = market_clearing(c, d, ell, t_c, z_c, t_d, z_d, epsilon_c, epsilon_d, r)
    
    # Household budgets
    tax_revenue = np.sum(tau_w * (w * (1 - ell) * phi)) + tau_z * (z_c + z_d)
    hh_budg = households_budget_array(c, d, ell, p_c, p_d, w, phi, tau_w, l, tax_revenue)
    hh_budg = hh_budg[1:]                                                     # We omit the first budget constraint

    # Concatenate all equations
    return np.concatenate((hh_focs, firmC, firmD, mkt, hh_budg)), tax_revenue

##############################################################################################################################################################
# 6) Main Solve Function
##############################################################################################################################################################
def main_solve(tau_w, tau_z, l, n):                                           # Defines the main function
    """
    Solve the full system for a given tau_z with n households.
    """
    # Set base parameters
    params = {
        'alpha':     0.7,
        'beta':      0.1,
        'gamma':     0.1,
        'd0':        0.01,
        'r':         0.5,
        'p_c':       1.0,
        'epsilon_c': 0.8,
        'epsilon_d': 0.6,
        'tau_z':     tau_z,
        'tau_w':     tau_w,
        'l':         l,
    }
    # Define a vector of household productivities
    phi = np.array([0.7, 0.8, 0.9, 1.2, 1.4])
    params['phi'] = phi

    # Total unknowns: 26 for 5 households
    u0 = np.zeros(26)                                                         # Initial guess
    u0[0:5] = np.log(1.0)
    u0[5:10] = np.log(1.0)
    u0[10:15] = 0.0
    u0[15:20] = 1.0
    u0[20] = 0.0
    u0[21] = 0.0
    u0[22] = 0.0
    u0[23] = 0.0
    u0[24] = np.log(1.0)
    u0[25] = np.log(1.0)

    # Solve
    sol = root(lambda x: full_system(x, params, n)[0], u0, method='lm', tol=1e-15)
    final_res, tax_revenue = full_system(sol.x, params, n)
    resid_norm = np.linalg.norm(final_res)
    converged = sol.success

    # Transform solution
    c, d, ell, mult, t_c, t_d, z_c, z_d, p_d, w = transform(sol.x, params['d0'], n)

    # Calculate utilities
    utilities = np.empty(n)
    for i in range(n):
        utilities[i] = (params['alpha'] * np.log(c[i]) +
                        params['beta']  * np.log(d[i] - params['d0']) +
                        params['gamma'] * np.log(ell[i]))

    # Calculate total pollution
    aggregate_polluting = np.sum(d)

    # Compute firm outputs for profit calculation
    y_c = firm_c_production(t_c, z_c, params['epsilon_c'], params['r'])
    y_d = firm_d_production(t_d, z_d, params['epsilon_d'], params['r'])

    # Compute profits
    profit_c = params['p_c'] * y_c - (w * t_c + params['tau_z'] * z_c)
    profit_d = p_d * y_d - (w * t_d + params['tau_z'] * z_d)

    # === PRINTING OUTPUT IN A STYLE SIMILAR TO full_het_solution.py ===
    print(f"=== Solving for tau_z={tau_z:.4f} with {n} households ===")
    print("Success:", sol.success)
    print("Message:", sol.message)
    print(f"Final residual norm: {resid_norm:.2e}")

    print("\n================== HOUSEHOLD RESULTS ==================")
    for i in range(n):
        print(f"Household {i+1}:")
        print(f"  Productivity (phi): {phi[i]:.4f}")
        print(f"  Clean good (c):      {c[i]:.4f}")
        print(f"  Polluting good (d):  {d[i]:.4f}")
        print(f"  Leisure (ell):       {ell[i]:.4f}")
        print(f"  Multiplier:          {mult[i]:.4f}")
        print("------------------------------------------------------")

    print("\n==================== FIRM C (CLEAN GOOD) ===================")
    print(f"  Labor (t_c):     {t_c:.4f}")
    print(f"  Pollution (z_c): {z_c:.20f}")
    print(f"  Output (y_c):    {y_c:.4f}")
    print(f"  Profit:          {profit_c:.4f}")

    print("\n==================== FIRM D (POLLUTING GOOD) ===================")
    print(f"  Labor (t_d):     {t_d:.4f}")
    print(f"  Pollution (z_d): {z_d:.20f}")
    print(f"  Output (y_d):    {y_d:.4f}")
    print(f"  Profit:          {profit_d:.4f}")

    print("\n==================== MARKET PRICES ============================")
    print(f"  p_d = {p_d:.4f}   (Price of the polluting good)")
    print(f"  w   = {w:.4f}    (Wage)")

    print("\n=============== FOC / CONSTRAINT RESIDUALS ============")
    print("  - Order of residuals in final_res:")
    print("    1) Household FOCs (c, d, ell), for each i in [1..n].")
    print("    2) Firm C's FOCs: labor (t_c), input (z_c).")
    print("    3) Firm D's FOCs: labor (t_d), input (z_d).")
    print("    4) Market clearing conditions (c, d, labor).")
    print("    5) Household budgets (n-1).")
    print(final_res)
    print("=======================================================")

    p_c = params['p_c']

    excluded_bc = households_budget_array(c, d, ell, p_c, p_d, w, phi, tau_w, l, tax_revenue)[0]
    print(f"Unused budget constraint (household 1): {excluded_bc:.20f}")

    return utilities, aggregate_polluting, converged

if __name__ == "__main__":                                                    # Runs this code if the script is executed
    l_vector = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    phi_vector = np.array([0.7, 0.8, 0.9, 1.2, 1.4])                   # Vector of lump-sum transfers
    tau_w_vector = np.array([0.10, 0.12, 0.15, 0.18, 0.20])                  # Vector of lump-sum transfers
    main_solve(tau_w=tau_w_vector, tau_z=3, l=l_vector, n=5)
