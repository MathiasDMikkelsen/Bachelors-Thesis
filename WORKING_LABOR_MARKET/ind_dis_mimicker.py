import numpy as np
from scipy.optimize import root, minimize
import copy

# ==============================================================
# Model 1: Segmented Labor Market (Function Definition for 3+2 Structure)
# ==============================================================

# Define base parameters for the segmented model (3+2 structure)
params_seg = {
    'alpha': 0.7, 'beta': 0.2, 'gamma': 0.2,
    't': 24.0, 'd0': 0.5, 'p_c': 1.0,
    # Fixed values (will not be changed by optimization)
    'epsilon_c_fixed': 0.995,
    'r_c_fixed': -1.0,
    'r_d_fixed': -1.0,
    # Base values for OPTIMIZED parameters (used for initial guess)
    'epsilon_d_base': 0.92,
    'phi_d_base_norm': np.array([0.164465/2, 0.327578/2, (1-0.164465-0.327578)/2]), # Length 3, sums to 1.0
    'phi_c_base_norm': np.array([0.319319/2, (1-0.319319)/2]),         # Length 2, sums to 1.0
    'n': 5 # Total households
}
# Update n_d and n_c based on the structure
params_seg['n_d'] = len(params_seg['phi_d_base_norm']) # Should be 3
params_seg['n_c'] = len(params_seg['phi_c_base_norm']) # Should be 2
print(f"Segmented Model Base Params Set for {params_seg['n_d']} Dirty + {params_seg['n_c']} Clean Structure.")


# --- solve_segmented function (accepts parameters, uses structure defined in params) ---
# (Assumed unchanged from previous version - flexible enough)
def solve_segmented(epsilon_c_val, epsilon_d_val, r_c_val, r_d_val, phi_d_val, phi_c_val, tau_w, tau_z, g, params):
    """Solves segmented model (structure based on params), taking specific eps, r, phi values."""
    n_d_expected = params['n_d']
    n_c_expected = params['n_c']
    if len(phi_d_val) != n_d_expected or len(phi_c_val) != n_c_expected:
        raise ValueError(f"Input phi dimensions mismatch params definition (Expected d:{n_d_expected}, c:{n_c_expected}; Got d:{len(phi_d_val)}, c:{len(phi_c_val)})")
    local_params = copy.deepcopy(params)
    local_params['epsilon_c'] = epsilon_c_val
    local_params['epsilon_d'] = epsilon_d_val
    phi_full_model = np.concatenate([phi_d_val, phi_c_val])
    n = local_params['n']
    n_d = local_params['n_d']
    n_c = local_params['n_c']
    if len(tau_w) != n:
         raise ValueError(f"tau_w length ({len(tau_w)}) != n ({n}) in solve_segmented")

    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y
        z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)
        epsilon_c = local_params['epsilon_c']; epsilon_d = local_params['epsilon_d']
        r_c = r_c_val; r_d = r_d_val
        d0 = local_params['d0']; alpha = local_params['alpha']
        beta = local_params['beta']; gamma = local_params['gamma']
        p_c = local_params['p_c']; t_time = local_params['t']
        f_c = (epsilon_c * ((t_c + 1e-9)**r_c) + (1 - epsilon_c) * ((z_c + 1e-9)**r_c))**(1/r_c)
        f_d = (epsilon_d * ((t_d + 1e-9)**r_d) + (1 - epsilon_d) * ((z_d + 1e-9)**r_d))**(1/r_d)
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Uses n_d=3, n_c=2
        Y_tilde = phi_full_model * wage * (1 - tau_w) * t_time + l - p_d * d0
        if np.any(Y_tilde <= 1e-9): return np.full(8, 1e6)
        denom_shares = alpha + beta + gamma
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_tilde + d0
        price_leisure = phi_full_model * wage * (1 - tau_w)
        if np.any(price_leisure <= 1e-9): return np.full(8, 1e6)
        l_agents = (gamma / (denom_shares * price_leisure)) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t_time - 1e-9)
        labor_supply_agents = t_time - l_agents
        agg_labor_d = np.sum(phi_d_val * labor_supply_agents[:n_d])
        agg_labor_c = np.sum(phi_c_val * labor_supply_agents[n_d:])
        agg_d = np.sum(d_agents)
        eq1 = t_c - agg_labor_c; eq2 = t_d - agg_labor_d
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c)); eq4 = w_c - MP_L_c
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r_c - 1)) * (f_c**(1 - r_c)); eq5 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d)); eq6 = w_d - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r_d - 1)) * (f_d**(1 - r_d)); eq7 = tau_z - MP_Z_d * p_d
        total_wage_tax_revenue = np.sum(tau_w * phi_full_model * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8, 'maxfev': 4000})
    if not sol.success: return None, False
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])
    phi_full_model = np.concatenate([phi_d_val, phi_c_val])
    Y_tilde = phi_full_model * wage * (1 - tau_w) * params['t'] + l - p_d * params['d0']
    if np.any(Y_tilde <= 1e-9): return None, False
    denom_shares = params['alpha'] + params['beta'] + params['gamma']
    price_leisure = phi_full_model * wage * (1 - tau_w)
    if np.any(price_leisure <= 1e-9): return None, False
    l_agents = (params['gamma'] / (denom_shares * price_leisure)) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, params['t'] - 1e-9)
    labor_supply_agents = params['t'] - l_agents
    net_labor_income = phi_full_model * wage * (1 - tau_w) * labor_supply_agents
    total_disposable_income = net_labor_income + l
    return total_disposable_income, sol.success
# --- END solve_segmented function ---


# ==============================================================
# Optimization Setup (for 3 Dirty + 2 Clean, optimizing eps_d, phi_d, phi_c)
# ==============================================================

# Define fixed inputs for the comparison
tau_w_input_std = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_input = tau_w_input_std # Use standard taxes
if len(tau_w_input) != params_seg['n']:
     raise ValueError("tau_w_input length mismatch")
print(f"Using tau_w: {tau_w_input} (First {params_seg['n_d']} for Dirty, Last {params_seg['n_c']} for Clean)")

tau_z_input = 0.1
g_input = 5.0
print(f"Using fixed inputs: tau_z = {tau_z_input}, g = {g_input}")

# Retrieve fixed parameter values from params_seg
epsilon_c_fixed = params_seg['epsilon_c_fixed']
r_c_fixed       = params_seg['r_c_fixed']
r_d_fixed       = params_seg['r_d_fixed']
# NOTE: phi_c_base_norm is now only used for the *initial guess*, not fixed in the objective function
print(f"Keeping fixed: eps_c={epsilon_c_fixed}, r_c={r_c_fixed}, r_d={r_d_fixed}")


# 1. Define the Target Income Distribution SHARES
# Shares for 3 Dirty + 2 Clean households
target_income_shares = np.array([
    0.03690,  # HH 1 (Dirty) share
    0.09420,  # HH 2 (Dirty) share
    0.1529,  # HH 3 (Dirty) share
    0.2363,  # HH 4 (Clean) share
    0.4798   # HH 5 (Clean) share
])
if len(target_income_shares) != params_seg['n']:
     raise ValueError("target_income_shares length mismatch")
print(f"Target Income Shares (for 3D+2C): {np.round(target_income_shares, 4)}")


# 2. Define the Objective Function (adapted for 4 variables, 3+2 structure)
def objective_function_shares_4var_3_2(x, target_shares, base_params_seg, fixed_eps_c, fixed_r_c, fixed_r_d, tau_w, tau_z, g):
    """
    Objective function comparing shares for 3+2 model.
    Optimizes epsilon_d, phi_d_norm, AND phi_c_norm.
    x = [eps_d, phi_d1_norm, phi_d2_norm, phi_c1_norm] # indices 0-3
    """
    epsilon_d_opt = x[0]
    phi_d1_norm   = x[1]
    phi_d2_norm   = x[2]
    phi_c1_norm   = x[3] # Added phi_c1

    # --- Input Validation and Reconstruction ---
    # Basic bounds check
    if not (1e-6 < epsilon_d_opt < 1.0 - 1e-6 and
            1e-6 < phi_d1_norm < 1.0 - 1e-6 and
            1e-6 < phi_d2_norm < 1.0 - 1e-6 and
            1e-6 < phi_c1_norm < 1.0 - 1e-6): # Added check for phi_c1
        return 1e10

    # Constraint check for phi_d sum (phi_d3 must be > 0)
    phi_d3_norm = 1.0 - phi_d1_norm - phi_d2_norm
    if phi_d3_norm < 1e-6: return 1e10

    # Constraint check for phi_c sum (phi_c2 must be > 0) - Added
    phi_c2_norm = 1.0 - phi_c1_norm
    if phi_c2_norm < 1e-6: return 1e10

    # Reconstruct the normalized phi vectors
    phi_d_opt = np.array([phi_d1_norm, phi_d2_norm, phi_d3_norm]) # Length 3
    phi_c_opt = np.array([phi_c1_norm, phi_c2_norm])             # Length 2 (NOW reconstructed)

    # Check lengths match parameter definition
    if len(phi_d_opt) != base_params_seg['n_d'] or len(phi_c_opt) != base_params_seg['n_c']:
        print(f"Runtime Error: Reconstructed phi length mismatch in objective")
        return 1e12

    # --- Run Segmented Model ---
    current_income, converged = solve_segmented(
        fixed_eps_c, epsilon_d_opt, fixed_r_c, fixed_r_d,
        phi_d_opt, phi_c_opt, # Pass reconstructed phi vectors
        tau_w, tau_z, g, base_params_seg
    )

    # --- Calculate Error based on SHARES ---
    if not converged or current_income is None: return 1e10
    current_total_income = np.sum(current_income)
    if current_total_income <= 1e-6: return 1e10
    current_income_shares = current_income / current_total_income
    if len(current_income_shares) != len(target_shares):
         print("Runtime Error: Share vector length mismatch")
         return 1e12
    error = np.sum((current_income_shares - target_shares)**2)
    # print(f"x={np.round(x,6)} -> Err={error:.4e}") # Optional progress print
    return error


# 3. Set up Bounds and Constraints (adapted for 4 variables)
# Variables: [eps_d, phi_d1_norm, phi_d2_norm, phi_c1_norm]
bounds = [
    (1e-6, 1.0 - 1e-6), # epsilon_d
    (1e-6, 1.0 - 1e-6), # phi_d1_norm
    (1e-6, 1.0 - 1e-6), # phi_d2_norm
    (1e-6, 1.0 - 1e-6)  # phi_c1_norm  -- Added bounds for phi_c1
]
# Constraint 1: phi_d1 + phi_d2 <= 1.0 - 1e-6 (ensures phi_d3 >= 1e-6)
# Constraint 2: phi_c1 <= 1.0 - 1e-6 (ensures phi_c2 >= 1e-6) - THIS IS COVERED BY BOUNDS
# We only need the constraint for phi_d sum.
constraints = ({'type': 'ineq', 'fun': lambda x: (1.0 - 1e-6) - x[1] - x[2]}) # Constraint on phi_d using x[1], x[2]


# 4. Initial Guess (x0) (adapted for 4 variables)
initial_epsilon_d   = params_seg['epsilon_d_base']
initial_phi_d1_norm = params_seg['phi_d_base_norm'][0]
initial_phi_d2_norm = params_seg['phi_d_base_norm'][1]
initial_phi_c1_norm = params_seg['phi_c_base_norm'][0] # Add initial guess for phi_c1
x0 = [initial_epsilon_d, initial_phi_d1_norm, initial_phi_d2_norm, initial_phi_c1_norm]
print(f"\nInitial Guess (x0): {np.round(x0, 4)}")
print(f"(Based on params_seg: eps_d={initial_epsilon_d:.2f}, phi_d_base={params_seg['phi_d_base_norm']}, phi_c_base={params_seg['phi_c_base_norm']})")


# 5. Run the Optimization
print("\nStarting optimization (3+2 structure, Optimizing eps_d, phi_d, AND phi_c)...")
opt_result = minimize(
    objective_function_shares_4var_3_2, # Use the 4-variable objective function
    x0,
    # UPDATED args: removed fixed_phi_c_norm
    args=(target_income_shares, params_seg, epsilon_c_fixed, r_c_fixed, r_d_fixed, tau_w_input, tau_z_input, g_input),
    method='SLSQP',
    bounds=bounds, # Use updated bounds (length 4)
    constraints=constraints, # Use updated constraints
    options={'disp': True, 'ftol': 1e-10, 'maxiter': 500}
)

# 6. Analyze Results (adapted for 4 variables)
print("\nOptimization finished.")
print(f"Success: {opt_result.success}")
print(f"Message: {opt_result.message}")
print(f"Number of Iterations: {opt_result.nit}")
print(f"Final Objective Function Value (Sum of Squared Share Errors): {opt_result.fun:.6e}")

if opt_result.success:
    optimal_x = opt_result.x
    print("\nOptimal Parameters Found:")
    print(f"  epsilon_d   = {optimal_x[0]:.6f}")
    print(f"  phi_d1_norm = {optimal_x[1]:.6f}")
    print(f"  phi_d2_norm = {optimal_x[2]:.6f}")
    print(f"  phi_c1_norm = {optimal_x[3]:.6f}") # Added phi_c1
    print("\nFixed Parameters Used:")
    print(f"  epsilon_c = {epsilon_c_fixed:.6f}")
    print(f"  r_c       = {r_c_fixed:.6f}")
    print(f"  r_d       = {r_d_fixed:.6f}")


    # Reconstruct the optimal phi vectors
    opt_phi_d1 = optimal_x[1]
    opt_phi_d2 = optimal_x[2]
    opt_phi_d3 = 1.0 - opt_phi_d1 - opt_phi_d2
    opt_phi_d = np.array([opt_phi_d1, opt_phi_d2, opt_phi_d3]) # Length 3

    opt_phi_c1 = optimal_x[3]
    opt_phi_c2 = 1.0 - opt_phi_c1
    opt_phi_c = np.array([opt_phi_c1, opt_phi_c2]) # Length 2 (NOW optimal)

    print(f"\n  Reconstructed Optimal phi_d (normalized, len={len(opt_phi_d)}): {np.round(opt_phi_d, 6)}")
    print(f"  Reconstructed Optimal phi_c (normalized, len={len(opt_phi_c)}): {np.round(opt_phi_c, 6)}") # NOW optimal
    print(f"  Sum check: phi_d sum = {np.sum(opt_phi_d):.6f}, phi_c sum = {np.sum(opt_phi_c):.6f}")

    # Verify by running the segmented model with optimal and fixed params
    print("\nVerifying: Running segmented model with optimal & fixed parameters...")
    final_income_seg, final_converged_seg = solve_segmented(
        epsilon_c_fixed, optimal_x[0], r_c_fixed, r_d_fixed,
        opt_phi_d, opt_phi_c, # Pass reconstructed optimal phi vectors
        tau_w_input, tau_z_input, g_input, params_seg
    )
    if final_converged_seg:
        final_total_income = np.sum(final_income_seg)
        if final_total_income > 1e-6:
            final_income_shares = final_income_seg / final_total_income
            print(f"Target Income Shares:   {np.round(target_income_shares * 100, 2)} %")
            print(f"Final Segmented Shares: {np.round(final_income_shares * 100, 2)} %")
            final_error = np.sum((final_income_shares - target_income_shares)**2)
            print(f"Final SSE (Shares) Check: {final_error:.6e}")
        else:
            print("Verification failed: Final total income is zero or negative.")
    else:
        print("Verification failed: Segmented model did not converge with optimal parameters.")
else:
    print("\nOptimization failed to find a solution.")