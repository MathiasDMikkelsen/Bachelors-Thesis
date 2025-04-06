import numpy as np
from scipy.optimize import root, minimize
import copy

# ==============================================================
# Model 1: Segmented Labor Market (Function Definition for 4+1 Structure, r_c/r_d args)
# ==============================================================

# Define base parameters for the segmented model (4+1 structure)
# Includes FIXED values for parameters NOT being optimized
params_seg = {
    'alpha': 0.7, 'beta': 0.2, 'gamma': 0.2, #'r': -1.0, # Removed
    't': 24.0, 'd0': 0.5, 'p_c': 1.0,
    # Fixed values (will not be changed by this optimization run)
    'epsilon_c_fixed': 0.995,
    'r_c_fixed': -1.0,
    'r_d_fixed': -1.0,
    # Base values for OPTIMIZED parameters (used for initial guess)
    'epsilon_d_base': 0.92, # Using value from user code context
    'phi_d_base_norm': np.array([0.1, 0.2, 0.3, 0.4]), # Sums to 1.0
    'phi_c_base_norm': np.array([1.0]),                # Sums to 1.0 (Fixed at [1.0])
    'n': 5 # Total households still 5
}
# Update n_d and n_c based on the structure
params_seg['n_d'] = len(params_seg['phi_d_base_norm']) # Should be 4
params_seg['n_c'] = len(params_seg['phi_c_base_norm']) # Should be 1

# --- solve_segmented function (accepts all parameters, uses 4+1 structure) ---
# (Ensure the function definition from the previous step, which accepts
# eps_c, eps_d, r_c, r_d, phi_d, phi_c is pasted here)
def solve_segmented(epsilon_c_val, epsilon_d_val, r_c_val, r_d_val, phi_d_val, phi_c_val, tau_w, tau_z, g, params):
    """Solves segmented model (4+1 structure), taking specific eps, r, phi values."""
    if len(phi_d_val) != params['n_d'] or len(phi_c_val) != params['n_c']:
         raise ValueError(f"Input phi dimensions mismatch params definition (Expected d:{params['n_d']}, c:{params['n_c']})")

    local_params = copy.deepcopy(params)
    # Use the specific epsilon and r values passed to the function
    local_params['epsilon_c'] = epsilon_c_val
    local_params['epsilon_d'] = epsilon_d_val
    current_r_c = r_c_val
    current_r_d = r_d_val

    phi_full_model = np.concatenate([phi_d_val, phi_c_val]) # Still sums to 2

    n = local_params['n']
    n_d = local_params['n_d'] # Should be 4
    n_c = local_params['n_c'] # Should be 1

    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y
        z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)
        epsilon_c = local_params['epsilon_c']; epsilon_d = local_params['epsilon_d']
        r_c = current_r_c; r_d = current_r_d # Use the passed r values
        d0 = local_params['d0']; alpha = local_params['alpha']
        beta = local_params['beta']; gamma = local_params['gamma']
        p_c = local_params['p_c']; t_time = local_params['t']

        f_c = (epsilon_c * ((t_c + 1e-9)**r_c) + (1 - epsilon_c) * ((z_c + 1e-9)**r_c))**(1/r_c)
        f_d = (epsilon_d * ((t_d + 1e-9)**r_d) + (1 - epsilon_d) * ((z_d + 1e-9)**r_d))**(1/r_d)
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Uses n_d=4, n_c=1
        Y_tilde = phi_full_model * wage * (1 - tau_w) * t_time + l - p_d * d0
        if np.any(Y_tilde <= 1e-9): return np.full(8, 1e6)
        denom_shares = alpha + beta + gamma
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_tilde + d0
        price_leisure = phi_full_model * wage * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t_time - 1e-9)
        labor_supply_agents = t_time - l_agents
        agg_labor_d = np.sum(phi_d_val * labor_supply_agents[:n_d]) # Use passed phi_d_val
        agg_labor_c = np.sum(phi_c_val * labor_supply_agents[n_d:]) # Use passed phi_c_val
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
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8})
    if not sol.success: return None, False
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)]) # Uses n_d=4, n_c=1
    phi_full_model = np.concatenate([phi_d_val, phi_c_val]) # Sums to 2
    Y_tilde = phi_full_model * wage * (1 - tau_w) * params['t'] + l - p_d * params['d0']
    denom_shares = params['alpha'] + params['beta'] + params['gamma']
    price_leisure = phi_full_model * wage * (1 - tau_w)
    l_agents = (params['gamma'] / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, params['t'] - 1e-9)
    labor_supply_agents = params['t'] - l_agents
    net_labor_income = phi_full_model * wage * (1 - tau_w) * labor_supply_agents
    total_disposable_income = net_labor_income + l
    return total_disposable_income, sol.success
# --- END solve_segmented function ---


# ==============================================================
# Optimization Setup
# ==============================================================

# Define fixed inputs for the comparison
tau_w_input = np.array([-1.75, -0.5, 0.0, 0.25, 0.55]) # Example tau_w (still length 5)
if len(tau_w_input) != params_seg['n']:
     raise ValueError("tau_w_input length mismatch")

# Define tau_z and g
tau_z_input = 1.0
g_input = 5.0
print(f"Using fixed inputs: tau_z = {tau_z_input}, g = {g_input}")

# Retrieve fixed parameter values
epsilon_c_fixed = params_seg['epsilon_c_fixed']
r_c_fixed       = params_seg['r_c_fixed']
r_d_fixed       = params_seg['r_d_fixed']
print(f"Keeping fixed: eps_c={epsilon_c_fixed}, r_c={r_c_fixed}, r_d={r_d_fixed}")


# 1. Define the Target Income Distribution SHARES (using the last provided image)
target_income_shares = np.array([
    0.0522,  # Household 1 share
    0.1041,  # Household 2 share
    0.1573,  # Household 3 share
    0.2329,  # Household 4 share
    0.4535   # Household 5 share
])
print(f"Target Income Shares: {np.round(target_income_shares, 4)}")


# 2. Define the Objective Function (adapted for 4 variables, 4+1 structure)
def objective_function_shares_4var_4_1(x, target_shares, base_params_seg, fixed_eps_c, fixed_r_c, fixed_r_d, tau_w, tau_z, g):
    """
    Objective function comparing shares for 4+1 model.
    Optimizes epsilon_d and phi_d parameters ONLY.
    x = [eps_d, phi_d1, phi_d2, phi_d3] # indices 0-3
    """
    epsilon_d_opt = x[0]
    phi_d1_norm   = x[1]
    phi_d2_norm   = x[2]
    phi_d3_norm   = x[3]

    # --- Input Validation and Reconstruction ---
    # Basic bounds check
    if not (1e-6 < epsilon_d_opt < 1.0 - 1e-6 and
            1e-6 < phi_d1_norm < 1.0 - 1e-6 and
            1e-6 < phi_d2_norm < 1.0 - 1e-6 and
            1e-6 < phi_d3_norm < 1.0 - 1e-6):
        return 1e10

    # Constraint check for phi_d sum
    phi_d4_norm = 1.0 - phi_d1_norm - phi_d2_norm - phi_d3_norm
    if phi_d4_norm < 1e-6: return 1e10

    # Reconstruct the normalized phi vectors
    phi_d_opt = np.array([phi_d1_norm, phi_d2_norm, phi_d3_norm, phi_d4_norm]) # Length 4
    phi_c_opt = np.array([1.0])                                                # Length 1 (fixed)

    # Check lengths match parameter definition
    if len(phi_d_opt) != base_params_seg['n_d'] or len(phi_c_opt) != base_params_seg['n_c']:
        print(f"Runtime Error: Reconstructed phi length mismatch")
        return 1e12

    # --- Run Segmented Model ---
    # Pass FIXED values for eps_c, r_c, r_d
    current_income, converged = solve_segmented(
        fixed_eps_c, epsilon_d_opt, fixed_r_c, fixed_r_d, # Pass fixed & opt values
        phi_d_opt, phi_c_opt, tau_w, tau_z, g, base_params_seg
    )

    # --- Calculate Error based on SHARES ---
    if not converged or current_income is None: return 1e10
    current_total_income = np.sum(current_income)
    if current_total_income <= 1e-6: return 1e10
    current_income_shares = current_income / current_total_income
    error = np.sum((current_income_shares - target_shares)**2)
    # Optional: print progress
    # print(f"x={np.round(x,6)} -> Err={error:.4e}")
    return error


# 3. Set up Bounds and Constraints (adapted for 4 variables)
# Variables: [eps_d, phi_d1, phi_d2, phi_d3]
bounds = [
    (1e-6, 1.0 - 1e-6), # epsilon_d
    (1e-6, 1.0 - 1e-6), # phi_d1_norm
    (1e-6, 1.0 - 1e-6), # phi_d2_norm
    (1e-6, 1.0 - 1e-6)  # phi_d3_norm
]
# Constraint: phi_d1 + phi_d2 + phi_d3 <= 1.0 - 1e-6 (ensures phi_d4 >= 1e-6)
# Indices: x[1], x[2], x[3]
constraints = ({'type': 'ineq', 'fun': lambda x: (1.0 - 1e-6) - x[1] - x[2] - x[3]})


# 4. Initial Guess (x0) (adapted for 4 variables)
initial_epsilon_d   = params_seg['epsilon_d_base']
initial_phi_d1_norm = params_seg['phi_d_base_norm'][0]
initial_phi_d2_norm = params_seg['phi_d_base_norm'][1]
initial_phi_d3_norm = params_seg['phi_d_base_norm'][2]
x0 = [initial_epsilon_d, initial_phi_d1_norm, initial_phi_d2_norm, initial_phi_d3_norm]
print(f"\nInitial Guess (x0): {np.round(x0, 4)}")
print(f"(Based on params_seg: eps_d={initial_epsilon_d:.2f}, phi_d_base={params_seg['phi_d_base_norm']})")


# 5. Run the Optimization
print("\nStarting optimization (4+1 structure, Optimizing ONLY eps_d, phi_d)...")
opt_result = minimize(
    objective_function_shares_4var_4_1, # Use the 4-variable objective function
    x0,
    # Pass FIXED parameters needed by objective function
    args=(target_income_shares, params_seg, epsilon_c_fixed, r_c_fixed, r_d_fixed, tau_w_input, tau_z_input, g_input),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
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
    print(f"  phi_d3_norm = {optimal_x[3]:.6f}")
    print("\nFixed Parameters Used:")
    print(f"  epsilon_c = {epsilon_c_fixed:.6f}")
    print(f"  r_c       = {r_c_fixed:.6f}")
    print(f"  r_d       = {r_d_fixed:.6f}")


    # Reconstruct the optimal phi vectors
    opt_phi_d1 = optimal_x[1]
    opt_phi_d2 = optimal_x[2]
    opt_phi_d3 = optimal_x[3]
    opt_phi_d4 = 1.0 - opt_phi_d1 - opt_phi_d2 - opt_phi_d3
    opt_phi_d = np.array([opt_phi_d1, opt_phi_d2, opt_phi_d3, opt_phi_d4]) # Length 4
    opt_phi_c = np.array([1.0]) # Length 1, fixed

    print(f"\n  Reconstructed Optimal phi_d (normalized, len={len(opt_phi_d)}): {np.round(opt_phi_d, 6)}")
    print(f"  Reconstructed Optimal phi_c (normalized, len={len(opt_phi_c)}): {np.round(opt_phi_c, 6)}")

    # Verify by running the segmented model with optimal and fixed params
    print("\nVerifying: Running segmented model with optimal & fixed parameters...")
    final_income_seg, final_converged_seg = solve_segmented(
        epsilon_c_fixed, optimal_x[0], r_c_fixed, r_d_fixed, # Pass fixed and optimal eps/r values
        opt_phi_d, opt_phi_c, tau_w_input, tau_z_input, g_input, params_seg
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