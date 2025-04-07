import numpy as np
from scipy.optimize import root, minimize
import copy
import warnings # To suppress potential runtime warnings if needed

# Suppress RuntimeWarnings that might occur during optimization (e.g., invalid value in sqrt)
# warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================
# Model 1: Segmented Labor Market (Function Definition for 3+2 Structure)
# ==============================================================

# Define base parameters for the segmented model (3+2 structure)
params_seg = {
    'alpha': 0.7, 'beta': 0.2, 'gamma': 0.2,
    't': 24.0,
    # d0: Role changed - Used as INITIAL GUESS for the endogenous d0 solver.
    'd0': 0.5,
    'p_c': 1.0,
    # Fixed values (will not be changed by optimization)
    'epsilon_c_fixed': 0.995,
    'r_c_fixed': -1.0,
    'r_d_fixed': -1.0,
    # Base values for OPTIMIZED parameters (used only if needed, e.g., if updated guess fails)
    'epsilon_d_base': 0.92,
    'phi_d_base_norm': np.array([0.164465, 0.327578, (1-0.164465-0.327578)]),
    'phi_c_base_norm': np.array([0.319319, (1-0.319319)]),
    'n': 5 # Total households
}
# Update n_d and n_c based on the structure
params_seg['n_d'] = len(params_seg['phi_d_base_norm']) # Should be 3
params_seg['n_c'] = len(params_seg['phi_c_base_norm']) # Should be 2
print(f"Segmented Model Base Params Set for {params_seg['n_d']} Dirty + {params_seg['n_c']} Clean Structure.")
print(f"NOTE: params_seg['d0'] = {params_seg['d0']} is now used as the INITIAL GUESS for endogenous d0.")


# --- solve_segmented function (MODIFIED for Endogenous d0 AND to return d0) ---
def solve_segmented(epsilon_c_val, epsilon_d_val, r_c_val, r_d_val, phi_d_val, phi_c_val, tau_w, tau_z, g, params):
    """
    Solves segmented model (structure based on params), taking specific eps, r, phi values.
    MODIFIED: d0 is now endogenous (d0 = 0.44 * mean(d_agents)).
    MODIFIED: Returns income, solved_d0, convergence_status
    """
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

    # --- System of Equations (Now 9 variables including d0) ---
    def system_eqns(y):
        # Unpack 9 variables: 8 original + d0
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l, d0_var = y # d0_var is the ENDOGENOUS d0

        # --- Input Checks & Transformations ---
        # Check for non-physical values during iteration
        if p_d <= 1e-6 or w_c <= 1e-6 or w_d <= 1e-6 or d0_var < 0:
            return np.full(9, 1e6) # Penalize bad price/wage/d0 guesses

        try:
            z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)
        except OverflowError:
            return np.full(9, 1e7) # Penalize if exp overflows

        # Use parameters from local_params or passed arguments
        epsilon_c = local_params['epsilon_c']; epsilon_d = local_params['epsilon_d']
        # Note: r_c_val and r_d_val are passed directly to solve_segmented
        alpha = local_params['alpha']
        beta = local_params['beta']; gamma = local_params['gamma']
        p_c = local_params['p_c']; t_time = local_params['t']

        # --- Core Model Calculations (using d0_var) ---
        # Add small epsilon for numerical stability in CES/logs etc.
        t_c_eff = max(t_c, 1e-9); t_d_eff = max(t_d, 1e-9)
        z_c_eff = max(z_c, 1e-9); z_d_eff = max(z_d, 1e-9)
        p_d_eff = max(p_d, 1e-9)
        d0_eff  = max(d0_var, 0) # Ensure d0 used in Y_tilde is non-negative

        # Firms
        # Add check for r_c_val/r_d_val being near zero if needed, assume they are valid non-zero here
        f_c = (epsilon_c * (t_c_eff**r_c_val) + (1 - epsilon_c) * (z_c_eff**r_c_val))**(1/r_c_val)
        f_d = (epsilon_d * (t_d_eff**r_d_val) + (1 - epsilon_d) * (z_d_eff**r_d_val))**(1/r_d_val)

        # Households
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])
        # Y_tilde calculation now uses the VARIABLE d0_var (via d0_eff)
        Y_tilde = phi_full_model * wage * (1 - tau_w) * t_time + l - p_d_eff * d0_eff
        if np.any(Y_tilde <= 1e-9): return np.full(9, 1e6)

        denom_shares = alpha + beta + gamma
        # d_agents calculation also uses the VARIABLE d0_var (via d0_eff)
        d_agents = (beta / (p_d_eff * denom_shares)) * Y_tilde + d0_eff
        if np.any(d_agents < 0): return np.full(9, 1e6) # Demand shouldn't be negative

        price_leisure = phi_full_model * wage * (1 - tau_w)
        # Ensure price of leisure > 0 for division
        if np.any(price_leisure <= 1e-9): return np.full(9, 1e6)

        l_agents = (gamma / (denom_shares * price_leisure)) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t_time - 1e-9)
        labor_supply_agents = t_time - l_agents

        # Aggregation
        agg_labor_d = np.sum(phi_d_val * labor_supply_agents[:n_d])
        agg_labor_c = np.sum(phi_c_val * labor_supply_agents[n_d:])
        agg_d = np.sum(d_agents) # Aggregate demand depends on variable d0

        # --- System of Equations (9 Equations) ---
        # Eq1-8: Original system (using effective values for safety)
        eq1 = t_c - agg_labor_c; eq2 = t_d - agg_labor_d
        eq3 = (agg_d + 0.5 * g / p_d_eff) - f_d # Dirty good market clearing
        # Add checks for f_c/f_d being non-zero before division if r close to 1
        MP_L_c = epsilon_c * (t_c_eff**(r_c_val - 1)) * (f_c**(1 - r_c_val)); eq4 = w_c - MP_L_c
        MP_Z_c = (1 - epsilon_c) * (z_c_eff**(r_c_val - 1)) * (f_c**(1 - r_c_val)); eq5 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * (t_d_eff**(r_d_val - 1)) * (f_d**(1 - r_d_val)); eq6 = w_d - MP_L_d * p_d_eff
        MP_Z_d = (1 - epsilon_d) * (z_d_eff**(r_d_val - 1)) * (f_d**(1 - r_d_val)); eq7 = tau_z - MP_Z_d * p_d_eff
        total_wage_tax_revenue = np.sum(tau_w * phi_full_model * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c_eff + z_d_eff) # Use eff versions?
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g) # Gov budget

        # Eq9: Endogenous d0 condition: d0 = 0.44 * mean(d_agents)
        # Rearranged: d0 - 0.44 * sum(d_agents) / n = 0
        # Need d_agents calculated *within* this function based on current y guess
        avg_d_agents = agg_d / n
        eq9 = d0_var - 0.44 * avg_d_agents

        residuals = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9])

        # Final check for NaN/Inf in residuals before returning
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
             return np.full(9, 1e7)

        return residuals
    # --- End of system_eqns ---

    # Initial guess: Extend y0 to 9 elements, using params['d0'] for d0's initial guess
    y0 = np.array([
        5.0, 5.0,                         # t_c, t_d
        np.log(0.6), np.log(0.4),         # log_z_c, log_z_d
        0.5, 0.6,                         # w_c, w_d
        1.5,                              # p_d
        0.1,                              # l
        local_params.get('d0', 0.5)       # d0 initial guess (use value from params)
        ])

    # Solve the 9x9 system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8, 'maxfev': 5000}) # Increase maxfev slightly

    # --- Post-processing ---
    if not sol.success:
        # Return consistent failure signature
        return None, np.nan, False # income, d0, success_status

    # Unpack the full solution vector including the SOLVED d0
    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l, solved_d0 = sol.x

     # Basic check on solved d0 and other critical values
    if not np.isfinite(solved_d0) or solved_d0 < 0 or not np.isfinite(p_d) or p_d <= 1e-6 or not np.isfinite(w_c) or w_c <=1e-6 or not np.isfinite(w_d) or w_d <= 1e-6 :
        # Return consistent failure signature if solution values are bad
        return None, np.nan, False # income, d0, success_status

    # Recalculate final incomes using the SOLVED d0
    wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])
    # Need l_agents based on solved d0
    Y_tilde = phi_full_model * wage * (1 - tau_w) * params['t'] + l - p_d * solved_d0
    if np.any(Y_tilde <= 1e-9):
        return None, solved_d0, False # Return solved d0 even if Y_tilde fails here

    denom_shares = params['alpha'] + params['beta'] + params['gamma']
    price_leisure = phi_full_model * wage * (1 - tau_w)
    if np.any(price_leisure <= 1e-9):
        return None, solved_d0, False # Return solved d0 even if leisure price fails

    l_agents = (params['gamma'] / (denom_shares * price_leisure)) * Y_tilde
    l_agents = np.clip(l_agents, 1e-9, params['t'] - 1e-9)
    labor_supply_agents = params['t'] - l_agents

    # Calculate final disposable income (used by objective function)
    net_labor_income = phi_full_model * wage * (1 - tau_w) * labor_supply_agents
    total_disposable_income = net_labor_income + l

    # **** MODIFIED RETURN STATEMENT ****
    return total_disposable_income, solved_d0, sol.success
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

tau_z_input = 0.2
g_input = 5.0
print(f"Using fixed inputs: tau_z = {tau_z_input}, g = {g_input}")

# Retrieve fixed parameter values from params_seg
epsilon_c_fixed = params_seg['epsilon_c_fixed']
r_c_fixed       = params_seg['r_c_fixed']
r_d_fixed       = params_seg['r_d_fixed']
print(f"Keeping fixed: eps_c={epsilon_c_fixed}, r_c={r_c_fixed}, r_d={r_d_fixed}")


# 1. Define the Target Income Distribution SHARES
target_income_shares = np.array([
    0.0400,  # HH 1 (Dirty) share
    0.0961,  # HH 2 (Dirty) share
    0.1537,  # HH 3 (Dirty) share
    0.2356,  # HH 4 (Clean) share
    0.4746   # HH 5 (Clean) share
])
if len(target_income_shares) != params_seg['n']:
     raise ValueError("target_income_shares length mismatch")
print(f"Target Income Shares (for 3D+2C): {np.round(target_income_shares, 4)}")


# 2. Define the Objective Function (No changes needed - only uses income from solve)
def objective_function_shares_4var_3_2(x, target_shares, base_params_seg, fixed_eps_c, fixed_r_c, fixed_r_d, tau_w, tau_z, g):
    """
    Objective function comparing shares for 3+2 model. Relies on solve_segmented.
    Optimizes epsilon_d, phi_d_norm, AND phi_c_norm.
    x = [eps_d, phi_d1_norm, phi_d2_norm, phi_c1_norm] # indices 0-3
    """
    epsilon_d_opt = x[0]
    phi_d1_norm   = x[1]
    phi_d2_norm   = x[2]
    phi_c1_norm   = x[3]
    # --- Input Validation and Reconstruction ---
    if not (1e-6 < epsilon_d_opt < 1.0 - 1e-6 and
            1e-6 < phi_d1_norm < 1.0 - 1e-6 and
            1e-6 < phi_d2_norm < 1.0 - 1e-6 and
            1e-6 < phi_c1_norm < 1.0 - 1e-6): return 1e10
    phi_d3_norm = 1.0 - phi_d1_norm - phi_d2_norm
    if phi_d3_norm < 1e-6: return 1e10
    phi_c2_norm = 1.0 - phi_c1_norm
    if phi_c2_norm < 1e-6: return 1e10
    phi_d_opt = np.array([phi_d1_norm, phi_d2_norm, phi_d3_norm])
    phi_c_opt = np.array([phi_c1_norm, phi_c2_norm])
    if len(phi_d_opt) != base_params_seg['n_d'] or len(phi_c_opt) != base_params_seg['n_c']:
        print(f"Runtime Error: Reconstructed phi length mismatch in objective")
        return 1e12
    # --- Run Segmented Model (Now with endogenous d0 handled inside) ---
    # **** MODIFIED CALL: Unpack three return values ****
    current_income, _, converged = solve_segmented( # Ignore d0 return value here
        fixed_eps_c, epsilon_d_opt, fixed_r_c, fixed_r_d,
        phi_d_opt, phi_c_opt,
        tau_w, tau_z, g, base_params_seg
    )
    # --- Calculate Error based on SHARES ---
    if not converged or current_income is None: return 1e10 # Penalize non-convergence
    current_total_income = np.sum(current_income)
    if current_total_income <= 1e-6: return 1e10 # Penalize zero/negative total income
    current_income_shares = current_income / current_total_income
    if len(current_income_shares) != len(target_shares):
         print("Runtime Error: Share vector length mismatch")
         return 1e12
    error = np.sum((current_income_shares - target_shares)**2)
    # print(f"x={np.round(x,6)} -> Err={error:.4e}") # Optional progress print
    return error


# 3. Set up Bounds and Constraints (No changes needed)
bounds = [
    (1e-6, 1.0 - 1e-6), # epsilon_d
    (1e-6, 1.0 - 1e-6), # phi_d1_norm
    (1e-6, 1.0 - 1e-6), # phi_d2_norm
    (1e-6, 1.0 - 1e-6)  # phi_c1_norm
]
constraints = ({'type': 'ineq', 'fun': lambda x: (1.0 - 1e-6) - x[1] - x[2]}) # Constraint on phi_d using x[1], x[2]


# **** SECTION 4: Using Updated Initial Guess ****
# 4. Updated Initial Guess (x0) based on previous optimal results
# Values taken from image_4c073d.png
initial_epsilon_d   = 0.919471
initial_phi_d1_norm = 0.164491
initial_phi_d2_norm = 0.327725
initial_phi_c1_norm = 0.319479

x0 = [initial_epsilon_d, initial_phi_d1_norm, initial_phi_d2_norm, initial_phi_c1_norm]
# Check if the guess respects constraints (mainly sum for phi_d)
phi_d_sum_guess = initial_phi_d1_norm + initial_phi_d2_norm
if phi_d_sum_guess >= 1.0:
    print(f"WARNING: Initial guess phi_d1 + phi_d2 = {phi_d_sum_guess:.6f} >= 1.0. Adjusting slightly.")
    # Simple adjustment: reduce the larger component slightly
    if initial_phi_d1_norm > initial_phi_d2_norm:
        x0[1] -= 1e-5
    else:
        x0[2] -= 1e-5

print(f"\nUpdated Initial Guess (x0): {np.round(x0, 6)}")
# **** END SECTION 4 ****


# 5. Run the Optimization (No changes needed)
print("\nStarting optimization (3+2 structure, Endogenous d0, Optimizing eps_d, phi_d, AND phi_c)...")
opt_result = minimize(
    objective_function_shares_4var_3_2,
    x0,
    args=(target_income_shares, params_seg, epsilon_c_fixed, r_c_fixed, r_d_fixed, tau_w_input, tau_z_input, g_input),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'disp': True, 'ftol': 1e-10, 'maxiter': 500}
)

# 6. Analyze Results (Modified to receive and print d0)
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
    print(f"  phi_c1_norm = {optimal_x[3]:.6f}")
    print("\nFixed Parameters Used:")
    print(f"  epsilon_c = {epsilon_c_fixed:.6f}")
    print(f"  r_c       = {r_c_fixed:.6f}")
    print(f"  r_d       = {r_d_fixed:.6f}")


    # Reconstruct the optimal phi vectors
    opt_phi_d1 = optimal_x[1]; opt_phi_d2 = optimal_x[2]
    opt_phi_d3 = 1.0 - opt_phi_d1 - opt_phi_d2
    opt_phi_d = np.array([opt_phi_d1, opt_phi_d2, opt_phi_d3])

    opt_phi_c1 = optimal_x[3]; opt_phi_c2 = 1.0 - opt_phi_c1
    opt_phi_c = np.array([opt_phi_c1, opt_phi_c2])

    print(f"\n  Reconstructed Optimal phi_d (normalized, len={len(opt_phi_d)}): {np.round(opt_phi_d, 6)}")
    print(f"  Reconstructed Optimal phi_c (normalized, len={len(opt_phi_c)}): {np.round(opt_phi_c, 6)}")
    print(f"  Sum check: phi_d sum = {np.sum(opt_phi_d):.6f}, phi_c sum = {np.sum(opt_phi_c):.6f}")

    # Verify by running the segmented model with optimal and fixed params
    print("\nVerifying: Running segmented model (with endogenous d0) using optimal & fixed parameters...")
    # **** MODIFIED CALL: Unpack three return values ****
    final_income_seg, final_d0_seg, final_converged_seg = solve_segmented(
        epsilon_c_fixed, optimal_x[0], r_c_fixed, r_d_fixed,
        opt_phi_d, opt_phi_c,
        tau_w_input, tau_z_input, g_input, params_seg
    )
    if final_converged_seg and final_income_seg is not None:
        final_total_income = np.sum(final_income_seg)
        if final_total_income > 1e-6:
            final_income_shares = final_income_seg / final_total_income
            print(f"Target Income Shares:   {np.round(target_income_shares * 100, 2)} %")
            print(f"Final Segmented Shares: {np.round(final_income_shares * 100, 2)} %")
            final_error = np.sum((final_income_shares - target_income_shares)**2)
            print(f"Final SSE (Shares) Check: {final_error:.6e}")
            # **** ADDED PRINT STATEMENT for d0 ****
            print(f"Final Endogenous d0 value: {final_d0_seg:.6f}")
        else:
            print("Verification failed: Final total income is zero or negative.")
    else:
        print("Verification failed: Segmented model did not converge with optimal parameters.")
else:
    print("\nOptimization failed to find a solution.")