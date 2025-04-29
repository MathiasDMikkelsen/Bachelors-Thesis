import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.optimize import minimize
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

a_solvers_dir = os.path.join(project_root, "a_solvers")
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)
import a_solvers.outer_solver_ext as outer_solver
import a_solvers.inner_solver_ext as solver
from a_solvers.inner_solver_ext import n, alpha, beta, gamma, d0, phi, t as T
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}  % give mathpazo the 'sc' option
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

# --- Simulation Parameters ---
G_value = 5.0
theta = 1.0
# Fixed tau_w sets
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.22559844, -0.11142818,  0.17570669,  0.36519415,  0.62727242])
xi_values = np.linspace(0.1, 1.0, 10)
# Different values of p_a to loop over
p_a_list = [1.0, 3.0, 6.0, 9.0]

# --- Function to optimize ONLY τ_z for FIXED τ_w ---
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr, p_a):
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr, p_a):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val, p_a)
            if not converged:
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            welfare = np.sum(utilities) - 5*xi_val * (agg_polluting**theta)
            return -welfare
        except Exception:
            return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [1.0]
    try:
        result = minimize(swf_obj_fixed_w,
                          initial_tau_z_guess,
                          args=(G, xi, fixed_tau_w_arr, p_a),
                          method='SLSQP',
                          bounds=tau_z_bounds,
                          options={'disp': False, 'ftol': 1e-15})
        if result.success:
            return result.x[0], -result.fun
        else:
            print("FAILURE in fixed-w τ_z optimization")
            return None, None
    except Exception:
        return None, None

# --- Create 2x4 subplot grid ---
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7.5))

for idx, p_a in enumerate(p_a_list):
    # Lists to store results for this p_a
    tau_z_optimal_w_results = []
    tau_z_fixed_preexisting_results = []
    tau_z_fixed_optimal_xi01_results = []
    valid_xi_optimal_w = []
    valid_xi_fixed_preexisting = []
    valid_xi_fixed_optimal_xi01 = []

    # Scenario 1: Optimal τ_w and τ_z
    for xi_val in xi_values:
        try:
            opt_tau_w, opt_tau_z, _ = outer_solver.maximize_welfare(G_value, xi_val, p_a)
            tau_z_optimal_w_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
            valid_xi_optimal_w.append(xi_val)
        except:
            tau_z_optimal_w_results.append(np.nan)
            valid_xi_optimal_w.append(xi_val)

    # Scenario 2: Fixed τ_w (Pre-existing)
    for xi_val in xi_values:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting, p_a)
        tau_z_fixed_preexisting_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_preexisting.append(xi_val)

    # Scenario 3: Fixed τ_w (Optimal at ξ=0.1)
    for xi_val in xi_values:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01, p_a)
        tau_z_fixed_optimal_xi01_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)

    # Convert to arrays
    tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
    valid_xi_optimal_w     = np.array(valid_xi_optimal_w)
    tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results)
    valid_xi_fixed_preexisting      = np.array(valid_xi_fixed_preexisting)
    tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)
    valid_xi_fixed_optimal_xi01      = np.array(valid_xi_fixed_optimal_xi01)

    ax = axes.flat[idx]
    # Plot each scenario
    mask1 = ~np.isnan(tau_z_optimal_w_results)
    mask2 = ~np.isnan(tau_z_fixed_preexisting_results)
    mask3 = ~np.isnan(tau_z_fixed_optimal_xi01_results)

    if np.any(mask1):
        ax.plot(valid_xi_optimal_w[mask1],
                tau_z_optimal_w_results[mask1],
                linestyle='-', label='Variable inc. tax')

    if np.any(mask2):
        ax.plot(valid_xi_fixed_preexisting[mask2],
                tau_z_fixed_preexisting_results[mask2],
                linestyle='--', label='Fixed inc. tax (pre-existing)')

    if np.any(mask3):
        ax.plot(valid_xi_fixed_optimal_xi01[mask3],
                tau_z_fixed_optimal_xi01_results[mask3],
                linestyle=':', label='Fixed inc. tax (optimal at $\\xi=0.1$)')

    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel(r'Environmental preference ($\xi$)', fontsize=14)
    ax.set_ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
    ax.set_title(f'$p_a={p_a}$', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

# Final layout and save
plt.tight_layout()
os.makedirs("d_extension", exist_ok=True)
plt.savefig("d_extension/b_robustness.pdf", bbox_inches='tight')
