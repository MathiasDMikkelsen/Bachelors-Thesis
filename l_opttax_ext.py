import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

import k_outer_solver_ext as outer_solver
import j_inner_solver_ext as solver
from j_inner_solver_ext import n, alpha, beta, gamma, d0, phi, t as T

G_value = 5.0
theta = 1.0
p_a = 5.0
varsigma = 2.0
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01 = np.array([-1.22559844, -0.11142818,  0.17570669,  0.36519415,  0.62727242])

xi_values = np.linspace(0.1, 1.0, 10)

def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr, p_a, varsigma):
    p_a = p_a
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr, p_a):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val, p_a, varsigma)
            if not converged:
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities
            welfare = np.sum(valid_utilities) - 5 * xi_val * (agg_polluting ** theta)
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
            print("failed")
            return None, None
    except Exception:
        return None, None

tau_z_optimal_w_results = []
tau_z_fixed_preexisting_results = []
tau_z_fixed_optimal_xi01_results = []
valid_xi_optimal_w = []
valid_xi_fixed_preexisting = []
valid_xi_fixed_optimal_xi01 = []

print("\nvariable inc. tax")
for xi_val in xi_values:
    try:
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val, p_a, varsigma)
        tau_z_optimal_w_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_optimal_w.append(xi_val)
    except Exception as e:
        print(f" error for xi = {xi_val:.4f}: {e}")
        tau_z_optimal_w_results.append(np.nan)
        valid_xi_optimal_w.append(xi_val)
print("variable inc. tax finished.")

print("\npre-existing inc. tax")
print(f"(inc. tax = {fixed_tau_w_preexisting})")
for xi_val in xi_values:
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting, p_a, varsigma)
        tau_z_fixed_preexisting_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
    except Exception as e:
        print(f" error for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_preexisting_results.append(np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
print("pre-existing inc. tax finished")

print("\nbaseline optimal inc. tax")
print(f"(inc. tax = {fixed_tau_w_optimal_xi01})")
for xi_val in xi_values:
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01, p_a, varsigma)
        tau_z_fixed_optimal_xi01_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)
    except Exception as e:
        print(f" error for xi = {xi_val:.4f}: {e}")
        tau_z_fixed_optimal_xi01_results.append(np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)
print("baseline optimal inc. tax finished")

print("completed")

tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w = np.array(valid_xi_optimal_w)
tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results)
valid_xi_fixed_preexisting = np.array(valid_xi_fixed_preexisting)
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)
valid_xi_fixed_optimal_xi01 = np.array(valid_xi_fixed_optimal_xi01)

plt.figure(figsize=(7, 5))

mask1 = ~np.isnan(tau_z_optimal_w_results)
mask2 = ~np.isnan(tau_z_fixed_preexisting_results)
mask3 = ~np.isnan(tau_z_fixed_optimal_xi01_results)

if np.any(mask1):
    plt.plot(valid_xi_optimal_w[mask1],
             tau_z_optimal_w_results[mask1], linewidth=2,
             linestyle='-', label='Variable inc. tax')

if np.any(mask2):
    plt.plot(valid_xi_fixed_preexisting[mask2],
             tau_z_fixed_preexisting_results[mask2], linewidth=2,
             linestyle='--', label='Fixed inc. tax (pre-existing)')

if np.any(mask3):
    plt.plot(valid_xi_fixed_optimal_xi01[mask3],
             tau_z_fixed_optimal_xi01_results[mask3], linewidth=2,
             linestyle=':', label='Fixed inc. tax (optimal at $\\xi=0.1$)')

plt.xlim(0.1, 1.0)
plt.xlabel(r'Environmental preference ($\xi$)', fontsize=14)
plt.ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
plt.legend(loc='best')
plt.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
plt.tight_layout()

plt.savefig("x_figs/fig8_opttax_ext.pdf", bbox_inches='tight')