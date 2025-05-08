import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":   ["Palatino"],
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})
    
import b_outer_solver               
import a_inner_solver as solver    
from a_inner_solver import n, alpha, beta, gamma, d0, phi, t as T

G_value = 5.0
theta = 1.0
r = -1.0   

fixed_tau_w_preexisting    = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
fixed_tau_w_optimal_xi01   = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
xi_values                  = np.linspace(0.1, 1.0, 10)

def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):

    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = float(tau_z_scalar[0]) if hasattr(tau_z_scalar, "__len__") else float(tau_z_scalar)
        solution, results, converged = solver.solve(fw_arr, tau_z, G_val, r, d0)
        if not converged:
            return 1e10
        utilities    = results['utilities']
        agg_pollute  = results['z_c'] + results['z_d']
        valid_utils  = utilities[utilities > -1e5]
        welfare      = np.sum(valid_utils) - 5 * xi_val * (agg_pollute ** theta)
        return -welfare

    try:
        res = minimize(
            swf_obj_fixed_w,
            x0=[0.5],
            args=(G, xi, fixed_tau_w_arr),
            bounds=[(1e-6, 100.0)],
            method='SLSQP',
            options={'ftol': 1e-7, 'disp': False}
        )
        if res.success:
            return res.x[0], -res.fun
    except Exception:
        pass
    return None, None


tau_z_optimal_w_results           = []
tau_z_fixed_preexisting_results   = []
tau_z_fixed_optimal_xi01_results  = []

for xi_val in xi_values:
    opt_tau_w, opt_tau_z, _ = b_outer_solver.maximize_welfare(G_value, xi_val, r, d0)
    tau_z_optimal_w_results.append(opt_tau_z if opt_tau_z is not None else np.nan)

for xi_val in xi_values:
    opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
    tau_z_fixed_preexisting_results.append(opt_tau_z if opt_tau_z is not None else np.nan)

for xi_val in xi_values:
    opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
    tau_z_fixed_optimal_xi01_results.append(opt_tau_z if opt_tau_z is not None else np.nan)

tau_z_optimal_w_results          = np.array(tau_z_optimal_w_results)
tau_z_fixed_preexisting_results  = np.array(tau_z_fixed_preexisting_results)
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)

plt.figure(figsize=(7, 5))

mask1 = ~np.isnan(tau_z_optimal_w_results)
mask2 = ~np.isnan(tau_z_fixed_preexisting_results)
mask3 = ~np.isnan(tau_z_fixed_optimal_xi01_results)

if mask1.any():
    plt.plot(
        xi_values[mask1],
        tau_z_optimal_w_results[mask1],
        '-', linewidth=2,
        color='steelblue',
        label='Variable inc. tax'
    )

if mask2.any():
    plt.plot(
        xi_values[mask2],
        tau_z_fixed_preexisting_results[mask2],
        '--', linewidth=2,
        color='tab:orange',
        label='Fixed inc. tax (pre-existing)'
    )
    
if mask3.any():
    plt.plot(
        xi_values[mask3],
        tau_z_fixed_optimal_xi01_results[mask3],
        ':', linewidth=2,
        color='tab:green',
        label='Fixed inc. tax (optimal at $\\xi=0.1$)'
    )

plt.xlim(0.1, 1.0)
plt.xlabel(r'Environmental preference ($\xi$)', fontsize=14)
plt.ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
plt.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
plt.legend(loc='best', fontsize='medium', frameon=True)

plt.tight_layout()
plt.savefig("x_figs/fig6_opttax.pdf", bbox_inches='tight')