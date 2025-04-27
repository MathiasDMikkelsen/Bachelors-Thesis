# plot_slopes_and_penalty_slope.py
#
# Computes and plots:
#  1) –Slopes of the two aggregate welfare‐difference curves
#     (pre‐existing τ_w vs. optimal τ_w at ξ=0.1), flipped by –1
#  2) –Slope of the penalty term d/dτ_z[5·ξ·(z_c + z_d)] for ξ ∈ {0.1, 0.55, 1.0}
# against τ_z, all in one figure with the original styling.
# Only three lines (marked +++) have been added to plot the penalty curves for the optimal τ_w system.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline
import sys, os

# ─── Matplotlib LaTeX setup ───────────────────────────────────────────────────
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from a_solvers.inner_solver import solve, n

# ─── Parameters & grids ───────────────────────────────────────────────────────
g              = 5.0
tau_w_init     = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_opt      = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
tau_z_values   = np.linspace(0.1, 20.0, 100)
tau_z_smooth   = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
tau_z_baseline = 0.1

# ─── Baseline utilities at τ_z = 0.1 ───────────────────────────────────────────
_, res1, c1 = solve(tau_w_init, tau_z_baseline, g)
_, res2, c2 = solve(tau_w_opt,   tau_z_baseline, g)
if not (c1 and c2):
    raise RuntimeError("Baseline did not converge.")
u0_1 = res1["utilities"]
u0_2 = res2["utilities"]

# ─── Compute aggregate welfare differences ─────────────────────────────────────
agg_diff1 = np.zeros_like(tau_z_values)
agg_diff2 = np.zeros_like(tau_z_values)
for j, tz in enumerate(tau_z_values):
    _, r1, conv1 = solve(tau_w_init, tz, g)
    _, r2, conv2 = solve(tau_w_opt,  tz, g)
    u1 = r1["utilities"] if conv1 else u0_1
    u2 = r2["utilities"] if conv2 else u0_2
    agg_diff1[j] = np.sum(u1 - u0_1)
    agg_diff2[j] = np.sum(u2 - u0_2)

# ─── Smooth and differentiate welfare curves ──────────────────────────────────
spl1   = make_interp_spline(tau_z_values, agg_diff1)
spl2   = make_interp_spline(tau_z_values, agg_diff2)
slope1 = spl1.derivative()(tau_z_smooth)
slope2 = spl2.derivative()(tau_z_smooth)

# ─── Compute aggregate pollution for pre-existing τ_w ─────────────────────────
agg_poll = np.zeros_like(tau_z_values)
for j, tz in enumerate(tau_z_values):
    _, rp, convp = solve(tau_w_init, tz, g)
    if not convp:
        raise RuntimeError(f"Did not converge at τ_z = {tz:.3f}")
    agg_poll[j] = rp["z_c"] + rp["z_d"]

# ─── Smooth pollution curve and compute its derivative ───────────────────────
poll_spline  = make_interp_spline(tau_z_values, agg_poll)
poll_smooth  = poll_spline(tau_z_smooth)
poll_slope   = poll_spline.derivative()(tau_z_smooth)

# +++ Compute pollution & slope for optimal τ_w +++
agg_poll_opt = np.zeros_like(tau_z_values)
for j, tz in enumerate(tau_z_values):
    _, rp_opt, convp_opt = solve(tau_w_opt, tz, g)
    agg_poll_opt[j] = (rp_opt["z_c"] + rp_opt["z_d"]) if convp_opt else agg_poll_opt[j]
poll_spline_opt = make_interp_spline(tau_z_values, agg_poll_opt)
poll_smooth_opt = poll_spline_opt(tau_z_smooth)
poll_slope_opt  = poll_spline_opt.derivative()(tau_z_smooth)

# ─── Plot all curves together ─────────────────────────────────────────────────
plt.figure(figsize=(7, 5))

# (1) Flipped slopes of welfare-difference curves
plt.plot(
    tau_z_smooth, -slope1,
    color='tab:orange', linestyle='--', linewidth=2,
    label='Blue loss (pre-existing $\\tau_w$)'
)
plt.plot(
    tau_z_smooth, -slope2,
    color='tab:green', linestyle=':', linewidth=2,
    label='Blue loss (optimal $\\tau_w$ at $\\xi=0.1$)'
)

# (2) Flipped slopes of penalty term for three ξ (pre-existing in solid, optimal in dashed)
xis = [0.1, 0.55, 1.0]
maroon_shades = ['#f2dede', '#c9302c', '#7b1113']
for xi_val, col in zip(xis, maroon_shades):
    # pre-existing τ_w
    plt.plot(
        tau_z_smooth,
        -5 * xi_val * poll_slope,
        color=col, linestyle='-', linewidth=2,
        label=f'Green benefit, pre-existing inc. tax ($\\xi={xi_val}$)'
    )
    # optimal τ_w (no extra legend entry)
    plt.plot(
        tau_z_smooth,
        -5 * xi_val * poll_slope_opt,
        color=col, linestyle='--', label=f'Green benefit, baseline opt. inc tax ($\\xi={xi_val}$)', linewidth=2
    )

# ─── Styling (unchanged) ─────────────────────────────────────────────────────
plt.xlim(0.1, 20.0)
plt.ylim(-0.25, 1.00)
plt.xlabel(r'Environmental tax ($\tau_z$)', fontsize=14)
plt.ylabel(r'Marginal welfare change', fontsize=14)
plt.grid(True, color='grey', linestyle='-', linewidth=0.3, alpha=0.5)
plt.legend(loc='best', fontsize=8)
plt.tight_layout()

# ─── Save & show ─────────────────────────────────────────────────────────────
plt.savefig("c_opttax/c_tradeoff.pdf", bbox_inches='tight')
