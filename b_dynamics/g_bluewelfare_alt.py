# plot_slopes_tauz.py (Only the slopes of the two welfare‐difference curves)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline
import sys, os

# Matplotlib LaTeX setup
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

# Ensure we can import the solver
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a_solvers.inner_solver import solve, n

# --- Parameters and grids ---
g = 5.0
tau_w_init  = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_opt   = np.array([-1.12963781, -0.06584074, 0.2043803, 0.38336986, 0.63241591])
tau_z_values   = np.linspace(0.1, 20.0, 100)
tau_z_baseline = 0.1

# --- Baseline utilities ---
_, res1, c1 = solve(tau_w_init,  tau_z_baseline, g)
_, res2, c2 = solve(tau_w_opt,   tau_z_baseline, g)
if not (c1 and c2):
    raise RuntimeError("Baseline did not converge.")
u0_1 = res1["utilities"]
u0_2 = res2["utilities"]

# --- Compute aggregate welfare differences over tau_z_values ---
agg_diff1 = np.zeros(len(tau_z_values))
agg_diff2 = np.zeros(len(tau_z_values))

for j, tz in enumerate(tau_z_values):
    _, r1, c1 = solve(tau_w_init, tz, g)
    _, r2, c2 = solve(tau_w_opt,  tz, g)
    u1 = r1["utilities"] if c1 else u0_1
    u2 = r2["utilities"] if c2 else u0_2
    agg_diff1[j] = np.sum(u1 - u0_1)
    agg_diff2[j] = np.sum(u2 - u0_2)

# --- Smooth and differentiate ---
tau_z_smooth = np.linspace(tau_z_values.min(), tau_z_values.max(), 300)
spl1 = make_interp_spline(tau_z_values, agg_diff1)
spl2 = make_interp_spline(tau_z_values, agg_diff2)

slope1 = spl1.derivative()(tau_z_smooth)
slope2 = spl2.derivative()(tau_z_smooth)

# --- Plot the slopes only ---
plt.figure(figsize=(7, 5))
plt.plot(
    tau_z_smooth, slope1,
    color='tab:orange', linestyle='--', linewidth=2,
    label=r'Slope (Pre-existing $\tau_w$)'
)
plt.plot(
    tau_z_smooth, slope2,
    color='tab:green', linestyle=':', linewidth=2,
    label=r'Slope (Optimal $\tau_w$ at $\xi=0.1$)'
)

plt.xlim(0.1, 20.0)
plt.xlabel(r'$\tau_z$', fontsize=14)
plt.ylabel(r'$\displaystyle \frac{d}{d\tau_z}\!\Bigl[\Delta\sum \log \tilde u_i\Bigr]$', fontsize=14)
plt.grid(True, color='grey', linestyle='-', linewidth=0.3, alpha=0.5)
plt.legend(loc='best', fontsize=10, frameon=False)
plt.tight_layout()

# Save & show
os.makedirs("c_opttax", exist_ok=True)
outpath = "c_opttax/slopes_tauz.pdf"
plt.savefig(outpath, bbox_inches='tight')
print(f"Slopes plot saved to {outpath}")
plt.show()