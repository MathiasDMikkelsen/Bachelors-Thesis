import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
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

from a_inner_solver import solve

alpha = 0.7; beta = 0.2; gamma = 0.2; t = 24.0
d0     = 0.5   
r      = -1.0  
phi    = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n      = len(phi)

tau_w     = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w_alt = np.array([-1.12963771, -0.06584069, 0.20438033, 0.38336989, 0.63241592])
g         = 5.0

def E_star(p_d, U_target):
    A = alpha + beta + gamma
    return (A/alpha) * np.exp((U_target
                               - beta*np.log(beta/(alpha*p_d))
                               - gamma*np.log(gamma/alpha)) / A) \
           + p_d*d0

tau_z_baseline = 1.0
sol_b1, res_b1, c1 = solve(tau_w,     tau_z_baseline, g, r, d0)
sol_b2, res_b2, c2 = solve(tau_w_alt, tau_z_baseline, g, r, d0)
if not (c1 and c2):
    raise RuntimeError("Baseline did not converge.")

w_b1, p_d_b1 = res_b1["w"], res_b1["p_d"]
l_agents_b1  = res_b1["l_agents"]
c_b1, d_b1   = res_b1["c_agents"], res_b1["d_agents"]

w_b2, p_d_b2 = res_b2["w"], res_b2["p_d"]
l_agents_b2  = res_b2["l_agents"]
c_b2, d_b2   = res_b2["c_agents"], res_b2["d_agents"]

U_b1 = alpha*np.log(c_b1) + beta*np.log(d_b1 - d0) + gamma*np.log(l_agents_b1)
U_b2 = alpha*np.log(c_b2) + beta*np.log(d_b2 - d0) + gamma*np.log(l_agents_b2)

base_exp1 = np.array([E_star(p_d_b1, U_b1[i]) for i in range(n)])
base_inc1 = np.array([
    phi[i]*w_b1*(1-tau_w[i])*(t-l_agents_b1[i]) + res_b1["l"]
    for i in range(n)
])
base_exp2 = np.array([E_star(p_d_b2, U_b2[i]) for i in range(n)])
base_inc2 = np.array([
    phi[i]*w_b2*(1-tau_w_alt[i])*(t-l_agents_b2[i]) + res_b2["l"]
    for i in range(n)
])

tau_z_values   = np.linspace(1.0, 20.0, 50)
m               = len(tau_z_values)
CV_rel1         = np.zeros((n, m))
CV_rel2         = np.zeros((n, m))
inc_ch1         = np.zeros((n, m))
inc_ch2         = np.zeros((n, m))
util_ch_exp1    = np.zeros((n, m))
util_ch_exp2    = np.zeros((n, m))
util_ch_log1    = np.zeros((n, m))
util_ch_log2    = np.zeros((n, m))

for j, tz in enumerate(tau_z_values):
    _, r1, c1 = solve(tau_w, tz, g, r, d0)
    if not c1: continue
    U1 = r1["utilities"]
    for i in range(n):
        CV_rel1[i,j]      = (E_star(r1["p_d"], U_b1[i]) - base_exp1[i]) / base_inc1[i]
        inc_ch1[i,j]      = phi[i]*r1["w"]*(1-tau_w[i])*(t-r1["l_agents"][i]) + r1["l"] - base_inc1[i]
        util_ch_exp1[i,j] = np.exp(U1[i]) - np.exp(U_b1[i])
        util_ch_log1[i,j] = U1[i] - U_b1[i]

    _, r2, c2 = solve(tau_w_alt, tz, g, r, d0)
    if not c2: continue
    U2 = r2["utilities"]
    for i in range(n):
        CV_rel2[i,j]      = (E_star(r2["p_d"], U_b2[i]) - base_exp2[i]) / base_inc2[i]
        inc_ch2[i,j]      = phi[i]*r2["w"]*(1-tau_w_alt[i])*(t-r2["l_agents"][i]) + r2["l"] - base_inc2[i]
        util_ch_exp2[i,j] = np.exp(U2[i]) - np.exp(U_b2[i])
        util_ch_log2[i,j] = U2[i] - U_b2[i]

blue_cmap = plt.cm.Blues
colors   = [blue_cmap(0.3 + 0.5*i/(n-1)) for i in range(n)]
lw       = 2.5

fig  = plt.figure(figsize=(10, 10))
gs   = gridspec.GridSpec(3, 2, height_ratios=[1,1,0.2],
                         hspace=0.40, wspace=0.05)

ax0  = fig.add_subplot(gs[0,0])  
ax1  = fig.add_subplot(gs[0,1])  
ax2  = fig.add_subplot(gs[1,0]) 
ax3  = fig.add_subplot(gs[1,1])  
ax4  = fig.add_subplot(gs[2,:])  

for i in range(n):
    ax0.plot(tau_z_values, CV_rel1[i], '-',  lw=lw, color=colors[i])
    ax0.plot(tau_z_values, CV_rel2[i], '--', lw=lw, color=colors[i])
ax0.set(xlabel=r'Environmental tax ($\tau_z$)',
        title='CV relative to disposable income')
ax0.tick_params(labelsize=12)
ax0.set_xlim(tau_z_values[0], tau_z_values[-1])
ax0.yaxis.set_label_coords(-0.15,0.5)
ax0.set_box_aspect(1)
ax0.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

for i in range(n):
    ax1.plot(tau_z_values, inc_ch1[i], '-',  lw=lw, color=colors[i])
    ax1.plot(tau_z_values, inc_ch2[i], '--', lw=lw, color=colors[i])
ax1.axhline(0, color='grey', linewidth=2.5)
ax1.set(xlabel=r'Environmental tax ($\tau_z$)',
        title='Total disposable income change')
ax1.tick_params(labelsize=12)
ax1.set_xlim(tau_z_values[0], tau_z_values[-1])
ax1.yaxis.set_label_coords(-0.15,0.5)
ax1.set_box_aspect(1)
ax1.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

for i in range(n):
    ax2.plot(tau_z_values, util_ch_exp1[i], '-',  lw=lw, color=colors[i])
    ax2.plot(tau_z_values, util_ch_exp2[i], '--', lw=lw, color=colors[i])
ax2.axhline(0, color='grey', linewidth=2.5)
ax2.set(xlabel=r'Environmental tax ($\tau_z$)',
        title='Change in exp utility')
ax2.tick_params(labelsize=12)
ax2.set_xlim(tau_z_values[0], tau_z_values[-1])
ax2.yaxis.set_label_coords(-0.15,0.5)
ax2.set_box_aspect(1)
ax2.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

for i in range(n):
    ax3.plot(tau_z_values, util_ch_log1[i], '-',  lw=lw, color=colors[i])
    ax3.plot(tau_z_values, util_ch_log2[i], '--', lw=lw, color=colors[i])
ax3.axhline(0, color='grey', linewidth=2.5)
ax3.set(xlabel=r'Environmental tax ($\tau_z$)',
        title='Change in log utility')
ax3.tick_params(labelsize=12)
ax3.set_xlim(tau_z_values[0], tau_z_values[-1])
ax3.yaxis.set_label_coords(-0.15,0.5)
ax3.set_box_aspect(1)
ax3.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

ax4.axis('off')
handles = [Line2D([0],[0],color=colors[i],lw=lw,label=rf'Household {i+1}')
           for i in range(n)]
ax4.legend(handles=handles, loc='center', ncol=n,
           frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("x_figs/fig2_dist_effects.pdf", bbox_inches='tight', pad_inches=0.05)