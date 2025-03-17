import numpy as np
import numba as nb

# 1. blocks
@nb.njit
def firm_c_production(t_c, z_c, epsilon_c, r, x):
    if z_c > x * t_c:
        return 0.0
    
    inside = epsilon_c * (t_c**r) + (1.0 - epsilon_c) * (z_c**r)
    if inside <= 0.0:
        return 0.0
    
    return inside**(1.0 / r)

@nb.njit
def firm_d_production(t_d, z_d, epsilon_d, r, x):
    if z_d > x * t_d:
        return 0.0
    
    inside = epsilon_d * (t_d**r) + (1.0 - epsilon_d) * (z_d**r)
    if inside <= 0.0:
        return 0.0
    
    return inside**(1.0 / r)

@nb.njit
def firm_c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r, x):
    y_c = firm_c_production(t_c, z_c, epsilon_c, r, x)
    if y_c <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_c = epsilon_c * (t_c**(r-1.0)) * (y_c**(1.0-r))
    dy_dz_c = (1.0 - epsilon_c) * (z_c**(r-1.0)) * (y_c**(1.0-r))
    eq_t_c  = p_c * dy_dt_c - w
    eq_z_c  = p_c * dy_dz_c - tau_z
    return np.array([eq_t_c, eq_z_c])

@nb.njit
def firm_d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r, x):
    y_d = firm_d_production(t_d, z_d, epsilon_d, r, x)
    if y_d <= 0.0:
        return np.array([1e6, 1e6])
    dy_dt_d = epsilon_d * (t_d**(r-1.0)) * (y_d**(1.0-r))
    dy_dz_d = (1.0 - epsilon_d) * (z_d**(r-1.0)) * (y_d**(1.0-r))
    eq_t_d  = p_d * dy_dt_d - w
    eq_z_d  = p_d * dy_dz_d - tau_z
    return np.array([eq_t_d, eq_z_d])
# end blocks
