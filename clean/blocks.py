import numba as nb
import numpy as np

# 1. hh
# 1a. hh focs
@nb.njit
def hh_focs(c, d, ell, p_c, p_d, w, alpha, beta, gamma, d0, mult, psi):
    n = c.shape[0] # number of hh                                                     
    res = np.empty(3 * n) # residual array
    for i in range(n): # loop over hh
        if (c[i] <= 0.0) or (d[i] <= d0) or (ell[i] <= 0.0) or (ell[i] >= 1.0):
            res[3*i : 3*i+3] = np.array([1e6, 1e6, 1e6]) # penalty if out of bounds
        else:
            foc_c   = alpha * (1/c[i]) - mult[i] * p_c # clean good foc
            foc_d   = beta  * (1/(d[i] - d0)) - mult[i] * p_d # dirty good foc
            foc_ell = gamma * (1/ell[i]) - mult[i] * w * psi[i] # leisure foc
            res[3*i : 3*i+3] = np.array([foc_c, foc_d, foc_ell]) # create array w residuals
    return res # return array w residuals

# 1b. hh budget
@nb.njit
def hh_budget(c, d, ell, p_c, p_d, w, psi):
    n = c.shape[0] # number of hh
    res = np.empty(n) # resdiual array
    for i in range(n): # loop over hh
        res[i] = p_c * c[i] + p_d * d[i] - w * (1.0 - ell[i]) * psi[i] # budget constraint for each hh
    return res # return array w residuals
# end hh

# 2. firms
# 2a. firm c ces
@nb.njit
def c_prod(t_c, z_c, epsilon_c, r):                             
    inside = epsilon_c * (t_c**r) + (1.0 - epsilon_c) * (z_c**r) # ces production function
    if inside <= 0.0: 
        return 0.0 # return zero if neg inside
    return inside**(1.0 / r) # return output

# 2b. firm d ces
@nb.njit
def d_prod(t_d, z_d, epsilon_d, r):   
    inside = epsilon_d * (t_d**r) + (1.0 - epsilon_d) * (z_d**r) # ces production function
    if inside <= 0.0:
        return 0.0 # return zero if neg inside
    return inside**(1.0 / r) # return output

# 2c. firm c focs
@nb.njit
def c_focs(t_c, z_c, p_c, w, tau_z, epsilon_c, r):        
    y_c = c_prod(t_c, z_c, epsilon_c, r) # output of c
    if y_c <= 0.0:
        return np.array([1e6, 1e6]) # return >0 if neg
    dy_dt_c = epsilon_c * (t_c**(r - 1.0)) * (y_c**(1.0 - r)) # derivative wrt. labor
    dy_dz_c = (1.0 - epsilon_c) * (z_c**(r - 1.0)) * (y_c**(1.0 - r)) # derivative wrt. pollution
    foc_t_c = p_c * dy_dt_c - w # labor foc
    foc_z_c = p_c * dy_dz_c - tau_z # pollution foc
    return np.array([foc_t_c, foc_z_c]) # return focs

# 2d. firm d focs
@nb.njit
def d_focs(t_d, z_d, p_d, w, tau_z, epsilon_d, r):      
    y_d = d_prod(t_d, z_d, epsilon_d, r) # output of d
    if y_d <= 0.0:
        return np.array([1e6, 1e6]) # return >0 if neg
    dy_dt_d = epsilon_d * (t_d**(r - 1.0)) * (y_d**(1.0 - r)) # derivative wrt. labor
    dy_dz_d = (1.0 - epsilon_d) * (z_d**(r - 1.0)) * (y_d**(1.0 - r)) # derivative wrt. pollution
    foc_t_d = p_d * dy_dt_d - w # labor foc
    foc_z_d = p_d * dy_dz_d - tau_z # pollution foc
    return np.array([foc_t_d, foc_z_d]) # return focs
# end firms

# 3. govt
# 3a. swf
@nb.njit
def swf(u,z,xi,theta,omega):
    exp = 1.0 - omega
    welfare = (1.0/exp)*np.sum(u**exp)
    if omega == 1.0:
        welfare = sum(np.log(u))
    return welfare
# end govt

# 4. equilibirum