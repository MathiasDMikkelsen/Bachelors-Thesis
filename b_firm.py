import numpy as np
from types import SimpleNamespace
from scipy import optimize

class firmProblem:
    
    # initialize parameters and solution vector
    def __init__(self):
        # define parameter vector
        self.parFirm = SimpleNamespace()
        
        # define exogenous parameters
        self.parFirm.r = 0.5         # elasticity exponent
        self.parFirm.epsilon = 0.5   # ces weight on t
        
        # define solution vector
        self.sol = SimpleNamespace()

    # define inside of production function (return 0 if inputs are negative)
    def inside(self, t, z):
        parFirm = self.parFirm
        inside = parFirm.epsilon * (t ** parFirm.r) + (1 - parFirm.epsilon) * (z ** parFirm.r)
        return inside if inside > 0 else 0.0

    # calculate foc errors
    def foc_errors(self, t, z):
        parFirm = self.parFirm
        f = self.inside(t, z)
        
        # avoid division by zero or negative production
        if f <= 0:
            return np.array([np.nan, np.nan])
        
        # define focs
        df_dt = parFirm.epsilon * (t ** (parFirm.r - 1)) * (f ** (1 - parFirm.r))
        df_dz = (1 - parFirm.epsilon) * (z ** (parFirm.r - 1)) * (f ** (1 - parFirm.r))
        
        # define errors
        err_t = parFirm.p * df_dt - parFirm.w
        err_z = parFirm.p * df_dz - parFirm.tau_z
        
        # return erros
        return np.array([err_t, err_z])

    def solve_firm(self, w, tau_z, p):
        # define variable parameters
        self.parFirm.w = w             # cost of t
        self.parFirm.tau_z = tau_z     # cost of z
        self.parFirm.p = p             # price of output
        
        # define grid for t (max 5 full units for five households)
        t_grid = np.linspace(1e-6, 5.0, 1000) # aviod zero division (t always larger than zero)
        
        # initialize best guess
        best_total_error = np.inf
        best_t = None
        best_z = None

        # loop over all values in the grid
        for t_val in t_grid:
            
            # define objective as squared error on z's foc
            def obj_z(z):
                foc = self.foc_errors(t_val, z[0])
                return foc[1]**2 

            # starting guess
            z0 = [0.5]
            
            # minimize squared error
            res_z = optimize.minimize(obj_z, z0, bounds=[(1e-6, None)], method='L-BFGS-B',
                          options={'ftol':1e-12, 'gtol':1e-12})
            
            # continue if convergence fails
            if not res_z.success:
                continue
            
            # update z value to optization problem output
            z_val = res_z.x[0]

            # compute sum of squared focs
            foc = self.foc_errors(t_val, z_val)
            total_error = foc[0]**2 + foc[1]**2

            # update best guess if total error is smaller then the current best guess
            if total_error < best_total_error:
                best_total_error = total_error
                best_t = t_val
                best_z = z_val

            #  break early if total squared error is small
            if best_total_error < 1e-10:
                break
        
        # print solution
        print(f"solution: t = {best_t:.8f}, z = {best_z:.8f}, total squared foc error = {best_total_error:.8e}")
        
        # return solution
        self.sol.t = best_t
        self.sol.z = best_z
        self.sol.y = self.inside(best_t, best_z)**(1/(self.parFirm.r))
        return self.sol

# test
model = firmProblem()
model.solve_firm(w=5, tau_z=10, p=10)
errors = model.foc_errors(model.sol.t, model.sol.z)
print(f"foc errors: df/dt error = {errors[0]:.8e}, df/dz error = {errors[1]:.8e}")