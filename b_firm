from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize

class firmProblem:
    # parameters & solution vector
    def __init__(self):
        self.parFirm = SimpleNamespace(r=0.5, epsilon=0.5)
        self.sol = SimpleNamespace()
    
    #production function is f(t,z) = [epsilon·t^r + (1-epsilon)·z^r]^(1/r)
    def production(self, t, z):
        par = self.parFirm
        return (par.epsilon * t**par.r + (1 - par.epsilon) * z**par.r)**(1/par.r)
    
    def solve_firm(self, w, tau_z, y):
        par = self.parFirm
        
        # error from ratio of focs is w/tau_z - (ε/(1-ε))*(t/z)^(r-1)
        def error1(t, z):
            return (w / tau_z) - (par.epsilon / (1 - par.epsilon)) * (t/z)**(par.r - 1)
        
        # error from production constraint is production - y = 0.
        def error2(t, z):
            return self.production(t, z) - y
        
        # Objective: sum of squared errors.
        def objective(x):
            t, z = x
            e1 = error1(t, z)
            e2 = error2(t, z)
            return e1**2 + e2**2
        
        # initial guess is t=z=y
        x0 = [y, y]
        bounds = [(1e-6, None), (1e-6, None)]
        
        # minimize
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'ftol': 1e-12, 'disp': True})
        
        # in case of problem not converging
        if not res.success:
            print("minimization did not converge", res.message)
            return None
        
        # cost-minimizing input combination
        t_opt, z_opt = res.x
        
        # solution vector
        self.sol.t = t_opt
        self.sol.z = z_opt
        self.sol.y = self.production(t_opt, z_opt)
        self.sol.obj_value = res.fun
        self.sol.error1 = error1(t_opt, z_opt)
        self.sol.error2 = error2(t_opt, z_opt)
        
        # print solution
        print(f"solution:")
        print(f"t = {t_opt:.8f}")
        print(f"z = {z_opt:.8f}")
        print(f"production = {self.sol.y:.8f} (target y = {y:.8f})")
        print(f"ratio error: {self.sol.error1:.8e}")
        print(f"production error: {self.sol.error2:.8e}")
        print(f"sum of squared errors: {self.sol.obj_value:.8e}")
        return self.sol

# test
model = firmProblem()
solution = model.solve_firm(w=8, tau_z=5, y=20)