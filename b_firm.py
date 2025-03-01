from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
from scipy import optimize
from scipy.optimize import differential_evolution

class firmProblem():
    def __init__(self):
        """ setup"""
        
        # a. define vector of parameters
        parFirm = self.parFirm = SimpleNamespace()
        
        # b. exogenous parameters
        parFirm.r = 1 # elasticity of substitution
        parFirm.x = 1 # treshhold? not used for now
        
        # c. define solution set
        solFirm = self.solFirm = SimpleNamespace()
        
        # d. solution parameters
        solFirm.t = 0 # clean input
        solFirm.z = 0 # polluting input
      
    def production(self,t,z):
        """ production of firm """
        
        # a. retrieve exogenous parameters
        parFirm = self.parFirm
        
        # b. return production function
        return (parFirm.epsilon*t**parFirm.r + (1-parFirm.epsilon)*z**parFirm.r)**(1/parFirm.r)
    
    def firm(self, epsilon, w, tau_z, p):
        """ maximize profit for firms"""
        parFirm = self.parFirm
        parFirm.epsilon = epsilon    # share of polluting input in production
        parFirm.tau_z = tau_z        # pollution tax rate
        parFirm.p = p                # price of output
        r = parFirm.r

        # a. system of focs
        def foc_error(vars):
            t, z = vars
            # common factor
            # A = epsilon*t**r + (1 - epsilon)*z**r
            # no division by zero
            # if A <= 0 or t <= 0 or z <= 0:
            #    return 1e6
            # common = A**(1/r - 1)
            mp_t = epsilon*t**(r-1)*self.production(t,z)**(1-r) # foc 1
            mp_z = (1 - epsilon)*z**(r-1)*self.production(t,z)**(1-r) # foc 2
            if t <= 0 or z <= 0 or self.production(t,z) <= 0:
                return 1e6
            err1= p*mp_t - w
            err2 = p*mp_z - tau_z
            return (err1**2 + err2**2) # return foc errors

        # b. choose initial guess and update
        #constraints = (
        #    {'type':'ineq','fun': lambda v: v[0]},  # t >= 0
        #    {'type':'ineq','fun': lambda v: v[1]},  # z >= 0
        #)

        # 3. Reasonable initial guess (strictly inside feasible region)
        x0 = [0.1, 0.05]  # ensures z < x*t initially if x=1

        # 4. Bounds for each variable
        bnds = ((0,None),(0,None))

        # 5. Solve using SLSQP
        res = optimize.minimize(foc_error, x0, method='SLSQP', bounds=bnds, constraints=None)
        
        if not res.success:
            print("WARNING: solver did not converge!")
            
        t_opt, z_opt = res.x
        y_opt = self.production(t_opt, z_opt)
        
        foc_err_value = foc_error([t_opt, z_opt])
        print(f'FOC error at solution: {foc_err_value:.6f}')

        # c. compute output and profit
        y_opt = self.production(t_opt, z_opt)
        profit = p * y_opt - w * t_opt - tau_z * z_opt

        # d. save solution
        self.solFirm.t = t_opt
        self.solFirm.z = z_opt
        self.solFirm.y = y_opt
        self.solFirm.pi = profit
        
        # e. print solution
        print(f"Converged = {res.success}, message = {res.message}")
        print(f'solution: t={self.solFirm.t:.2f}, z={self.solFirm.z:.2f}, '
          f'y={self.solFirm.y:.2f}, profit={self.solFirm.pi:.2f}')
        return self.solFirm
        
# test
model = firmProblem()
model.firm(epsilon=0.5, w=10, tau_z=10, p=12) 
