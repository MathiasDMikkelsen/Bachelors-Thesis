from types import SimpleNamespace
import numpy as np
from scipy import optimize
from scipy.optimize import minimize

class workerProblem():
    
    def __init__(self):
        
        # a. parameters
        par = self.par = SimpleNamespace()
        
        par.phi = 1 # productivity
        par.w = 0.2 # wage
        par.t = 1 # total time endowment
        par.tau = 0.25 # tax rate
        par.pb = 0.5 # price of polluting good
        par.pc = 0.5 # price of clean good
        par.l = 0.1 # lump sum transfer
        par.alpha = 0.5 
        par.beta = 0.5
        par.gamma = 0.2
        par.b0 = 0.1

        # b. solution space
        sol = self.sol = SimpleNamespace()
        
        sol.ell = 1 # leisure
        sol.b = 1 # consumption of polluting good
        sol.c = 1 #consumption of clean good

    def utility(self,c,b,ell):
        """ utility of workers """
        
        par = self.par

        return par.alpha*np.log(c)+par.beta*np.log(b-par.b0)+par.gamma*np.log(ell)
    
    def worker(self): 
        """ maximize utility for workers """
        
        sol = self.sol # retrieves the solution set
        par = self.par # retrieves a set of parameters and names them par
        
        # objective function
        def obj(x):
            ell, b = x
            c= (par.w*(par.t-ell)-b*par.pb)/par.pc
            return -self.utility(c,b,ell)
        
        # a. solve
        x0 = [0.1, 0.5] # initial guess
       
        res = minimize(obj, x0, method='BFGS') # minimize the objective function
        
        # b.save
        sol.ell = res.x[0]
        sol.b = res.x[1]
        sol.c = (par.w*(par.t-sol.ell)-sol.b*par.pb)/par.pc
        
        print(f'solution: ell={sol.ell:.2f}, b={sol.b:.2f}, c={sol.c:.2f}')
        
model = workerProblem()
model.worker()