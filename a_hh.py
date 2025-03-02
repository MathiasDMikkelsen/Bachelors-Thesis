from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
from scipy import optimize

class workerProblem():
    
    def __init__(self):
        """ setup """
        
        # a. define vetor of parameters
        par = self.par = SimpleNamespace()
        
        # b. exogenous parameters
        par.time = 1 # total time endowment
        par.alpha = 0.5 # clean good utility weight
        par.beta = 0.5 # polluting good utility weight
        par.gamma = 0.5 # leisure utility weight
        par.b0 = 0.1 # subsistence consumption of polluting good

        # c. define solution set
        sol = self.sol = SimpleNamespace()
        
        # d. solution parameters
        sol.ell = 0 # leisure
        sol.b = 0 # consumption of polluting good
        sol.c = 0 #consumption of clean good

    def utility(self,c,b,ell):
        """ utility of workers """
        
        # a. retrieve exogenous parameters
        par = self.par 
        
        # b. return utility function
        return par.alpha*np.log(c)+par.beta*np.log(b-par.b0)+par.gamma*np.log(ell) 
    
    def worker(self, phi, tau, w, pb, pc, l): 
        """ maximize utility for workers """
        
        # a. retrieve solution set and parameter vector
        sol = self.sol 
        par = self.par 
        
        # b. add endogenous parameters to parameter vector
        par.w = w # wage
        par.pb = pb # price of polluting good
        par.pc = pc # price of clean good
        par.l = l # lump sum transfer
        
        # c. add variable parameters to parameter vector
        par.phi = phi # productivity
        par.tau = tau # tax rate
        
        # d. define objective function as negative utility
        def obj(x):
            ell, b = x # define variables
            return -self.utility((par.w*par.phi*(1-tau)*(par.time-ell)-b*par.pb+par.l)/par.pc,b,ell) # substitute c from budget constraint
        
        # e. constraints
        cons = [ 
            {'type': 'ineq', 'fun': lambda x: x[0]},  # ell >= 0
            {'type': 'ineq', 'fun': lambda x: x[1] - par.b0},  # b >= b0
            {'type': 'ineq', 'fun': lambda x: (par.w*par.phi*(1-tau)*(par.time-x[0])-x[1]*par.pb+par.l)/par.pc},  # c >= 0
            {'type': 'ineq', 'fun': lambda x: par.time - x[0]}  # ell <= t (upper bound)
        ]

        # f. initial guess
        x0 = [par.time / 2, par.b0 + 0.1]  # start within feasible region

        # g. solve using a constrained optimizer
        res = minimize(obj, x0, method='SLSQP', constraints=cons)
        
        # h. save solution
        sol.ell = res.x[0]
        sol.b = res.x[1]
        sol.c = (par.w*par.phi*(1-par.tau)*(par.time-sol.ell)-sol.b*par.pb+par.l)/par.pc # calculate c from budget constraint
        
        # i. print solution
        print(f'solution: ell={sol.ell:.2f}, b={sol.b:.2f}, c={sol.c:.2f}, budget: {sol.c*par.pc+sol.b*par.pb:.2f}')
        
        # j. return solution
        return sol

# test
model = workerProblem()
model.worker(phi=1, tau=0, w=1, pb=1, pc=1, l=0)

