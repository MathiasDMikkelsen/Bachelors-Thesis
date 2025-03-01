from types import SimpleNamespace
import numpy as np
from scipy import optimize
from scipy.optimize import minimize, bisect

#####################################################
# workers
#####################################################

class workerProblem():
    
    def __init__(self):
        """ setup """
        
        # a. define vetor of parameters
        par = self.par = SimpleNamespace()
        
        # b. exogenous parameters
        par.t = 1 # total time endowment
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
        par.pb = pb # price of polluting good
        par.l = l # lump sum transfer
        
        # c. add variable parameters to parameter vector
        par.phi = phi # productivity
        par.tau = tau # tax rate
        
        # d. define objective function as negative utility
        def obj(x):
            ell, b = x # define variables
            return -self.utility((par.w*(1-tau)*(par.t-ell)-b*par.pb)/par.pc,b,ell) # substitute c from budget constraint
        
        # e. constraints
        cons = [ 
            {'type': 'ineq', 'fun': lambda x: x[0]},  # ell >= 0
            {'type': 'ineq', 'fun': lambda x: x[1] - par.b0},  # b >= b0
            {'type': 'ineq', 'fun': lambda x: (par.w*(1-tau)*(par.t-x[0])-x[1]*par.pb)/par.pc},  # c >= 0
            {'type': 'ineq', 'fun': lambda x: par.t - x[0]}  # ell <= t (upper bound)
        ]

        # f. initial guess
        x0 = [par.t / 2, par.b0 + 0.1]  # start within feasible region

        # g. solve using a constrained optimizer
        res = minimize(obj, x0, method='SLSQP', constraints=cons)
        
        # h. save solution
        sol.ell = res.x[0]
        sol.b = res.x[1]
        sol.c = (par.w*(1-par.tau)*(par.t-sol.ell)-sol.b*par.pb)/par.pc # calculate c from budget constraint
        
        # i. print solution
        print(f'solution: ell={sol.ell:.2f}, b={sol.b:.2f}, c={sol.c:.2f}, budget: {sol.c*par.pc+sol.b*par.pb:2f}')
        
        # j. return solution
        return sol

#####################################################
# firms
#####################################################

class firmProblem():
    def __init__(self):
        """ setup"""
        
        # a. define vector of parameters
        parFirm = self.parFirm = SimpleNamespace()
        
        # b. exogenous parameters
        parFirm.r = 0.4 # elasticity of substitution
        parFirm.x = 0.4 # treshhold?
        
        # c. define solution set
        solFirm = self.solfFirm = SimpleNamespace()
        
        # d. solution parameters
        solFirm.t = 0 # clean input
        solFirm.z = 0 # polluting input
      
    def production(self, t, z):
        """ production of firm """
        
        # a. retrieve exogenous parameters
        parFirm = self.parFirm
        
        # b. return production function
        return (parFirm.epsilon*t**parFirm.r + (1-parFirm.epsilon)*z**parFirm.r)**(1/parFirm.r)*(z<=parFirm.x*t).astype(int) + 0*(z>parFirm.x*t).astype(int)
    
    def firm(self, epsilon, w, tau_z)
        """ maximize profit for firms """
        
        # a. retrieve solution set and parameter vector
        solFirm = self.solFirm
        parFirm = self.parFirm
        
        # b. add endogenous parameters to parameter vector
        parFirm.w = w # wage
        
        # c. add variable parameters to parameter vector
        parFirm.epsilon = epsilon # share of polluting input in production
        parFirm.tau_z = tau_z # pollution tax rate aka price for polluting input
        parFirm.price = price # price of output
        
        # d. define objective function as negative profit
        def obj(x):
            t, z = x # define variables
            return -price*self.production(self, t, z)+ w*t + tau_z*z
        
        # e. constraints
        cons = [ 
            {'type': 'ineq', 'fun': lambda x: x[0]},  # t >= 0
            {'type': 'ineq', 'fun': lambda x: x[0]},  # z >= 0
        ]
        
        # f. initial guess
        x0 = [1,1]  # start within feasible region 
        
        # g. solve using a constrained optimizer
        res = minimize(obj, x0, method='SLSQP', constraints=cons)
        

#####################################################
# equilbirium
#####################################################

class equilibirium():
    
    def __init__(self):
    
        # b. supply 
        par = self.par = SimpleNamespace()
        par.c_s = 1

    def market_clearing_excess(self, pc):
        
        # c. calculate demand
        model = workerProblem()
        model.worker(phi=0.5,tau=0.2, w=1, pb=1, pc=pc, l=0)
        demand=model.sol.c
        supply = self.par.c_s
        excess_demand = demand -supply
        print (f'Excess supply: {excess_demand:.2f}')
        return excess_demand

    def find_equilibirum(self, lower=0.1, upper=5):
        eq_pc = bisect(self.market_clearing_excess, lower, upper)
        print(f'Equilibrium pc: {eq_pc:.2f}')
        return eq_pc

eq = equilibirium()
equilibrium_price = eq.find_equilibirum(lower=0.1, upper=5)