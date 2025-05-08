from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ProductionEconomyClass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. parameters
        par.kappa = 0.1 # home production
        par.omega = 10 # disutility of labor supply factor
        par.eta = 1.50 # curvature of disutility of labor supply
        par.alpha = 1 # curvature of production function
        par.Nw = 99 # number of workers
        par.phi1 = 0.7 # productivity of rich workers
        par.phi2 = 0.3 # productivity of poor workers

        # b. grids
        par.num_w = 10
        par.grid_w = np.linspace(0.1,1.5,par.num_w)
        par.grid_mkt_clearing = np.zeros(par.num_w)

        # c. solution
        sol = self.sol = SimpleNamespace()
        
        sol.p = 1 # output price
        sol.w = 1 # wage

    def utility_w1(self,c1_w_star,l1):
        """ utility of workers """
        
        par = self.par

        return np.log(c1_w_star+par.kappa)-par.omega*l1**par.eta
    
    def utility_w2(self,c2_w_star,l2):
        """ utility of workers """
        
        par = self.par

        return np.log(c2_w_star+par.kappa)-par.omega*l2**par.eta

    def workers1(self): # rich
        """ maximize utility for workers """
        
        sol = self.sol 
        par = self.par # retrieves a set of parameters and names them par
        
        p = sol.p
        w = sol.w

        # a. solve
        obj = lambda l1: -self.utility_w1((w*par.phi1*l1)/p,l1) # substitute in the budget constraint
        res = optimize.minimize_scalar(obj,bounds=(0,1),method='bounded')
        
        # b. save
        sol.l_w1_star = res.x
        sol.c1_w_star = (w*sol.l_w1_star)/p
        
    def workers2(self): # poor
        """ maximize utility for workers """
        
        sol = self.sol # retrieves a set of parameters and names them sol
        par = self.par 
        
        p = sol.p
        w = sol.w

        # a. solve
        obj = lambda l2: -self.utility_w2((w*par.phi2*l2)/p,l2) # substitute in the budget constraint
        res = optimize.minimize_scalar(obj,bounds=(0,1),method='bounded')
        
        # b. save
        sol.l_w2_star = res.x
        sol.c2_w_star = (w*sol.l_w2_star)/p
        
    def firm(self):
        """ maximize firm profits """
        
        par = self.par
        sol = self.sol

        p = sol.p
        w = sol.w

        # a. solve
        f = lambda l_agg: l_agg**par.alpha
        obj = lambda l_agg: -(p*f(l_agg)-w*l_agg)
        x0 = [0.0]
        res = optimize.minimize(obj,x0,bounds=((0,None),),method='L-BFGS-B')
        
        # b. save
        sol.l_agg_star = res.x[0]
        sol.y_star = f(sol.l_agg_star)
        sol.Pi = p*sol.y_star-w*sol.l_agg_star
        
    def evaluate_equilibrium(self):
        """ evaluate equilirium """

        par = self.par
        sol = self.sol

        # a. optimal behavior of firm
        self.firm()
        sol.pi = sol.Pi/par.Nw

        # b. optimal behavior of households
        self.workers1()
        self.workers2()

        # c. market clearing
        sol.goods_mkt_clearing = par.Nw*sol.c1_w_star + par.Nw*sol.c2_w_star  - sol.y_star
        sol.labor_mkt_clearing = par.Nw*sol.l_w1_star + par.Nw*sol.l_w2_star - sol.l_agg_star
    
    def find_equilibrium(self):

        par = self.par
        sol = self.sol

        # a. grid search
        print('grid search:')
        for i,w in enumerate(par.grid_w):
            sol.w = w
            self.evaluate_equilibrium()
            par.grid_mkt_clearing[i] = sol.goods_mkt_clearing
            print(f' w = {w:.2f} -> {par.grid_mkt_clearing[i]:12.8f}')
        
        print('')

        # b. find bounds
        left = np.max(par.grid_w[par.grid_mkt_clearing < 0])
        right = np.min(par.grid_w[par.grid_mkt_clearing > 0])
        print(f'equilibrium price must be in [{left:.2f},{right:.2f}]\n')            

        # c. bisection search
        def obj(w):
            sol.w = w
            self.evaluate_equilibrium()
            return sol.goods_mkt_clearing

        res = optimize.root_scalar(obj,bracket=[left,right],method='bisect')
        sol.w = res.root
        print(f'the equilibrium wage is {sol.w:.4f}\n')

        # d. show result
        u_w1 = self.utility_w1(sol.c1_w_star,sol.l_w1_star)
        u_w2 = self.utility_w2(sol.c2_w_star,sol.l_w2_star)
        print(f'workers      : l1 = {sol.l_w1_star:6.4f}, l2 = {sol.l_w2_star:6.4f}, u1 = {u_w1:7.4f},  u2 = {u_w2:7.4f}')


model = ProductionEconomyClass()
print(model.par.kappa)
model.find_equilibrium()
