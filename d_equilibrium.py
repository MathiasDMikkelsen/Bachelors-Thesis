from types import SimpleNamespace
from scipy.optimize import minimize, bisect
from a_hh import workerProblem
from b_firm import firmProblem

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