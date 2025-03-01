from types import SimpleNamespace
from scipy.optimize import minimize, bisect
from a_hh import workerProblem
from b_firm import firmProblem

class equilibirium():
    
    def __init__(self):
        """ setup """
        par = self.par = SimpleNamespace()
        par.c_s = 1

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

    def find_equilibirum(self, lower=0.1, upper=5):
        eq_pc = bisect(self.market_clearing_excess, lower, upper)
        print(f'Equilibrium pc: {eq_pc:.2f}')
        return eq_pc

eq = equilibirium()
equilibrium_price = eq.find_equilibirum(lower=0.1, upper=5)