from types import SimpleNamespace
from a_hh import workerProblem
from alt_b_firm_old import firmProblem
from scipy import optimize
import numpy as np

class equilibirium():
    
    def __init__(self):
        # a. define vector of equilivirum variables
        self.parEq = SimpleNamespace()
        self.solEq = SimpleNamespace()
        
        # Initialize with default values
        self.solEq.p = 1.0
        self.solEq.w = 1.0
        
    def evaluate_equilibrium(self, p, w):
        """Evaluate economic conditions at given price and wage."""
        # a. optimal behavior of household
        hh = workerProblem()
        # No transfers for simplicity
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=0.0)
        
        # b. firm uses all available labor
        labor_supply = 1 - hhSol.ell
        firm = firmProblem()
        firm.parFirm.w = w
        firm.parFirm.p = p
        firm.parFirm.tau_z = 0.0
        
        # c. find z that makes profit zero
        def profit_obj(z):
            firm.sol.t = labor_supply
            firm.sol.z = z
            firm.sol.y = firm.inside(firm.sol.t, firm.sol.z)**(1/firm.parFirm.r)
            return p*firm.sol.y - w*firm.sol.t
        
        result = optimize.minimize_scalar(profit_obj, bounds=(0.0, 10.0), method='bounded')
        firm.sol.z = result.x
        firm.sol.t = labor_supply
        firm.sol.y = firm.inside(firm.sol.t, firm.sol.z)**(1/firm.parFirm.r)
        
        # d. calculate market conditions
        goods_supply = firm.sol.y
        goods_demand = hhSol.c
        profit = p*goods_supply - w*labor_supply
        
        # e. save results
        self.parEq.g_m = goods_demand - goods_supply
        self.parEq.l_m = 0.0  # Labor market clears by construction
        self.parEq.profit = profit
        
        print(f'p={p:.4f}, w={w:.4f} → goods excess={self.parEq.g_m:.4f}, labor excess={self.parEq.l_m:.4f}, profit={profit:.4f}')
        
        return self.parEq

# Create a single unified solution function
def find_equilibrium():
    eq = equilibirium()
    
    # First, normalize wage to 1.0 (price numeraire)
    w_fixed = 1.0
    
    def obj_price(p):
        eq.evaluate_equilibrium(p, w_fixed)
        # Focus primarily on goods market clearing
        return eq.parEq.g_m**2
    
    # Try different initial prices in wider range
    best_p = None
    best_error = float('inf')
    
    for p_start in [0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            result = optimize.minimize_scalar(obj_price, 
                                             bracket=[0.1*p_start, p_start, 10*p_start], 
                                             method='brent',
                                             options={'xtol': 1e-8})
            if result.fun < best_error:
                best_error = result.fun
                best_p = result.x
        except Exception as e:
            print(f"Error with starting price {p_start}: {e}")
    
    if best_p is not None:
        print("\n=== FINAL SOLUTION ===")
        eq.solEq.p = best_p
        eq.solEq.w = w_fixed
        eq.evaluate_equilibrium(best_p, w_fixed)
        print(f"Found equilibrium: p = {best_p:.4f} (with w = {w_fixed})")
        return eq
    else:
        print("Failed to find equilibrium")
        return None

# Run the solution
equilibrium = find_equilibrium()