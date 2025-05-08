from pyomo.environ import *

# Create the model
model = ConcreteModel()

# ---------------------
# Define Sets
# ---------------------
model.I = RangeSet(1, 5)               # Demand index i = 1,...,5
model.J = Set(initialize=['C','D'])    # Production sectors: C and D

# ---------------------
# Define Parameters (default values, mutable for adjustments)
# ---------------------
model.alpha = Param(initialize=0.7, mutable=True)
model.beta  = Param(initialize=0.2, mutable=True)
model.gamma = Param(initialize=0.2, mutable=True)
model.r     = Param(initialize=-1.0, mutable=True)  # elasticity parameter
model.T     = Param(initialize=5.0, mutable=True)     # Time endowment per consumer
model.L     = Param(initialize=0.0, mutable=True)     # Lump-sum transfers (set to 0)
model.D0    = Param(initialize=0.01, mutable=True)
model.x     = Param(initialize=0, mutable=True)   # For indicator: Z[j] <= x * T_prod[j]

model.tau_z = Param(initialize=0.1, mutable=True)

def phi_init(model, i):
    return 1.0/5.0  
model.phi = Param(model.I, initialize=phi_init, mutable=True)

# Production weights: epsilon_j for j in {'C','D'}
def epsilon_init(model, j):
    return 0.5  
model.epsilon = Param(model.J, initialize=epsilon_init, mutable=True)

# ---------------------
# Define Variables
# ---------------------
# Demand-side variables (for each i in I)
model.C = Var(model.I, domain=NonNegativeReals)  # Consumption of good C
model.D = Var(model.I, domain=NonNegativeReals)  # Demand for good D
model.l = Var(model.I, domain=NonNegativeReals)  # Leisure

# Production-side variables (for each sector j in {'C','D'})
model.T_prod = Var(model.J, domain=NonNegativeReals, bounds=(1e-6, None))  # Production time input
model.Z      = Var(model.J, domain=NonNegativeReals)                         # Complementary input
model.F      = Var(model.J, domain=NonNegativeReals, bounds=(1e-6, None))      # Production output
model.p      = Var(model.J, domain=NonNegativeReals, bounds=(1e-6, None))      # Sector price

# Common wage variable (wage must be positive)
model.w = Var(domain=NonNegativeReals, bounds=(1e-6, None))

# ---------------------
# Fix p_C as numeraire: p_C = 1
# ---------------------
model.p['C'].fix(1.0)

# ---------------------
# Define Constraints
# ---------------------

def demand_C_rule(model, i):
    return model.C[i] == (model.alpha/(model.p['C']*(model.alpha + model.beta + model.gamma))) * \
           (model.phi[i]*model.w*model.T + model.tau_z*(model.Z['C'] + model.Z['D']) - model.p['D']*model.D0)
model.demand_C = Constraint(model.I, rule=demand_C_rule)

def demand_D_rule(model, i):
    # Demand for D remains unchanged
    return model.D[i] == (model.beta/(model.p['D']*(model.alpha + model.beta + model.gamma))) * \
           (model.phi[i]*model.w*model.T + model.L - model.p['D']*model.D0) + model.D0
model.demand_D = Constraint(model.I, rule=demand_D_rule)

def demand_l_rule(model, i):
    # Leisure decision remains unchanged
    return model.l[i] == (model.gamma/((model.alpha + model.beta + model.gamma)*model.phi[i]*model.w)) * \
           (model.phi[i]*model.w*model.T + model.L - model.p['D']*model.D0)
model.demand_l = Constraint(model.I, rule=demand_l_rule)

# --- Production Constraints (for each production sector j ∈ {'C','D'}) ---
def production_w_rule(model, j):
    # w = ε_j * T_prod[j]^(r-1) * F[j]^(1-r) * p[j]
    return model.w == model.epsilon[j] * model.T_prod[j]**(model.r - 1) * model.F[j]**(1 - model.r) * model.p[j]
model.production_w = Constraint(model.J, rule=production_w_rule)

def production_F_rule(model, j):
    # F[j] = (ε_j*T_prod[j]^r + (1-ε_j)*Z[j]^r)^(1/r)
    return model.F[j] == (model.epsilon[j]*model.T_prod[j]**model.r + (1 - model.epsilon[j])*model.Z[j]**model.r)**(1/model.r)
model.production_F = Constraint(model.J, rule=production_F_rule)

def production_indicator_rule(model, j):
    # Enforce the indicator: Z[j] <= x * T_prod[j]
    return model.Z[j] <= model.x * model.T_prod[j]
model.production_indicator = Constraint(model.J, rule=production_indicator_rule)

# --- Equilibrium Constraints ---
def equilibrium_T_rule(model):
    return model.T_prod['C'] + model.T_prod['D'] == sum(model.phi[i]*(model.T - model.l[i]) for i in model.I)
model.equilibrium_T = Constraint(rule=equilibrium_T_rule)

# Note: We drop the equilibrium condition for good C since p_C is numeraire.
def equilibrium_D_rule(model):
    # Market clearing for good D: p_D * Σ_i D_i = F_D * p_D.
    # Since p_D > 0, this simplifies to: Σ_i D_i = F_D.
    return model.p['D'] * sum(model.D[i] for i in model.I) == model.F['D'] * model.p['D']
model.equilibrium_D = Constraint(rule=equilibrium_D_rule)

model.obj = Objective(expr=0, sense=minimize)

solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)

#model.display()

from pyomo.environ import value
print("\nSolution Summary:")
print("Demand Variables:")
for i in model.I:
    print(f"  C[{i}] = {value(model.C[i])}")
    print(f"  D[{i}] = {value(model.D[i])}")
    print(f"  l[{i}] = {value(model.l[i])}")

print("\nProduction Variables:")
for j in model.J:
    print(f"  T_prod[{j}] = {value(model.T_prod[j])}")
    print(f"  Z[{j}]      = {value(model.Z[j])}")
    print(f"  F[{j}]      = {value(model.F[j])}")
    print(f"  p[{j}]      = {value(model.p[j])}")

print("\nCommon Wage:")
print(f"  w = {value(model.w)}")