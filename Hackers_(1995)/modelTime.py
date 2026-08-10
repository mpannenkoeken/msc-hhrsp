import gurobipy as grb
import pickle
import pandas as pd
from pathlib import Path
# combinations(iterable, r) gives every r-length subset, order-independent --
# e.g. combinations([1,2,3], 2) -> (1,2), (1,3), (2,3). used below to enumerate
# candidate pairs/couples out of a unit's members w/o hand-rolling nested loops
from itertools import combinations
from datetime import datetime
"""
HELPER FUNCTIONS
"""
# normalize(df): turn a results dataframe into a hashable, order-independent
# fingerprint, so two solutions that assign the same caregivers to the same
# days/localities compare as "equal" even if their rows/columns came out in
# a different order. used downstream to dedupe candidate solutions.
def normalize(df):
    # column order doesn't matter for equality, so pin it down first
    df = df.sort_index(axis=1)
    # cast everything to str so e.g. int 3 and "3" don't compare as different
    df = df.astype(str)
    # sort rows too (row order is also an artifact, not a real difference) --
    # kind="mergesort" is a STABLE sort, and reset_index(drop=True) throws out
    # the now-meaningless old row numbers instead of keeping them as a column
    df = df.sort_values(by=list(df.columns), kind="mergesort").reset_index(
        drop=True)
    # a dataframe itself isn't hashable (can't go in a set), so flatten it down
    # to a tuple of tuples -- tuples ARE hashable, which is what normalize()
    # is for: letting us dedupe solutions via a set() later on
    return tuple(tuple(str(x) for x in row) for row in df.to_numpy())

# get_sol(...): after m.optimize() has run, walk every decision variable and
# pull out a plain, human-readable dataframe of "who got assigned where" for
# the CURRENT solved state of the model. gets called once per objective/
# lambda-weight solve below, so we get one dataframe snapshot per solution.
def get_sol(D, units_d, L, xud, zuld, lu):
    rows = []
    for d in D:
        
        # paired caregivers
        # generator expression: "for u in (... if ...)" filters units_d[d] down
        # to just the pair-type units before we even start the loop body
        for u in (u for u in units_d[d] if u["type"] == "pair"):
            # .X reads a gurobi variable's SOLVED value (only valid after optimize()
            # has run) -- since xud is binary, .X will be ~0 or ~1, so > 0.5 is the
            # standard "was this actually selected" check (avoids float rounding issues)
            if xud[u["id"],d].X > 0.5:
                # find locality assignment
                # next(generator, default) returns the first l where zuld is "on",
                # or None if nothing matched -- python's version of a "find first" loop
                lAssignment = next(
                    (l for l in L if zuld[u["id"],l,d].X > 0.5),
                    None
                    )
                
                # zuld is only defined for DRIVING units (see wuld/zuld declarations
                # below), so non-driving pair units won't have a solved zuld entry --
                # fall back to their fixed "home" locality (lu) in that case
                if lAssignment == None:
                    lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                
                # plain two-person pair: just report both real caregiver ids
                if not u["couples"]:
                    rows.append({
                        "Day": d,
                        "Unit Type": u["type"],
                        "(eq.) Caregiver ID 1": u["members"][0],
                        "(eq.) Caregiver ID 2": u["members"][1],
                        "Locality Assignment": lAssignment
                        })
                
                # pair with a couple somewhere: figure out which two of the
                # three members are the actual glued-together couple
                if u["couples"]:
                    a, b = u["couple"]
                    remaining = [x for x in u["members"] if x not in (a,b)]
                    if u["dc"]:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"],
                            "(eq.) Caregiver ID 1": (a, b),
                            "(eq.) Caregiver ID 2": remaining[0],
                            "Locality Assignment": lAssignment
                            })
                    else:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"],
                            "(eq.) Caregiver ID 1": remaining[0],
                            "(eq.) Caregiver ID 2": (a, b),
                            "Locality Assignment": lAssignment
                            })
                    
        # solo caregivers
        for u in (u for u in units_d[d] if u["type"] == "solo"):
            if xud[u["id"],d].X > 0.5:
                # driving solo units have a real zuld to check, same pattern as pairs above
                if u["driver"]:
                    # find locality assignment
                    lAssignment = next(
                        (l for l in L if zuld[u["id"],l,d].X > 0.5),
                        None
                        )
                    
                    if lAssignment == None:
                        lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                    
                    if u["couples"]:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"],
                            "(eq.) Caregiver ID 1": u["members"],
                            "(eq.) Caregiver ID 2": None,
                            "Locality Assignment": lAssignment
                            })
                    else:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"][0],
                            "(eq.) Caregiver ID 1": u["members"],
                            "(eq.) Caregiver ID 2": None,
                            "Locality Assignment": lAssignment
                            })
                # non-driving solo units never get a zuld/wuld entry at all
                # (see the decision variable declarations: those are only built
                # over driveUnits_d), so just read their fixed home locality
                else:
                    lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                    if u["couples"]:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"],
                            "(eq.) Caregiver ID 1": u["members"],
                            "(eq.) Caregiver ID 2": None,
                            "Locality Assignment": lAssignment
                            })
                    else:
                        rows.append({
                            "Day": d,
                            "Unit Type": u["type"][0],
                            "(eq.) Caregiver ID 1": u["members"],
                            "(eq.) Caregiver ID 2": None,
                            "Locality Assignment": lAssignment
                            })
    # save results
    # pd.DataFrame(rows) turns our list of dicts straight into a table --
    # each dict becomes one row, dict keys become column names automatically
    currSol = pd.DataFrame(rows)
    return currSol
    
"""
INITIALIZATIONS
"""
run_start = datetime.now()
# figure out the folder we're currently in
currDir = Path(__file__).resolve().parent
currParent = currDir.parent
currAbuela = currParent.parent

# go find the nearby input pickle we need
# (this is the inputs.pkl written out at the end of the preprocessing script)
carersFile = currAbuela / "Home HealthCare Data" / "inputs.pkl"

# get the pickled sets and parameters out
# "rb" = read, binary -- pickle files are binary, mirrors the "wb" used to write them
with open(carersFile, "rb") as f:
    data = pickle.load(f)

# unpack the precomp dict back into individual named variables -- these are the
# exact same sets/params built in the preprocessing script, just deserialized
# (D = days, Cd = carers per day, units_d = feasible units per day, etc.)
D = data["D"]
Cd = data["Cd"]
units_d = data["units_d"]
driveUnits_d = data["drive_d"]
L = data["L"]
du = data["du"]
dij = data["dij"]
rul = data["rul"]
Fud = data["Fud"]
fud = data["fud"]
sud = data["sud"]
Vpd = data["Vpd"]
Vsd = data["Vsd"]
Vlpd = data["Vlpd"]
Vlsd = data["Vlsd"]
pairShare = data["pairShare"]
lu = data["lu"]
K = data["K"]

# define the tolerances on geospatial solo/pair distributions
epsi = 2 # tolerance on solo caregiving units
delta = 1 # tolerance on pair caregiving units

# Initialize the model
# grb.Model(name) creates an empty gurobi model shell -- variables, objective,
# and constraints all get attached to this same "m" object as we go
m = grb.Model("Caregiver_Assignments")

"""
DECLARATION OF DECISION VARIABLES
-- m.addVars(index_generator, vtype=...) is gurobi's bulk variable constructor:
   pass it a generator of index tuples and it returns a "tupledict" of
   variables keyed exactly by those tuples (e.g. wuld[u,l,d]), one call
   instead of manually looping and calling addVar() per variable
"""
# represents the linearization of $z_{ul}^d x_u^d$
# vtype=CONTINUOUS defaults to lb=0, ub=inf -- fine here since it's pinned to
# {0,1} by the wuld constraints further down rather than by its own bounds
wuld = m.addVars(((u,l,d) for d in D for u in driveUnits_d[d] for l in L) 
                  , vtype=grb.GRB.CONTINUOUS)

# 1 if unit u \in U^d assigned on day d, 0 else
xud = m.addVars(((u["id"],d) for d in D for u in units_d[d]), 
                 vtype=grb.GRB.BINARY)

# 1 if unit u \in U^d assigned to locality l \in L on day d, 0 else
zuld = m.addVars(((u,l,d) for d in D for u in driveUnits_d[d] for l in L) 
                  , vtype=grb.GRB.BINARY)

"""
OBJECTIVES
-- grb.quicksum(...) builds a gurobi linear expression from a generator, same
   role as python's builtin sum() but far faster for many-term gurobi exprs.
   dict.get((key), 0) reads a value from Fud/fud/sud w/ a 0 fallback, so a
   unit/day combo that never got a feasibility entry just contributes nothing
   instead of throwing a KeyError -- these are still just LINEAR EXPRESSIONS
   at this point, not yet attached to the model as THE objective (that
   happens later w/ m.setObjective)
"""
# objective one is to maximize the potential of familiar visits
potentialFamiliarVisits = grb.quicksum(
                            grb.quicksum(sud.get((u["id"],d),0) * xud[u["id"],d] + 
                                         grb.quicksum(Fud.get((v["id"],d),0) 
                                                      * xud[v["id"],d]
                                                      + fud.get((v["id"],d),0) 
                                                      * xud[v["id"],d]
                                                      for v in units_d[d] 
                                                      if v["type"] == "pair")
                                         for u in units_d[d] 
                                         if u["type"] == "solo") 
                            for d in D)
# objective two is to minimize the travel time for caregivers' assignments
carerTravelCeiling = grb.quicksum(
                             grb.quicksum(du[u] * xud[u,d] +
                                    grb.quicksum(rul[(u,l)] 
                                                 * zuld[u,l,d]
                                        for l in L)
                                 for u in driveUnits_d[d])
                        for d in D)
"""
CONSTRAINTS
-- m.addConstrs(generator) is the bulk equivalent of addVars: pass a generator
   of gurobi expressions (each ending in ==/<=/>= something) and it adds one
   constraint per item, all in a single call, instead of looping addConstr()
"""
# assign each available caregiver once per day available
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d] if i in u["members"])
             == 1 for d in D for i in Cd[d])
        
# do not assign u if an individual link is greater than K apart
# NOTE: "members = [m for m in u["members"]]" LOOKS like it's overwriting the
# model object m, but it isn't -- list comprehensions get their own private
# scope in python 3 (unlike a plain for-loop, whose loop variable does leak
# into the surrounding scope), so this m stays local to the comprehension and
# the outer model "m" is untouched once we're past this line
for d in D:
    for u in units_d[d]:
        members = [m for m in u["members"]]
        for i in members:
            for j in members:
                # .ub is a variable's upper bound -- setting it to 0 here forces
                # xud to 0 for this unit, i.e. hard-excludes it from being chosen
                # at all (cheaper than adding a whole extra constraint for it)
                if dij[i].loc[j] > K:        
                    xud[u["id"],d].ub = 0
                
                
# upper bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(lu[(u["id"],l)] * xud[u["id"],d] for u in units_d[d] 
                          if (u["type"] == "solo" and u["driver"] == False)) +
             grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                          if (u["type"] == "solo" and u["driver"])) <= 
             Vlsd.get((l,d), 0) / Vsd[d] * ((1 - pairShare[d]) 
                                            * grb.quicksum(xud[u["id"], d] 
                                                           for u in units_d[d]))
                          + epsi
             for l in L for d in D)

# lower bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(lu[(u["id"],l)] * xud[u["id"],d] for u in units_d[d] 
                          if (u["type"] == "solo" and u["driver"] == False)) +
             grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                          if (u["type"] == "solo" and u["driver"])) >= 
             Vlsd.get((l,d), 0) / Vsd[d] * ((1 - pairShare[d]) 
                                            * grb.quicksum(xud[u["id"], d] 
                                                           for u in units_d[d]))
                          - epsi
             for l in L for d in D)

# upper bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                             if u["type"] == "pair") 
                <= Vlpd.get((l,d), 0) / Vpd[d] * (pairShare[d] 
                                                  * grb.quicksum(xud[u["id"], d] 
                                                            for u in units_d[d]))
                             + delta
            for l in L for d in D)

# lower bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                             if u["type"] == "pair")
                >= Vlpd.get((l,d), 0) / Vpd[d] * (pairShare[d] 
                                                  * grb.quicksum(xud[u["id"], d] 
                                                            for u in units_d[d]))
                             - delta
            for l in L for d in D)

# get total pairs within tolerance
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "pair")
             <= pairShare[d] * grb.quicksum(xud[u["id"], d] for u in units_d[d])
             + delta for d in D)

m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "pair")
             >= pairShare[d] * grb.quicksum(xud[u["id"], d] for u in units_d[d])
             - delta for d in D)

# get total solos within tolerance
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "solo")
                <= (1 - pairShare[d]) * grb.quicksum(xud[u["id"], d] 
                                                     for u in units_d[d])
                + epsi for d in D)

m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "solo")
             >= (1 - pairShare[d]) * grb.quicksum(xud[u["id"], d] 
                                                      for u in units_d[d])
             - epsi for d in D)

# make wuld do what i want it to
# standard McCormick/big-M linearization of wuld = zuld * xud (both binary):
# these three constraints together pin wuld to exactly that product
m.addConstrs(wuld[u,l,d] <= xud[u,d] for d in D 
             for u in driveUnits_d[d] for l in L)
m.addConstrs(wuld[u,l,d] <= zuld[u,l,d] for d in D 
             for u in driveUnits_d[d] for l in L)
m.addConstrs(wuld[u,l,d] >= zuld[u,l,d] + xud[u,d] - 1
             for d in D for u in driveUnits_d[d] for l in L)

"""
GUROBI WORKS ITS MAGIC
-- this whole section is a lexicographic / epsilon-constraint sweep over the
   two objectives: solve for familiarity alone first to find the best-case
   travel ceiling, solve for travel alone to find the best-case travel floor,
   then re-optimize familiarity again several times while capping travel at
   points in between -- building out a small frontier of tradeoff solutions
   rather than committing to one single-objective answer
"""
# optimize for familiarity/objective 1
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
# OutputFlag = 0 silences gurobi's solver log (no branch-and-bound chatter
# printed to console) -- flip to 1 if you want to see solver progress
m.Params.OutputFlag = 0
m.optimize()
# store travel time in unrestricted case
# .getValue() evaluates a linear expression at the CURRENT solved variable
# values -- i.e. "what's carerTravelCeiling actually worth in this solution",
# even though carerTravelCeiling isn't the active objective right now
maxTravel = carerTravelCeiling.getValue()

allResults = []
# store objective 1 results
maxTravelSol = get_sol(D, units_d, L, xud, zuld, lu)
allResults.append(maxTravelSol)

print(f"Maximum Observed Travel Ceiling: {maxTravel}")

# add constraint on travel time
# m.addConstr (singular) adds ONE constraint and hands back a Constr object we
# can keep a reference to -- we need that handle below to tweak its RHS in place
travel = m.addConstr(carerTravelCeiling <= maxTravel)

# optimize for travel time/objective 2
m.setObjective(carerTravelCeiling, sense=grb.GRB.MINIMIZE)
# m.update() flushes pending changes (new objective/constraint) into the
# model before the next optimize()/attribute read -- gurobi batches edits
# for performance, so this is needed any time you change structure mid-script
m.update()
m.optimize()
# store travel time when minimized
# .ObjVal reads the solved objective value directly (equivalent to
# carerTravelCeiling.getValue() here, since it IS the active objective now)
minTravel = m.ObjVal

# store objective 2 results
minTravelSol = get_sol(D, units_d, L, xud, zuld, lu)
allResults.append(minTravelSol)

print(f"Minimum Observed Travel Ceiling: {minTravel}")

# reset model objective as objective 1/familiarity
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
m.update()

# optimize for linear combinations of min and max travel times
# lamb sweeps the travel cap between the two extremes found above: lamb=0.25
# means "allow travel up to 25% of the way from best-case up to worst-case",
# etc -- each pass re-solves familiarity under a progressively looser travel cap
for lamb in [0.25, 0.5, 0.75]:
    # update rhs of travel constraint and solve
    # .rhs lets us edit an EXISTING constraint's right-hand side in place,
    # rather than removing travel and calling addConstr all over again
    travel.rhs = lamb * maxTravel + (1 - lamb) * minTravel
    m.update()
    print(f"Current Travel RHS: {travel.rhs}")
    m.optimize()
    print(f"Observed Travel: {carerTravelCeiling.getValue()}")
    
    # save results
    currSol = get_sol(D, units_d, L, xud, zuld, lu)
    allResults.append(currSol)
    
"""
COMPARE RESULTS AND EXPORT WHILE IGNORING DUPLICATES
-- across the 5 solves above (2 single-objective + 3 lambda-weighted), some
   may have landed on the exact same assignment -- normalize() + a set gives
   us cheap duplicate detection so we only export genuinely distinct solutions
"""
unique_results = []
seen = set()
for df in allResults:
    norm=normalize(df)
    if norm not in seen:
        seen.add(norm)
        unique_results.append(df)

# enumerate(...) hands back (index, item) pairs, so i+1 gives us a clean
# 1-based file suffix instead of starting the naming at candidate_pairings_0
for i, df in enumerate(unique_results):
    unique_results[i].to_csv(f"candidate_pairings_{i+1}.csv", index=False)
    
run_end = datetime.now()
print(f"Total Optimisation Runtime: {run_end - run_start}")