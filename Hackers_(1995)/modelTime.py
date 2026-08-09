#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:37:47 2026

@author: nolanbearw
"""

import gurobipy as grb
import pickle
import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime
"""
HELPER FUNCTIONS
"""
def normalize(df):
    df = df.sort_index(axis=1)
    df = df.astype(str)
    df = df.sort_values(by=list(df.columns), kind="mergesort").reset_index(
        drop=True)
    return tuple(tuple(str(x) for x in row) for row in df.to_numpy())

def split_unit(u, allCouples_d):
    members = list(u["members"])
    
    # no couple case
    if not u["couples"]:
        return {
            "couple": None,
            "single": members if len(members) == 1 else None,
            "pair": members if len(members) == 2 else None
        }

    # --- find which two form the couple ---
    couple_set = {tuple(sorted(c)) for c in allCouples_d}
    
    for a, b in combinations(members, 2):
        if tuple(sorted((a, b))) in couple_set:
            remaining = [x for x in members if x not in (a, b)]
            return {
                "couple": (a, b),
                "single": remaining[0] if remaining else None
            }

    # fallback (should not happen)
    return {
        "couple": None,
        "single": members
    }

def get_sol(D, units_d, L, xud, zuld, lu, allCouples_d):
    rows = []
    for d in D:
        
        todaysCouples = {tuple(sorted(c)) for c in allCouples_d[d]}
        
        # paired caregivers
        for u in (u for u in units_d[d] if u["type"] == "pair"):
            if xud[u["id"],d].X > 0.5:
                # find locality assignment
                lAssignment = next(
                    (l for l in L if zuld[u["id"],l,d].X > 0.5),
                    None
                    )
                
                if lAssignment == None:
                    lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                
                if not u["couples"]:
                    rows.append({
                        "Day": d,
                        "Unit Type": u["type"],
                        "(eq.) Caregiver ID 1": u["members"][0],
                        "(eq.) Caregiver ID 2": u["members"][1],
                        "Locality Assignment": lAssignment
                        })
                
                if u["couples"]:
                    for a, b in combinations(u["members"], 2):
                        if tuple(sorted((a,b))) in todaysCouples:
                            remaining = [x for x in u["members"] if x not in (a,b)]
                            parts = u["id"].split("_")
                            if parts[0] == remaining:
                                rows.append({
                                    "Day": d,
                                    "Unit Type": u["type"],
                                    "(eq.) Caregiver ID 1": remaining,
                                    "(eq.) Caregiver ID 2": (a, b),
                                    "Locality Assignment": lAssignment
                                    })
                    
        # solo caregivers
        for u in (u for u in units_d[d] if u["type"] == "solo"):
            if xud[u["id"],d].X > 0.5:
                if u["driver"]:
                    # find locality assignment
                    lAssignment = next(
                        (l for l in L if zuld[u["id"],l,d].X > 0.5),
                        None
                        )
                    
                    if lAssignment == None:
                        lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                    
                    rows.append({
                        "Day": d,
                        "Unit Type": u["type"],
                        "(eq.) Caregiver ID 1": u["members"],
                        "(eq.) Caregiver ID 2": None,
                        "Locale Assignment": lAssignment
                        })
                else:
                    lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                    rows.append({
                        "Day": d,
                        "Unit Type": u["type"],
                        "(eq.) Caregiver ID 1": u["members"],
                        "(eq.) Caregiver ID 2": None,
                        "Locale Assignment": lAssignment
                        })
    # save results
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
carersFile = currAbuela / "Home HealthCare Data" / "inputs.pkl"

# get the pickled sets and parameters out
with open(carersFile, "rb") as f:
    data = pickle.load(f)

D = data["D"]
Cd = data["Cd"]
units_d = data["units_d"]
driveUnits_d = data["drive_d"]
allCouples_d = data["allCouples"]
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
m = grb.Model("Caregiver_Assignments")

"""
DECLARATION OF DECISION VARIABLES
"""
# represents the linearization of $z_{ul}^d x_u^d$
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
"""
# assign each available caregiver once per day available
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d] if i in u["members"])
             == 1 for d in D for i in Cd[d])
        
# do not assign u if an individual link is greater than K apart
for d in D:
    for u in units_d[d]:
        members = [m for m in u["members"]]
        for i in members:
            for j in members:
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
m.addConstrs(wuld[u,l,d] <= xud[u,d] for d in D 
             for u in driveUnits_d[d] for l in L)
m.addConstrs(wuld[u,l,d] <= zuld[u,l,d] for d in D 
             for u in driveUnits_d[d] for l in L)
m.addConstrs(wuld[u,l,d] >= zuld[u,l,d] + xud[u,d] - 1
             for d in D for u in driveUnits_d[d] for l in L)

"""
GUROBI WORKS ITS MAGIC
"""
# optimize for familiarity/objective 1
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
m.Params.OutputFlag = 0
m.optimize()
# store travel time in unrestricted case
maxTravel = carerTravelCeiling.getValue()

allResults = []
# store objective 1 results
maxTravelSol = get_sol(D, units_d, L, xud, zuld, lu, allCouples_d)
allResults.append(maxTravelSol)

print(f"Maximum Observed Travel Ceiling: {maxTravel}")

# add constraint on travel time
travel = m.addConstr(carerTravelCeiling <= maxTravel)

# optimize for travel time/objective 2
m.setObjective(carerTravelCeiling, sense=grb.GRB.MINIMIZE)
m.update()
m.optimize()
# store travel time when minimized
minTravel = m.ObjVal

# store objective 2 results
minTravelSol = get_sol(D, units_d, L, xud, zuld, lu, allCouples_d)
allResults.append(minTravelSol)

print(f"Minimum Observed Travel Ceiling: {minTravel}")

# reset model objective as objective 1/familiarity
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
m.update()

# optimize for linear combinations of min and max travel times
for lamb in [0.25, 0.5, 0.75]:
    # update rhs of travel constraint and solve
    travel.rhs = lamb * maxTravel + (1 - lamb) * minTravel
    m.update()
    print(f"Current Travel RHS: {travel.rhs}")
    m.optimize()
    print(f"Observed Travel: {carerTravelCeiling.getValue()}")
    
    # save results
    currSol = get_sol(D, units_d, L, xud, zuld, lu, allCouples_d)
    allResults.append(currSol)
    
"""
COMPARE RESULTS AND EXPORT WHILE IGNORING DUPLICATES
"""
unique_results = []
seen = set()
for df in allResults:
    norm=normalize(df)
    if norm not in seen:
        seen.add(norm)
        unique_results.append(df)


for i, df in enumerate(unique_results):
    unique_results[i].to_csv(f"candidate_pairings_{i+1}.csv", index=False)
    
run_end = datetime.now()
print(f"Total Optimisation Runtime: {run_end - run_start}")