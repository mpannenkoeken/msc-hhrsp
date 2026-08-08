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
import math

"""
HELPER FUNCTIONS
"""
def normalize(df):
    df = df.sort_index(axis=1)
    return df.sort_values(
        by=list(df.columns), kind="mergesort"
                ).reset_index(drop=True)

def get_sol(D, units_d, L, xud, zuld, lu):
    rows = []
    for d in D:
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
                
                rows.append({
                    "Day": d,
                    "Unit ID": u["id"],
                    "Locale Assignment": lAssignment
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
                        "Unit ID": u["id"],
                        "Locale Assignment": lAssignment
                        })
                else:
                    lAssignment = next(l for l in L if lu[(u["id"],l)] == 1)
                    rows.append({
                        "Day": d,
                        "Unit ID": u["id"],
                        "Locale Assignment": lAssignment
                        })
    # save results
    currSol = pd.DataFrame(rows)
    return currSol
    
"""
INITIALIZATIONS
"""
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
Pd = data["Pd"]
Sd = data["Sd"]
lu = data["lu"]
K = data["K"]

# define the tolerances on geospatial solo/pair distributions
epsi = 0 # tolerance on solo caregiving units
delta = 0 # tolerance on pair caregiving units

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
                    
# do not assign u if sum of links is greater than 2K apart
for d in D:
    for u in units_d[d]:
        if du[u["id"]] > 2 * K:
            xud[u["id"],d].ub = 0
                
# upper bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(lu[(u["id"],l)] * xud[u["id"],d] for u in units_d[d] 
                          if (u["type"] == "solo" and u["driver"] == False)) +
             grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                          if (u["type"] == "solo" and u["driver"])) <= 
             math.ceil(Vlsd.get((l,d), 0) / Vsd[d] * (Sd[d] + epsi))
             for l in L for d in D)

# lower bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(lu[(u["id"],l)] * xud[u["id"],d] for u in units_d[d] 
                          if (u["type"] == "solo" and u["driver"] == False)) +
             grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                          if (u["type"] == "solo" and u["driver"])) >= 
             math.floor(Vlsd.get((l,d), 0) / Vsd[d] * (Sd[d] - epsi))
             for l in L for d in D)

# upper bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                             if u["type"] == "pair") 
                <= math.ceil(Vlpd.get((l,d), 0) / Vpd[d] * (Pd[d] + delta))
            for l in L for d in D)

# lower bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(wuld[u["id"],l,d] for u in units_d[d]
                             if u["type"] == "pair")
                >= math.floor(Vlpd.get((l,d), 0) / Vpd[d] * (Pd[d] + delta))
            for l in L for d in D)

# get total pairs within tolerance
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "pair")
             <= Pd[d] + delta for d in D)

m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "pair")
             >= Pd[d] - delta for d in D)

# get total solos within tolerance
m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "solo")
                <= Sd[d] + epsi for d in D)

m.addConstrs(grb.quicksum(xud[u["id"],d] for u in units_d[d]
                          if u["type"] == "solo")
             >= Sd[d] - epsi for d in D)

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
maxTravelSol = get_sol(D, units_d, L, xud, zuld, lu)
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
minTravelSol = get_sol(D, units_d, L, xud, zuld, lu)
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
    currSol = get_sol(D, units_d, L, xud, zuld, lu)
    allResults.append(currSol)
    
"""
COMPARE RESULTS AND EXPORT WHILE IGNORING DUPLICATES
"""
unique_results = []
seen = []
for df in allResults:
    norm=normalize(df)
    if not any(norm.equals(existing) for existing in seen):
        seen.append(norm)
        unique_results.append(df)


for i, df in enumerate(unique_results):
    unique_results[i].to_csv(f"candidate_pairings_{i+1}.csv", index=False)
    