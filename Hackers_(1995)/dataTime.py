import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pickle

"""
HELPERS AND FILES
"""

# figure out the folder we're currently in
currDir = Path(__file__).resolve().parent
currParent = currDir.parent
currAbuela = currParent.parent

# go find the nearby data files we need
carersFile = currParent.parent / "Home HealthCare Data" / "NW_Carers.csv"
clientsFile = currParent.parent / "Home HealthCare Data" / "NW_Clients.csv"
visitsFile = currParent.parent / "Home HealthCare Data" / "NW_CareVisits.csv"
travelFile = currParent.parent / "Home HealthCare Data" / "NW_TravelTimes.csv"

# convert "h:mm" strings into integer minutes
def to_minutes(x):
    if pd.isna(x):
        return None
    h, m = map(int, str(x).strip().split(":"))
    return 60*h + m


# check whether a visit window can start during an agent's shift
# we only care that the visit *can* begin during the shift
def within_shift(shift_start, shift_end, visit_start, visit_end):
    return (shift_start <= visit_start <= shift_end) or (shift_start 
                                                         <= visit_end <= 
                                                         shift_end)


# check if a unit (agent or couple) is familiar with a client
def known_unit(units, unit, known_set):
    members = units[unit]["members"]
    return any(m in known_set for m in members)

"""
IMPORTS
"""

# the carer file ends in a ragged column
carersData = []
with open(carersFile) as f:
    # get the header row outta the way
    header = next(f).strip().split(";")
    
    # go row by row through this csv
    for line in f:
        # extract each value
        parts = [p.strip() for p in line.split(";")]
        # first five cols are consistent w/ csv structure
        fixedLine = parts[:5]
        # \exists a variable length tail in last col(s)
        tail = [p for p in parts[5:] if p != ""]
        
        # make a fixed row and add to running data list
        row = fixedLine + [list(map(str, tail)) if tail else []]
        carersData.append(row)

# create a dataframe from the extracted data and cast ids as ints
carers_df = pd.DataFrame(carersData, columns=header, index=None)
carers_df["Carer ID"] = carers_df["Carer ID"].astype(int)

# convert the carer start/end times to a workable type
string_start = carers_df["Shift Start Time"].astype(str).str.strip()
string_end = carers_df["Shift End Time"].astype(str).str.strip()

hms = string_start.str.split(":", expand = True)
hme = string_end.str.split(":", expand = True)

carers_df["start_min"] = (
    pd.to_timedelta(hms[0].astype(int), unit ="h") +
    pd.to_timedelta(hms[1].astype(int), unit ="m")
    )

carers_df["end_min"] = (
    pd.to_timedelta(hme[0].astype(int), unit ="h") +
    pd.to_timedelta(hme[1].astype(int), unit ="m")
    )

# make a lookup table for carer shift start/end
starts = carers_df.set_index("Carer ID")["start_min"].to_dict()
ends = carers_df.set_index("Carer ID")["end_min"].to_dict()

# the client file ends in a ragged column
clientsData = []
with open(clientsFile) as f:
    # get the header row outta the way
    newheader = next(f).strip().split(";")
    
    # go row by row through this god-awful mess pretending to be a csv
    for line in f:
        # extract each value
        parts = [p.strip() for p in line.split(";")]
        # first two cols are consistent w/ csv structure
        fixedLine = parts[:2]
        # \exists a variable length tail in last col(s)
        tail = [p for p in parts[2:] if p != ""]
        
        # make a fixed row and add to running data list
        row = fixedLine + [list(map(int, tail)) if tail else []]
        clientsData.append(row)

# create a dataframe from the relevant extracted data and cast ids as ints
clients_df = pd.DataFrame(clientsData, 
                          columns=["Client ID", 
                                   "Gender Preference", 
                                   "Known Carers"], 
                          index=None)
clients_df["Client ID"] = clients_df["Client ID"].astype(int)

# create a dataframe from the visit data file
visits_df = pd.read_csv(visitsFile, sep=";", index_col="Visit ID")

# begin converting duration to timedelta by reading as a string
string_dur = visits_df["Visit Duration"].astype(str).str.strip()

# split into hours and minutes
hm = string_dur.str.split(":", expand=True)

# now parse as a timedelta
visits_df["Visit Duration"] = (
    pd.to_timedelta(hm[0].astype(int), unit="h") +
    pd.to_timedelta(hm[1].astype(int), unit="m")
)

# standardize naming convention across frames
visits_df = visits_df.rename(columns={"ClientID": "Client ID"})

# normalize time windows to workable type
string_start = visits_df["Time Window Start"].astype(str).str.strip()
string_end = visits_df["Time Window End"].astype(str).str.strip()

hms = string_start.str.split(":", expand = True)
hme = string_end.str.split(":", expand = True)

visits_df["Time Window Start"] = (
    pd.to_timedelta(hms[0].astype(int), unit ="h") +
    pd.to_timedelta(hms[1].astype(int), unit ="m")
    )

visits_df["Time Window End"] = (
    pd.to_timedelta(hme[0].astype(int), unit ="h") +
    pd.to_timedelta(hme[1].astype(int), unit ="m")
    )

# split pair visits from single visits
pair_visits_df = visits_df[visits_df["Number of Carers"] == 2]
single_visits_df = visits_df[visits_df["Number of Carers"] == 1]

# add information on known caregivers per visit
pair_visits_df = pair_visits_df.merge(clients_df, on="Client ID")
single_visits_df = single_visits_df.merge(clients_df, on = "Client ID")

# create a dataframe from the travel data file
travel_df = pd.read_csv(travelFile, sep=";", index_col=0)
# travel time matrix ends up with an empty column
travel_df = travel_df.dropna(axis=1, how="all")

# convert data to ints for later
travel_df = travel_df.map(to_minutes)
travel_df.index = travel_df.index.astype(int)
travel_df.columns = travel_df.columns.astype(int)

# get a list of all client/carer ids
carers = np.array(carers_df["Carer ID"])
clients = np.array(clients_df["Client ID"])

"""
EXTRACT DAYS AND CAREGIVER AVAILABILITY
"""

# get the days we are planning
Days = np.array(visits_df["Visit Date"].unique())

# figure out which carers and driving carers we have on those days
carers_exploded = carers_df.explode("Available Working Days")
drivers_exploded = carers_exploded.loc[
    carers_exploded["Driver"].astype(int) == 1]

# grab every carer id available daily
Cd = (
    carers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

# grab every driver id available daily
CdD = (
    drivers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

"""
BUILD ALL FEASIBLE PART-TIME COUPLES
"""
# initialize couple dicts
allCouples = {d: [] for d in Days}
driveCouples = {d: [] for d in Days}

for d in Days:
    # get the lists of carers and drivers today
    carers_d = list(Cd[d])
    CdD_set = set(CdD[d])
    
    # check each carer
    for idx, c in enumerate(carers_d):
        # check each remaining carer
        for c_prime in carers_d[idx+1:]:
            
            # non-overlapping shifts means feasible coupling
            if ends[c] <= starts[c_prime] or ends[c_prime] <= starts[c]:
                # this is a couple so add
                couple = (c, c_prime)
                allCouples[d].append(couple)
                
                # if both are drivers, this is a driving couple
                if c in CdD_set and c_prime in CdD_set:
                    driveCouples[d].append(couple)
                    
"""
BUILD FEASIBLE UNITS PER DAY
"""

feasibleUnits_d = {d: [] for d in Days}

for d in Days:
    seen_today = set()
    # solo real caregivers
    for c in Cd[d]:
        members = (c,)
        unitID = "_".join(map(str, members)) + "_s"
        key = (tuple(sorted(members)), "solo")
        unit = {
            "id": unitID,
            "members": members,
            "type": "solo",
            "couples": False,
            "driver": c in CdD[d]
        }
        if key not in seen_today:
            seen_today.add(key)
            feasibleUnits_d[d].append(unit)
    
    # solo couplings (treated as full-time frankenstein caregivers)
    for (a,b) in allCouples[d]:
        members = (a,b)
        unitID = "_".join(map(str, members)) + "_s"
        key = (tuple(sorted(members)), "solo")
        unit = {
            "id": unitID,
            "members": members,
            "type": "solo",
            "couples": True,
            "driver": (a,b) in driveCouples[d]
        }
        if key not in seen_today:
            seen_today.add(key)
            feasibleUnits_d[d].append(unit)
    
    # real driver + real caregiver
    for i in CdD[d]:
        for j in Cd[d]:
            if i != j and ((starts[i] <= starts[j]
                            and ends[i] >= ends[j]) or (starts[j] <= starts[i]
                                                        and ends[j] >= ends[i]
                                                        )):
                members = (i,j)
                unitID = "_".join(map(str,members)) + "_p"
                key = (tuple(sorted(members)), "pair")
                unit = {
                    "id": unitID,
                    "members": members,
                    "type": "pair",
                    "couples": False,
                    "driver": True
                    }
                
                if key not in seen_today:
                    seen_today.add(key)
                    feasibleUnits_d[d].append(unit)
    
    # driver coupling + real caregiver
    for (a,b) in driveCouples[d]:
        for j in Cd[d]:
            if j not in (a,b):
                members = (a,b,j)
                unitID = "_".join(map(str,members)) + "_p"
                key = (tuple(sorted(members)), "pair")
                unit = {
                    "id": unitID,
                    "members": members,
                    "type": "pair",
                    "couples": True,
                    "driver": True
                }
                
                if key not in seen_today:
                    seen_today.add(key)
                    feasibleUnits_d[d].append(unit)
    
    # real driver + coupling
    for i in CdD[d]:
        for (a,b) in allCouples[d]:
            if i not in (a,b):
                members = (i,a,b)
                unitID = "_".join(map(str,members)) + "_p"
                key = (tuple(sorted(members)), "pair")
                unit = {
                    "id": unitID,
                    "members": members,
                    "type": "pair",
                    "couples": True,
                    "driver": True
                }
                
                if key not in seen_today:
                    seen_today.add(key)
                    feasibleUnits_d[d].append(unit)

# split drivers set now for easier future stuff
driverUnits = {
    d: [u["id"] for u in feasibleUnits_d[d] if u["driver"]]
    for d in Days
}

"""
GET VISIT MINUTES AND SHARES
"""
# count the number of visit minutes daily
Vd = visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
# split by pair and solo
Vpd = pair_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
Vsd = single_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()

# daily pair/solo share initialization
Pd = {}
Sd = {}
for d in Days:
    # the share of pair/solo minutes today
    pairShare = Vpd[d]/Vd[d]
    # print(f"{d} share of visit minutes requiring pair: {pairShare}%")
    # there is a derivation for this i swear
    Pd[d] = int(round(pairShare * len(Cd[d]) / (1 + pairShare)))
    # bound it from below by 0 and above by max pairs possible today
    Pd[d] = max(0, min(Pd[d], len(Cd[d]) // 2))
    # get target solo assignments from target pair assignments
    Sd[d] = len(Cd[d]) - 2*Pd[d]

# default max travel for pair assignment (temporary?)
K = 40

"""
BUILD sud: SOLO VISIT MINUTES PER UNIT (SHIFT-AWARE)
"""
# standardize column labels
single_visits_df.columns = single_visits_df.columns.str.strip().str.replace(" ", "_").str.lower()

# initialize carer-day -> duration dictionary
sud = defaultdict(lambda: pd.Timedelta(0))

# check every single carer visit
for row in single_visits_df.itertuples(index=False):
    # for the day, duration, known carers, and time window
    d = row.visit_date
    duration = row.visit_duration
    known = set(map(int, row.known_carers))
    v_start = row.time_window_start
    v_end = row.time_window_end
    
    unit_list = feasibleUnits_d[d]
    eqsolo_list = [u for u in unit_list if u["type"] == "solo"]
    
    for u in eqsolo_list:
        if not u["couples"]:
            i = u["members"][0]
            if i in known and within_shift(starts[i], ends[i], v_start, v_end):
                sud[(u["id"],d)] += duration
                
        else:
            i, j = u["members"]
            if i in known and within_shift(starts[i], ends[i], v_start, v_end):
                sud[(u["id"],d)] += duration
            elif j in known and within_shift(starts[j], ends[j], v_start, v_end):
                sud[(u["id"],d)] += duration
        

"""
BUILD Fijd AND fijd (SHIFT-AWARE, WITH COUPLES)
"""
# standardize column names
pair_visits_df.columns = pair_visits_df.columns.str.strip().str.replace(" ", "_").str.lower()

# initialize default dictionaries for both active caregivers known
Fud = defaultdict(lambda: pd.Timedelta(0))
# and at least one active caregiver known
fud = defaultdict(lambda: pd.Timedelta(0))

for row in pair_visits_df.itertuples(index=False):
    d = row.visit_date
    duration = row.visit_duration
    known = set(map(int, row.known_carers))
    v_start = row.time_window_start
    v_end = row.time_window_end
    
    unit_list = feasibleUnits_d[d]
    eqpair_list = [u for u in unit_list if u["type"] == "pair"]

    for u in eqpair_list:
        
        if not u["couples"]:
            i, j = u["members"]
            
            i_active = within_shift(starts[i], ends[i], v_start, v_end)
            j_active = within_shift(starts[j], ends[j], v_start, v_end)
            
            if not i_active and j_active:
                continue
            
            i_known = i in known
            j_known = j in known
            
            if i_known and j_known:
                Fud[(u["id"],d)] += duration
            if i_known or j_known:
                fud[(u["id"],d)] += duration
        
        if u["couples"]:
            i, j, k = u["members"]
            
            i_active = within_shift(starts[i], ends[i], v_start, v_end)
            j_active = within_shift(starts[j], ends[j], v_start, v_end)
            k_active = within_shift(starts[k], ends[k], v_start, v_end)
            
            i_known = i in known
            j_known = j in known
            k_known = k in known
        
            if i_active and j_active:
                if i_known and j_known:
                    Fud[(u["id"],d)] += duration
                    
                if i_known or j_known:
                    fud[(u["id"],d)] += duration
                
            elif i_active and k_active:
                if i_known and k_known:
                    Fud[(u["id"],d)] += duration
                    
                if i_known or k_known:
                    fud[(u["id"],d)] += duration
                    
            elif j_active and k_active:
                if j_known and k_known:
                    Fud[(u["id"],d)] += duration
                    
                if j_known or k_known:
                    fud[(u["id"],d)] += duration
            
"""
CREATE LOCALITIES AND DERIVE RELEVANT SET AND PARAMETERS
"""
# first augment distance matrix to include carer -> carer
allNodes = list(carers) + list(clients)
n = len(allNodes)

D = pd.DataFrame(np.zeros((n,n)), index= allNodes, columns= allNodes)

# keep well-defined client->client and carer->client distances
for i in carers:
    for j in clients:
        D.loc[i,j] = travel_df.loc[i,j]
        D.loc[j,i] = travel_df.loc[i,j]
        
for i in clients:
    for j in clients:
        D.loc[i,j] = travel_df.loc[i,j]
        D.loc[j,i] = travel_df.loc[i,j]

# next dummy carer->carer by going to nearest clients as middlemen
nearestClient = {
    c: travel_df.loc[c].astype(float).idxmin()
    for c in carers    
}

for i in carers:
    for j in carers:
        if i == j:
            D.loc[i,j] = 0
        else:
            ci = nearestClient[i]
            cj = nearestClient[j]
            D.loc[i,j] = D.loc[i, ci] + D.loc[ci,cj] + D.loc[cj,j]
           
# then hierarchical cluster to create localities
numLocalities = 8
D_clients = D.loc[clients, clients]

condensed = squareform(D_clients.values)
Z = linkage(condensed, method="complete")
clientLocalities = fcluster(Z, numLocalities, criterion="maxclust")

localityMap = dict(zip(D_clients.index, clientLocalities))
L = [int(x) for x in set(clientLocalities)]

# now find each units's locality, furthest neighbor in all other localities
# and intra-unit distances
rul = {}
lu = {}
du = {}

for d in Days:
    for u in feasibleUnits_d[d]:
        u_id = u["id"]
        members = u["members"]
        
        # choose representative (i.e. nearest) client
        rep = min(
            members,
            key=lambda a: travel_df.loc[a].astype(float).min()
        )
        
        home = localityMap[nearestClient[rep]]
        
        for l in L:
            lu[(u_id, l)] = int(l == home)
            members_in_l = [n for n in clients if localityMap[n] == l]
            
            if lu[(u_id, l)] == 1:
                rul[(u_id, l)] = 0
            else:
                rul[(u_id, l)] = max(
                    max(D.loc[a, n] for n in members_in_l)
                    for a in members
                )
                
        du[u_id] = sum(D.loc[i,j] for i in members for j in members)


# finally break down visit minutes by locality and pair/single status        
pair_visits_df["locality"] = pair_visits_df["client_id"].map(localityMap)
single_visits_df["locality"] = single_visits_df["client_id"].map(localityMap)

Vpld = (
        pair_visits_df.groupby(["locality", "visit_date"])["visit_duration"]
        .sum()
        .to_dict()
)

Vsld = (
        single_visits_df.groupby(["locality", "visit_date"])["visit_duration"]
        .sum()
        .to_dict()
)

# convert all timedeltas to floats/ints
Fud = {k: v.total_seconds()/60 for k, v in Fud.items()}
fud = {k: v.total_seconds()/60 for k, v in fud.items()}
sud = {k: v.total_seconds()/60 for k, v in sud.items()}
Vd = {k: v.total_seconds()/60 for k, v in Vd.items()}
Vpd = {k: v.total_seconds()/60 for k, v in Vpd.items()}
Vsd = {k: v.total_seconds()/60 for k, v in Vsd.items()}
Vpld = {k: v.total_seconds()/60 for k, v in Vpld.items()}
Vsld = {k: v.total_seconds()/60 for k, v in Vsld.items()}

# export clients' locality assignments if using these localities
# clientAssignment = pd.DataFrame(list(localityMap.items()), columns=["Client ID", "Locality"])
# clientAssignment.to_csv(currParent.parent / "Home HealthCare Data" / "locality_assignments.csv", index=False)

"""
EXPORT SETS AND PARAMETERS AS A PICKLE
"""
precomp = {
        "D": Days,
        "Cd": Cd,
        "units_d": feasibleUnits_d,
        "drive_d": driverUnits,
        "L": L,
        "du": du,
        "dij": D,
        "rul": rul,
        "Fud": dict(Fud),
        "fud": dict(fud),
        "sud": sud,
        "Vpd": Vpd,
        "Vsd": Vsd,
        "Vlpd": Vpld,
        "Vlsd": Vsld,
        "Pd": Pd,
        "Sd": Sd,
        "lu": lu,
        "K": K
}

with open(currParent.parent / "Home HealthCare Data" / "inputs.pkl", "wb") as f:
    pickle.dump(precomp, f)