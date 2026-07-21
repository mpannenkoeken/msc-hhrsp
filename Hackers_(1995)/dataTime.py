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


def to_minutes(x):
    if pd.isna(x):
        return None
    h, m = map(int, str(x).strip().split(":"))
    return 60*h + m

"""
IMPORTS
"""

# the carer file ends in a ragged column
carersData = []
with open(carersFile) as f:
    # get the header row outta the way
    header = next(f).strip().split(";")
    
    # go row by row through this god-awful mess pretending to be a csv
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

carers_df = pd.DataFrame(carersData, columns=header, index=None)
carers_df["Carer ID"] = carers_df["Carer ID"].astype(int)

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
        
clients_df = pd.DataFrame(clientsData, columns=["Client ID", "Gender Preference", "Known Carers"], index=None)
clients_df["Client ID"] = clients_df["Client ID"].astype(int)
# these files should be relatively straightforward
visits_df = pd.read_csv(visitsFile, sep=";", index_col="Visit ID")

string_dur = visits_df["Visit Duration"].astype(str).str.strip()

# split into hours and minutes
hm = string_dur.str.split(":", expand=True)

visits_df["Visit Duration"] = (
    pd.to_timedelta(hm[0].astype(int), unit="h") +
    pd.to_timedelta(hm[1].astype(int), unit="m")
)

visits_df = visits_df.rename(columns={"ClientID": "Client ID"})

pair_visits_df = visits_df[visits_df["Number of Carers"] == 2]
single_visits_df = visits_df[visits_df["Number of Carers"] == 1]

pair_visits_df = pair_visits_df.merge(clients_df, on="Client ID")
single_visits_df = single_visits_df.merge(clients_df, on = "Client ID")

# travel time matrix ends up with an empty column for some reason
travel_df = pd.read_csv(travelFile, sep=";", index_col=0)
travel_df = travel_df.dropna(axis=1, how="all")

# convert data to ints for later
travel_df = travel_df.map(to_minutes)
travel_df.index = travel_df.index.astype(int)
travel_df.columns = travel_df.columns.astype(int)

# get a list of all client/carer ids
carers = np.array(carers_df["Carer ID"])
clients = np.array(clients_df["Client ID"])

"""
EXTRACT LOCALITY-INDEPENDENT SETS AND PARAMETERS
"""

# get the days we are planning
Days = np.array(visits_df["Visit Date"].unique())

# figure out which carers and driving carers we have on those days
carers_exploded = carers_df.explode("Available Working Days")
drivers_exploded = carers_exploded.loc[carers_exploded["Driver"].astype(int) == 1]

Cd = (
    carers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

CdD = (
    drivers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

Vd = visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
Vpd = pair_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
Vsd = single_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()

Pd = {}
Sd = {}
for d in Days:
    pairShare = Vpd[d]/Vd[d]
    Pd[d] = int(round(pairShare * len(Cd[d]) / (1 + pairShare)))
    Pd[d] = max(0, min(Pd[d], len(Cd[d]) // 2))
    Sd[d] = len(Cd[d]) - 2*Pd[d]

K = 40

explode_singles = single_visits_df.explode("Known Carers")

sid = explode_singles.groupby(["Known Carers", "Visit Date"])["Visit Duration"].sum().to_dict()

Fijd = defaultdict(lambda: pd.Timedelta(0))
fijd = defaultdict(lambda: pd.Timedelta(0))

pair_visits_df.columns = pair_visits_df.columns.str.strip().str.replace(" ", "_").str.lower()

for row in pair_visits_df.itertuples(index=False):
    d = getattr(row, "visit_date")
    duration = getattr(row, "visit_duration")
    known = set(getattr(row, "known_carers"))
    
    Cd_set = set(Cd[d])
    CdD_set = set(CdD[d])
    
    known = set(map(int, known))
    Cd_set = set(map(int, Cd[d]))
    CdD_set = set(map(int, CdD[d]))
    
    known_in_Cd = known & Cd_set
    known_in_CdD = known & CdD_set
    
    # identify pairs where both are known and get durations
    for i in known_in_CdD:
        for j in known_in_Cd:
            if i != j and j in known:
                Fijd[(i, j, d)] += duration
    
    # identify pairs where either are known and get durations (double count ANDs)
    for i in CdD_set:
        for j in Cd_set:
            if (i in known) or (j in known):
                fijd[(i, j, d)] += duration
       

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

# now find each carer's locality and furthest neighbor in all other localities
ril = {}
li = {}
for c in carers:
    carerHome = localityMap[nearestClient[c]]
    for l in set(clientLocalities):
        li[(c,l)] = int(l == carerHome)
        members = [n for n in D_clients.index if localityMap[n] == l]
        if l == carerHome:
            ril[(c,l)] = 0
        else:
            ril[(c,l)] = max(D.loc[c,n] for n in members)

# finally break down visit minutes by locality and pair/single status        
pair_visits_df["locality"] = pair_visits_df["client_id"].map(localityMap)
single_visits_df["Locality"] = single_visits_df["Client ID"].map(localityMap)

Vpld = (
        pair_visits_df.groupby(["locality", "visit_date"])["visit_duration"]
        .sum()
        .to_dict()
)

Vsld = (
        single_visits_df.groupby(["Locality", "Visit Date"])["Visit Duration"]
        .sum()
        .to_dict()
)

# convert all timedeltas to floats/ints
Fijd = {k: v.total_seconds()/60 for k, v in Fijd.items()}
fijd = {k: v.total_seconds()/60 for k, v in fijd.items()}
sid = {k: v.total_seconds()/60 for k, v in sid.items()}
Vd = {k: v.total_seconds()/60 for k, v in Vd.items()}
Vpd = {k: v.total_seconds()/60 for k, v in Vpd.items()}
Vsd = {k: v.total_seconds()/60 for k, v in Vsd.items()}
Vpld = {k: v.total_seconds()/60 for k, v in Vpld.items()}
Vsld = {k: v.total_seconds()/60 for k, v in Vsld.items()}

# export clients' locality assignments so that we have some idea what any of this means
clientAssignment = pd.DataFrame(list(localityMap.items()), columns=["Client ID", "Locality"])
clientAssignment.to_csv(currParent.parent / "Home HealthCare Data" / "locality_assignments.csv", index=False)

"""
EXPORT SETS AND PARAMETERS AS A PICKLE
"""
precomp = {
        "D": Days,
        "Cd": Cd,
        "CdD": CdD,
        "L": L,
        "dij": D,
        "ril": ril,
        "Fijd": dict(Fijd),
        "fijd": dict(fijd),
        "sid": sid,
        "Vpd": Vpd,
        "Vsd": Vsd,
        "Vlpd": Vpld,
        "Vlsd": Vsld,
        "Pd": Pd,
        "Sd": Sd,
        "li": li,
        "K": K
}

with open(currParent.parent / "Home HealthCare Data" / "inputs.pkl", "wb") as f:
    pickle.dump(precomp, f)