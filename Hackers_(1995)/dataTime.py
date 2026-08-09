# pandas = our workhorse for tables (like a df is basically a spreadsheet w/ superpowers)
import pandas as pd
# numpy = fast array math, we mostly just use it for arrays of ids here
import numpy as np
# Path = os-agnostic way to build file paths (works on windows/mac/linux w/o string hacking)
from pathlib import Path
# defaultdict = a dict that auto-fills a default value instead of throwing a KeyError
from collections import defaultdict
# hierarchical clustering tools -- we use these later to group clients into localities
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
# pickle = python's native way to serialize (save/load) objects to disk as-is
import pickle
# just for timing the run
from datetime import datetime

"""
HELPERS AND FILES
"""
# start the clock, we print the total runtime at the very end of the script
run_start = datetime.now()

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
    # pd.isna catches missing/NaN cells, so blanks don't blow up the parse below
    if pd.isna(x):
        return None
    # force x to a string, strip whitespace, split on ":" -> ["h", "mm"]
    # map(int, ...) runs int() over both pieces at once, then we unpack straight into h, m
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
# "with open(...) as f" opens the file and auto-closes it when we're done w/ the block,
# even if something errors out partway through -- no manual f.close() needed
with open(carersFile) as f:
    # next(f) pulls the very first line off the file iterator (the header),
    # so the loop below starts on row 2 -- get the header row outta the way
    header = next(f).strip().split(";")
    
    # go row by row through this csv (a file object is iterable line-by-line in python)
    for line in f:
        # extract each value
        # [p.strip() for p in line.split(";")] is a list comprehension: shorthand for
        # "build a list by running p.strip() over every p in line.split(';')"
        parts = [p.strip() for p in line.split(";")]
        # first five cols are consistent w/ csv structure
        fixedLine = parts[:5]
        # \exists a variable length tail in last col(s)
        # same list comprehension idea, but filtered to drop empty strings
        tail = [p for p in parts[5:] if p != ""]
        
        # make a fixed row and add to running data list
        # list(map(str, tail)) just re-stringifies each tail entry (cheap safety net)
        # the "if tail else []" is a one-line if/else (a ternary) -- empty list when no tail
        row = fixedLine + [list(map(str, tail)) if tail else []]
        carersData.append(row)

# create a dataframe from the extracted data and cast ids as ints
carers_df = pd.DataFrame(carersData, columns=header, index=None)
carers_df["Carer ID"] = carers_df["Carer ID"].astype(int)

# convert the carer start/end times to a workable type
# (going from raw "h:mm" strings -> pandas Timedelta, so we can compare/subtract times later)
string_start = carers_df["Shift Start Time"].astype(str).str.strip()
string_end = carers_df["Shift End Time"].astype(str).str.strip()

# str.split(":", expand=True) splits every row's string on ":" and spreads the
# pieces into their own columns -- so hms[0] is all the hour strings, hms[1] all the minutes
hms = string_start.str.split(":", expand = True)
hme = string_end.str.split(":", expand = True)

# rebuild each time as hours-as-timedelta + minutes-as-timedelta
carers_df["start_min"] = (
    pd.to_timedelta(hms[0].astype(int), unit ="h") +
    pd.to_timedelta(hms[1].astype(int), unit ="m")
    )

carers_df["end_min"] = (
    pd.to_timedelta(hme[0].astype(int), unit ="h") +
    pd.to_timedelta(hme[1].astype(int), unit ="m")
    )

# make a lookup table for carer shift start/end
# set_index(...)[col].to_dict() turns two columns into a {Carer ID: value} dict,
# so elsewhere we can just do starts[carer_id] instead of filtering the dataframe every time
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
        # same idea as the carer file, but the tail here is carer IDs so we cast to int
        row = fixedLine + [list(map(int, tail)) if tail else []]
        clientsData.append(row)

# create a dataframe from the relevant extracted data and cast ids as ints
# (naming the columns manually here since this file's header didn't survive the ragged parse)
clients_df = pd.DataFrame(clientsData, 
                          columns=["Client ID", 
                                   "Gender Preference", 
                                   "Known Carers"], 
                          index=None)
clients_df["Client ID"] = clients_df["Client ID"].astype(int)

# create a dataframe from the visit data file
# (this one's a well-behaved csv so pd.read_csv handles it directly, no manual parsing needed)
# index_col="Visit ID" makes that column the row index instead of a regular data column
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

# standardize naming convention across frames (so "Client ID" matches clients_df/carers_df)
visits_df = visits_df.rename(columns={"ClientID": "Client ID"})

# normalize time windows to workable type -- same h:mm -> timedelta trick as the carer shifts
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
# (boolean indexing: the condition inside [...] builds a True/False mask, one per row,
# and only the True rows survive -- no explicit loop needed)
pair_visits_df = visits_df[visits_df["Number of Carers"] == 2]
single_visits_df = visits_df[visits_df["Number of Carers"] == 1]

# add information on known caregivers per visit
# merge = a sql-style join: glue clients_df onto each visits frame by matching Client ID
pair_visits_df = pair_visits_df.merge(clients_df, on="Client ID")
single_visits_df = single_visits_df.merge(clients_df, on = "Client ID")

# create a dataframe from the travel data file
travel_df = pd.read_csv(travelFile, sep=";", index_col=0)
# travel time matrix ends up with an empty column
# dropna(axis=1, how="all") drops any COLUMN that is entirely empty (axis=1 means columns)
travel_df = travel_df.dropna(axis=1, how="all")

# convert data to ints for later
# .map(to_minutes) runs our to_minutes() helper (defined up top) over every single cell
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
# .explode("Available Working Days") turns a single row w/ a LIST of days
# into one row PER day (duplicating the rest of that carer's info each time) --
# so a carer available mon/wed/fri becomes 3 separate rows, one per day
carers_exploded = carers_df.explode("Available Working Days")
drivers_exploded = carers_exploded.loc[
    carers_exploded["Driver"].astype(int) == 1]

# grab every carer id available daily
# groupby(...)[...] buckets rows by day, .apply(list) collapses each bucket's
# Carer IDs into a python list, .to_dict() turns the whole thing into {day: [ids...]}
Cd = (
    carers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

# grab every driver id available daily (same pattern, just on the driver-only subset)
CdD = (
    drivers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

"""
BUILD ALL FEASIBLE PART-TIME COUPLES
-- the idea: two part-time carers whose shifts don't overlap can be glued together
   into one "frankenstein" full-shift unit (see BUILD FEASIBLE UNITS PER DAY below)
"""
# initialize couple dicts
# {d: [] for d in Days} is a dict comprehension -- same shorthand idea as a list
# comprehension, just building a dict {day: empty_list} for every day up front
allCouples = {d: [] for d in Days}
driveCouples = {d: [] for d in Days}

for d in Days:
    # get the lists of carers and drivers today
    carers_d = list(Cd[d])
    CdD_set = set(CdD[d])
    
    # check each carer
    # enumerate(carers_d) hands back (index, carer) pairs so we know idx below
    for idx, c in enumerate(carers_d):
        # check each remaining carer
        # carers_d[idx+1:] = everything after c in the list, so we only ever
        # compare each pair once (c vs c_prime) instead of twice in both orders
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
-- a "unit" is anything that can be dispatched to a visit: one carer, a coupled
   pair of part-timers acting as one solo carer, or a real two-person pair.
   "members" holds the underlying carer id(s), "type" is solo/pair, "couples"
   flags whether it's a frankenstein-glued unit, "driver" flags if it can drive.
"""

feasibleUnits_d = {d: [] for d in Days}

for d in Days:
    # seen_today guards against building the same unit twice under a different id
    # (e.g. if the same two carers show up as a couple from two different loops below)
    # -- a set gives us O(1) "have I seen this key before" lookups
    seen_today = set()
    # solo real caregivers
    for c in Cd[d]:
        # members is a 1-tuple here so the id/key logic below works the same
        # regardless of whether a unit has 1, 2, or 3 people in it
        members = (c,)
        # build a human-readable id like "1001_s" (carer 1001, solo)
        # "_".join(map(str, members)) turns the tuple of ids into an underscore-joined string
        unitID = "_".join(map(str, members)) + "_s"
        # key is what actually gets deduped on: sorted members + type,
        # so member ORDER doesn't matter for detecting a duplicate
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
    
    # pair real driver + real caregiver
    # a driver i and a caregiver j can pair up if one of their shifts fully
    # contains the other's
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
    
    # pair driver coupling + real caregiver
    # a frankenstein driving couple (a,b) can pair w/ a third caregiver j,
    # as long as j isn't already one of the two people making up the couple
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
    
    # pair real driver + coupling
    # mirror image of the block above: a real driver i teamed w/ a (non-driving) couple
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

# split drivers off in their own set now for easier downstream indexing stuff
# nested comprehension: for each day d, build a list of unit ids by filtering
# feasibleUnits_d[d] down to just the units flagged as drivers
driverUnits = {
    d: [u["id"] for u in feasibleUnits_d[d] if u["driver"]]
    for d in Days
}

"""
GET VISIT MINUTES AND SHARES
"""
# count the number of visit minutes daily
# groupby("Visit Date")[...].sum() adds up Visit Duration per day, .to_dict() -> {day: total}
Vd = visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
# split by pair and solo
Vpd = pair_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()
Vsd = single_visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()

# daily pair/solo share initialization
pairShare = {}
for d in Days:
    # the share of pair/solo minutes today
    pairShare[d] = Vpd[d]/Vd[d]
    # print(f"{d} share of visit minutes requiring pair: {pairShare}%")

# default max travel for pair assignment (temporary? think further later)
K = 40

"""
BUILD sud: SOLO VISIT MINUTES PER UNIT (SHIFT-AWARE)
-- sud[(unit_id, day)] = total minutes of single-carer visits that unit COULD cover
   that day, i.e. visits where the unit is both known to the client and on shift
"""
# standardize column labels
# .str.strip/.replace/.lower chained together turns "Visit Date" -> "visit_date" etc,
# so itertuples() below gives us clean attribute names like row.visit_date
single_visits_df.columns = single_visits_df.columns.str.strip(
    ).str.replace(" ", "_").str.lower()

# initialize carer-day -> duration dictionary
# defaultdict(lambda: pd.Timedelta(0)) means: reading a key that doesn't exist yet
# just hands back a zero Timedelta instead of raising an error, so we can do
# sud[key] += duration straightaway w/o checking "is this key already there?" first
sud = defaultdict(lambda: pd.Timedelta(0))

# check every single carer visit
# itertuples(index=False) walks the dataframe row-by-row as lightweight named tuples --
# much faster than .iterrows(), and row.colname reads like a struct field
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
        # plain solo unit: just the one carer, check they're known + on shift
        if not u["couples"]:
            i = u["members"][0]
            if i in known and within_shift(starts[i], ends[i], v_start, v_end):
                sud[(u["id"],d)] += duration
                
        # frankenstein coupled unit: either half being known + on shift is enough
        # to cover the visit, since only one of them actually shows up
        else:
            i, j = u["members"]
            if i in known and within_shift(starts[i], ends[i], v_start, v_end):
                sud[(u["id"],d)] += duration
            elif j in known and within_shift(starts[j], ends[j], v_start, v_end):
                sud[(u["id"],d)] += duration
        

"""
BUILD Fijd AND fijd (SHIFT-AWARE, WITH COUPLES)
-- same idea as sud, but for visits that NEED two carers at once. Fud is the strict
   version (BOTH people on the visit must be known to the client), fud is the loose
   version (at LEAST ONE known is enough) -- these end up being two different
   feasibility thresholds downstream
"""
# standardize column names
pair_visits_df.columns = pair_visits_df.columns.str.strip(
    ).str.replace(" ", "_").str.lower()

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
        
        # plain two-person pair (no frankenstein coupling involved)
        if not u["couples"]:
            i, j = u["members"]
            
            i_active = within_shift(starts[i], ends[i], v_start, v_end)
            j_active = within_shift(starts[j], ends[j], v_start, v_end)
            
            # check both are currently active
            if not (i_active and j_active):
                continue
            
            # check if each caregiver is known
            i_known = i in known
            j_known = j in known
            
            # strict count: both must be known
            if i_known and j_known:
                Fud[(u["id"],d)] += duration
            # loose count: either being known is enough
            if i_known or j_known:
                fud[(u["id"],d)] += duration
        
        # three-person unit (a driving couple + a third caregiver) --
        # here we have to work out which TWO of the three are actually on shift
        # together, since the "couple" half only ever sends one person at a time
        if u["couples"]:
            i, j, k = u["members"]
            
            i_active = within_shift(starts[i], ends[i], v_start, v_end)
            j_active = within_shift(starts[j], ends[j], v_start, v_end)
            k_active = within_shift(starts[k], ends[k], v_start, v_end)
            
            i_known = i in known
            j_known = j in known
            k_known = k in known
        
            # check each of the 3 possible on-shift-together pairings in turn
            # (i,j) / (i,k) / (j,k) -- elif means only the first match that's
            # active counts, assuming at most one pairing is on shift at once
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
-- travel_df only has real travel times between carers and clients (not 
   carer->carer as needed). D below is a full square distance matrix over 
    EVERY node (carers + clients) so we can cluster clients into localities
   and measure distances between any two units later on.
"""
# first augment distance matrix to include carer -> carer
allNodes = list(carers) + list(clients)
n = len(allNodes)

# start w/ an n x n grid of zeros, labeled by node id on both axes, and fill it in below
D = pd.DataFrame(np.zeros((n,n)), index= allNodes, columns= allNodes)

# keep well-defined client->client and carer->client distances
# (setting both D.loc[i,j] and D.loc[j,i] makes sure the matrix is symmetric)
for i in carers:
    for j in clients:
        D.loc[i,j] = travel_df.loc[i,j]
        D.loc[j,i] = travel_df.loc[i,j]
        
for i in clients:
    for j in clients:
        D.loc[i,j] = travel_df.loc[i,j]
        D.loc[j,i] = travel_df.loc[i,j]

# next dummy carer->carer by going to nearest clients as middlemen
# (there's no real travel time between two carers, so we estimate one:
#  find each carer's single nearest client, then route carer->their client->
#  the other carer's client->other carer, as a stand-in distance)
# travel_df.loc[c] grabs carer c's whole row (distance to every client),
# .idxmin() returns the COLUMN LABEL (client id) of the smallest value, not the value itself
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
# (grouping clients that are close together into numLocalities neighborhoods,
# so we can consider travel caregiver(s) must do under assignment scenarios
numLocalities = 8
D_clients = D.loc[clients, clients]

# squareform converts our full symmetric client x client matrix into scipy's
# expected "condensed" 1-D form (just the upper triangle, no redundant duplicate values)
condensed = squareform(D_clients.values)
# linkage does the actual clustering: repeatedly merges the two closest clusters
# together until everything's in one tree; method="complete" measures cluster
# distance by the FARTHEST pair between them (keeps clusters tight/compact)
Z = linkage(condensed, method="complete")
# fcluster cuts that tree into exactly numLocalities groups and returns, for each
# client (in the same order as D_clients.index), which group number it landed in
clientLocalities = fcluster(Z, numLocalities, criterion="maxclust")

# zip pairs up client ids w/ their cluster number 1-to-1, dict(...) turns those
# pairs into a lookup {client_id: locality_number}
localityMap = dict(zip(D_clients.index, clientLocalities))
# set(...) dedupes down to the unique locality numbers, then we just clean the type
L = [int(x) for x in set(clientLocalities)]

# now find each units's locality, furthest neighbor in all other localities
# and intra-unit distances
# lu[(unit, locality)] = 1 if that's the unit's "home" locality, else 0
# rul[(unit, locality)] = worst-case (furthest) distance from the unit to that
#                         locality, used as a conservative travel-time bound
# du[unit] = total distance between all of a multi-person unit's own members
#            (0 for solo units, since there's nothing to sum over)
rul = {}
lu = {}
du = defaultdict(float)

for d in Days:
    for u in feasibleUnits_d[d]:
        u_id = u["id"]
        members = u["members"]
        
        # choose representative (i.e. nearest) client
        # key=lambda a: ... tells min() to compare members by "distance to their
        # nearest client" rather than by the raw member id -- so rep is whichever
        # member of the unit lives closest to some client overall
        rep = min(
            members,
            key=lambda a: travel_df.loc[a].astype(float).min()
        )
        
        # that rep's nearest client's locality becomes the whole unit's "home base"
        home = localityMap[nearestClient[rep]]
        
        for l in L:
            # flag: is l this unit's home locality? int(True)/int(False) -> 1/0
            lu[(u_id, l)] = int(l == home)
            members_in_l = [n for n in clients if localityMap[n] == l]
            
            if lu[(u_id, l)] == 1:
                # no travel needed to reach your own home locality
                rul[(u_id, l)] = 0
            else:
                # worst case: take the single furthest client in locality l from
                # ANY member of the unit, then the worst-case across all members too
                # (nested max: inner max is per-member worst distance into l,
                #  outer max picks the worst of those across all members)
                rul[(u_id, l)] = max(
                    max(D.loc[a, n] for n in members_in_l)
                    for a in members
                )
        # sum the distance between every distinct pair of members in this unit
        # (range(i+1, len(members)) again avoids double-counting a pair both ways)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                du[u_id] += D.loc[members[i], members[j]]

# finally break down visit minutes by locality and pair/single status        
# .map(localityMap) looks up each row's client id in the dict and stamps the
# matching locality number onto a new "locality" column
pair_visits_df["locality"] = pair_visits_df["client_id"].map(localityMap)
single_visits_df["locality"] = single_visits_df["client_id"].map(localityMap)

# same groupby-and-sum pattern as Vpd/Vsd above, just now grouping on TWO keys
# at once (locality AND date), so the dict comes back keyed by (locality, date) tuples
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
# everything above was built up as pandas Timedelta objects (handy for date math),
# but downstream proesses want plain numbers -- .total_seconds()/60 converts a 
# Timedelta to minutes as a float. {k: ... for k, v in X.items()} is a dict 
# comprehension: rebuild the same dict, same keys, but w/ every value run 
# through the conversion
Fud = {k: v.total_seconds()/60 for k, v in Fud.items()}
fud = {k: v.total_seconds()/60 for k, v in fud.items()}
sud = {k: v.total_seconds()/60 for k, v in sud.items()}
Vd = {k: v.total_seconds()/60 for k, v in Vd.items()}
Vpd = {k: v.total_seconds()/60 for k, v in Vpd.items()}
Vsd = {k: v.total_seconds()/60 for k, v in Vsd.items()}
Vpld = {k: v.total_seconds()/60 for k, v in Vpld.items()}
Vsld = {k: v.total_seconds()/60 for k, v in Vsld.items()}



# export clients' locality assignments if using these localities
# clientAssignment = pd.DataFrame(
#     list(localityMap.items()), 
#     columns=["Client ID", "Locality"])
# clientAssignment.to_csv(
#     currParent.parent / "Home HealthCare Data" / "locality_assignments.csv", 
#     index=False)


"""
EXPORT SETS AND PARAMETERS AS A PICKLE
-- bundle every set/dict we've built up above into one dict, then dump the whole
   thing to disk in one shot. whatever reads inputs.pkl later (e.g. the solver
   script) gets these exact python objects back out, no re-parsing needed.
"""
precomp = {
        "D": Days,
        "Cd": Cd,
        "units_d": feasibleUnits_d,
        "drive_d": driverUnits,
        "allCouples": allCouples,
        "driveCouples": driveCouples,
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
        "pairShare": pairShare,
        "lu": lu,
        "K": K
}

# "wb" = write, binary mode -- pickle files aren't human-readable text, so this
# has to be binary, not the plain "w" we might use for a csv/text file
with open(currParent.parent / "Home HealthCare Data" / "inputs.pkl", "wb") as f:
    pickle.dump(precomp, f)
    
run_end = datetime.now()
# f-string: the f"..." prefix lets us drop {expressions} straight into the string,
# they get evaluated and stringified in place -- no manual + concatenation needed
print(f"Total Preprocessing Runtime: {run_end - run_start}")