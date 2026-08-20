"""
pipeline_illustration.py

Generates matplotlib figures illustrating the HHRSP preprocessing +
optimization pipeline (dataTime.py -> modelTime.py), for use in a report.

This is a standalone, self-contained companion script -- it does NOT import
from or run dataTime.py/modelTime.py. It builds a small synthetic example
(30 clients, 5 caregivers) with the same *shape* as the real data, and runs
the real scipy clustering call on it so the dendrogram/clusters shown are
genuinely computed, not hand-drawn. Everything past that (the example
Monday assignment in particular) is illustrative -- swap it for get_sol()'s
real output whenever you've got an actual solved model to show.

Run directly to write all five figures into ./figures/:
    python pipeline_illustration.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path

PALETTE = {
    "locality1": "#7F77DD",
    "locality2": "#1D9E75",
    "locality3": "#D85A30",
    "caregiver": "#BA7517",
    "muted": "#9B9B96",
    "line": "#5F5E5A",
}
LOCALITY_COLORS = [PALETTE["locality1"], PALETTE["locality2"], PALETTE["locality3"]]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.edgecolor": "#CFCFCA",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

currDir = Path(__file__).resolve().parent
writeFolder = currDir.parent / "creature_report"

def generate_example_data(seed=42, n_per_cluster=10):
    rng = np.random.default_rng(seed)
    centers = {"A": (170, 150), "B": (500, 140), "C": (330, 340)}

    clients = {}
    cid = 1
    for _, (cx, cy) in centers.items():
        for _ in range(n_per_cluster):
            x = cx + rng.integers(-70, 71)
            y = cy + rng.integers(-55, 56)
            clients[f"cl{cid}"] = (float(x), float(y))
            cid += 1

    carers = {
        "c1": (280.0, 120.0),
        "c2": (420.0, 290.0),
        "c3": (120.0, 350.0),
        "c4": (560.0, 330.0),
        "c5": (340.0, 220.0),
    }
    return clients, carers


def carer_shifts():
    return {
        "c1": (7, 22),
        "c2": (7, 22),
        "c3": (8, 14),
        "c4": (15, 21),
        "c5": (7, 22),
    }


def example_monday_assignment():
    return [
        {"members": ("c1",), "locality": 1},
        {"members": ("c3", "c4"), "locality": 3},
        {"members": ("c2", "c5"), "locality": 2},
    ]

def _draw_dashed(ax, p, q, color=PALETTE["caregiver"], lw=1):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, linestyle="--")


def _draw_solid(ax, p, q, color=PALETTE["line"], lw=1):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw)


def _style_map_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# STAGE 1: raw scatter
# ---------------------------------------------------------------------------
def plot_raw_scatter(clients, carers, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    xs, ys = zip(*clients.values())
    ax.scatter(xs, ys, s=24, color=PALETTE["muted"], label="Client")

    cxs, cys = zip(*carers.values())
    ax.scatter(cxs, cys, s=60, marker="s", color=PALETTE["caregiver"], label="Caregiver")

    ax.set_title("Original Client and Caregiver locations")
    ax.legend(loc="lower right", frameon=False)
    _style_map_axes(ax)

    if standalone:
        fig.tight_layout()
        fig.savefig(writeFolder / "01_raw_scatter.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# STAGE 2: unit-building from non-overlapping shifts
# ---------------------------------------------------------------------------
def plot_unit_building(shifts, couple=("c3", "c4")):
    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(shifts.keys())
    for i, name in enumerate(names):
        start, end = shifts[name]
        color = PALETTE["locality3"] if name in couple else PALETTE["muted"]
        ax.barh(i, end - start, left=start, height=0.6, color=color)

    unit_start = min(shifts[c][0] for c in couple)
    unit_end = max(shifts[c][1] for c in couple)
    ax.barh(len(names), unit_end - unit_start, left=unit_start, height=0.6,
            color=PALETTE["caregiver"])

    ax.set_yticks(list(range(len(names))) + [len(names)])
    ax.set_yticklabels([f"c{n[1:]}" for n in names] + ["couple"])
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_title("Example Shift Patterns")

    fig.tight_layout()
    fig.savefig(writeFolder / "02_unit_building.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------------
# STAGE 3: hierarchical clustering into localities, dendrogram alongside
# ---------------------------------------------------------------------------
def plot_clustering(clients, carers, num_localities=3):
    client_ids = list(clients.keys())
    coords = np.array([clients[c] for c in client_ids])

    condensed = pdist(coords)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, num_localities, criterion="maxclust")

    fig, (ax_map, ax_dendro) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # --- left: clustered scatter ---
    for k in sorted(set(labels)):
        pts = coords[labels == k]
        ax_map.scatter(pts[:, 0], pts[:, 1], s=28,
                        color=LOCALITY_COLORS[(k - 1) % len(LOCALITY_COLORS)],
                        label=f"locality {k}")
    cxs, cys = zip(*carers.values())
    ax_map.scatter(cxs, cys, s=60, marker="s", color=PALETTE["caregiver"], label="Caregiver")
    ax_map.set_title("Localities Derived From Hierarchical Clustering")
    ax_map.legend(loc="lower right", frameon=False, fontsize=8)
    _style_map_axes(ax_map)

    n_leaves = len(client_ids)
    cut_row = n_leaves - num_localities
    color_threshold = Z[cut_row, 2] if 0 <= cut_row < len(Z) else 0

    dendrogram(
        Z,
        ax=ax_dendro,
        no_labels=True,  # 30 leaf labels is unreadable at this size -- turn off
        color_threshold=color_threshold,
        above_threshold_color=PALETTE["muted"],
    )
    ax_dendro.set_title("Example Dendrogram")
    ax_dendro.set_ylabel("Distance")
    ax_dendro.axhline(color_threshold, color=PALETTE["muted"], lw=0.75, linestyle=":")

    fig.tight_layout()
    fig.savefig(writeFolder / "03_clustering_and_dendrogram.png", dpi=200)
    plt.close(fig)
    return labels

# ---------------------------------------------------------------------------
# STAGE 4: derived travel distances (carer-carer chaining, unit->locality reach)
# ---------------------------------------------------------------------------
def plot_travel_distances(clients, carers, labels, client_ids, cluster_of):
    fig, ax = plt.subplots(figsize=(8, 6))

    coords = {cid: np.array(pt) for cid, pt in clients.items()}
    for k in sorted(set(labels)):
        pts = np.array([clients[c] for c in client_ids if cluster_of[c] == k])
        ax.scatter(pts[:, 0], pts[:, 1], s=24, alpha=0.55,
                   color=LOCALITY_COLORS[(k - 1) % len(LOCALITY_COLORS)])
    for pt in carers.values():
        ax.scatter(*pt, s=60, marker="s", color=PALETTE["caregiver"], zorder=3)

    def nearest_client(carer_pt):
        dists = {cid: np.linalg.norm(np.array(carer_pt) - pt) for cid, pt in coords.items()}
        return min(dists, key=dists.get)

    p_c1, p_c4 = carers["c1"], carers["c4"]
    n1, n4 = nearest_client(p_c1), nearest_client(p_c4)
    _draw_dashed(ax, p_c1, coords[n1])
    _draw_solid(ax, coords[n1], coords[n4])
    _draw_dashed(ax, coords[n4], p_c4)

    home_cluster = cluster_of[nearest_client(carers["c5"])]
    foreign_cluster = next(k for k in set(labels) if k != home_cluster)
    foreign_pts = [c for c in client_ids if cluster_of[c] == foreign_cluster]
    farthest = max(foreign_pts, key=lambda c: np.linalg.norm(np.array(carers["c5"]) - coords[c]))
    _draw_solid(ax, carers["c5"], coords[farthest], lw=2)

    ax.set_title("Derived Travel Distances")
    _style_map_axes(ax)

    legend_elems = [
        Line2D([0], [0], color=PALETTE["caregiver"], lw=1, ls="--",
               label="Caregiver - Client"),
        Line2D([0], [0], color=PALETTE["line"], lw=1, label="Caregiver - Caregiver via Nearest Clients"),
        Line2D([0], [0], color=PALETTE["line"], lw=2, label="Caregiver/Unit - Locality"),
    ]
    ax.legend(loc="lower right", handles=legend_elems, frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(writeFolder / "04_travel_distances.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# STAGE 5: final assignment -- unit membership (xud) + locality assignment (zuld)
# ---------------------------------------------------------------------------
def plot_assignment(clients, carers, labels, client_ids, cluster_of, units):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for k in sorted(set(labels)):
        pts = np.array([clients[c] for c in client_ids if cluster_of[c] == k])
        ax.scatter(pts[:, 0], pts[:, 1], s=24, alpha=0.5,
                   color=LOCALITY_COLORS[(k - 1) % len(LOCALITY_COLORS)])

    locality_centroid = {
        k: np.array([clients[c] for c in client_ids if cluster_of[c] == k]).mean(axis=0)
        for k in set(labels)
    }

    for pt in carers.values():
        ax.scatter(*pt, s=60, marker="s", color=PALETTE["caregiver"], zorder=3)

    for u in units:
        members = u["members"]
        loc = u["locality"]
        anchor = carers[members[0]]
        for other in members[1:]:
            _draw_dashed(ax, anchor, carers[other], color=PALETTE["muted"], lw=1.5)
        centroid = locality_centroid[loc]
        _draw_solid(ax, anchor, centroid, color=LOCALITY_COLORS[(loc - 1) % len(LOCALITY_COLORS)], lw=2)

    ax.set_title("Example Candidate Solution")
    _style_map_axes(ax)

    legend_elems = [
        Line2D([0], [0], color=PALETTE["muted"], lw=1.5, ls="--", label="Unit Assigned (xud)"),
        Line2D([0], [0], color=PALETTE["line"], lw=2, label="Locality Assigned to Unit (zuld)"),
    ]
    ax.legend(loc="lower right", handles=legend_elems, frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(writeFolder / "05_assignment.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    clients, carers = generate_example_data()
    client_ids = list(clients.keys())

    plot_raw_scatter(clients, carers)

    labels = plot_clustering(clients, carers, num_localities=3)
    cluster_of = dict(zip(client_ids, labels))

    plot_travel_distances(clients, carers, labels, client_ids, cluster_of)
    plot_unit_building(carer_shifts())
    plot_assignment(clients, carers, labels, client_ids, cluster_of, example_monday_assignment())

    print("donezo")


if __name__ == "__main__":
    main()