from core.graph import GraphGeodesics
from core.correspondence import compute_cluster_matches
from data.structures import CorrespondenceGroup
from data.io import save_clusters
import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def run(projects):

    fsaverage_dir = (projects[0].data_dir / "fsaverage")
    output_dir = projects[0].project_root / "data" / "group_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    geo_fsavg_lh = GraphGeodesics(fsaverage_dir / "surf" / "lh.white")
    geo_fsavg_rh = GraphGeodesics(fsaverage_dir / "surf" / "rh.white")

    all_clusters = {}

    for project in projects:

        cluster_set = project.cluster_set
        for hemi in ["lh", "rh"]:
            all_clusters[(project.subject, hemi)] = [c for c in cluster_set.final_clusters if c.hemi == hemi]

    pairwise_matches = {}

    for hemi in ["lh", "rh"]:
        if hemi == "lh":
            geo = geo_fsavg_lh
        else:
            geo = geo_fsavg_rh

        for i, project1 in enumerate(projects):

            for project2 in projects[i + 1:]:
                clusters1 = all_clusters[(project1.subject, hemi)]
                clusters2 = all_clusters[(project2.subject, hemi)]

                matches = compute_cluster_matches(clusters1, clusters2, geo)
                pairwise_matches[(project1.subject, project2.subject, hemi)] = matches

    def build_similarity_lookup(pairwise_matches, hemi):

        lookup = {}
        for (s1, s2, h), matches in pairwise_matches.items():

            if h != hemi:
                continue

            for m in matches:

                c1 = m["cluster1"]
                c2 = m["cluster2"]

                lookup[(c1, c2)] = m["similarity"]
                lookup[(c2, c1)] = m["similarity"]

        return lookup
    
   
    all_groups = {}

    reference = projects[0].subject

    for hemi in ["lh", "rh"]:

        similarity_lookup = build_similarity_lookup(pairwise_matches, hemi)

        groups = []
        for cluster in all_clusters[(reference, hemi)]:
            group = CorrespondenceGroup(id=len(groups) + 1)
            group.members[reference] = cluster
            groups.append(group)

        for project in projects[1:]:
            subject = project.subject
            subject_clusters = all_clusters[(subject, hemi)]
            cost = np.zeros((len(groups), len(subject_clusters)))

            for i, group in enumerate(groups):
                for j, cluster in enumerate(subject_clusters):
                    scores = [similarity_lookup.get((member, cluster), 0) for member in group.members.values()]
                    cost[i, j] = -np.mean(scores)

            rows, cols = linear_sum_assignment(cost)
            print(f"\nAssignments for {subject} ({hemi})")
            for r, c in zip(rows, cols):
                print(
                    f"Group {groups[r].id} <- "
                    f"Cluster {subject_clusters[c].id} "
                    f"(cost={cost[r, c]:.3f})"
                )

                groups[r].members[subject] = subject_clusters[c]

        for group in groups:
            for cluster in group.members.values():
                cluster.group_id = group.id

        all_groups[hemi] = groups
        print(f"\n{hemi.upper()} groups")

        for group in groups:
            print(group.id, sorted((c.subject, c.id) for c in group.members.values()))

    print(f"\nReference {reference} ({hemi})")
    for i, group in enumerate(groups):
        for j, cluster in enumerate(subject_clusters):

            print(
                f"{subject} "
                f"cluster {cluster.id} "
                f"-> group {group.id} "
                f"{-cost[i,j]:.3f}"
            )

    for g in groups:
        print(g.id, g.members[reference].id)

    coords = {
        "lh": nib.freesurfer.read_geometry(fsaverage_dir / "surf" / "lh.white")[0],
        "rh": nib.freesurfer.read_geometry(fsaverage_dir / "surf" / "rh.white")[0],
    }

    group_centroids = {"lh": [], "rh": []}

    for hemi in ["lh", "rh"]:
        for group in all_groups[hemi]:
            xyz = np.array([
                coords[hemi][cluster.fsaverage_centroid]
                for cluster in group.members.values()
            ])

            group_centroids[hemi].append(xyz.mean(axis=0))

    lh_centroids = np.asarray(group_centroids["lh"])
    rh_reflected = np.asarray(group_centroids["rh"]).copy()

    rh_reflected[:, 0] *= -1


    cost = cdist(lh_centroids, rh_reflected)
    lh_idx, rh_idx = linear_sum_assignment(cost)
    for i, j in zip(lh_idx, rh_idx):
        print(
            all_groups["lh"][i].id,
            "<->",
            all_groups["rh"][j].id,
            cost[i, j]
        )

    bilateral_groups = []

    for gid, (i, j) in enumerate(zip(lh_idx, rh_idx), start=1):
        group = CorrespondenceGroup(id=gid)

        for cluster in all_groups["lh"][i].members.values():
            group.members[(cluster.subject, "lh")] = cluster
            cluster.group_id = gid

        for cluster in all_groups["rh"][j].members.values():
            group.members[(cluster.subject, "rh")] = cluster
            cluster.group_id = gid

        bilateral_groups.append(group)

    save_clusters({"correspondence": bilateral_groups}, output_dir / "correspondence_groups.npy")