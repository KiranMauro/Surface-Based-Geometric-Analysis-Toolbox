import numpy as np
import nibabel as nib
from scipy.sparse import lil_matrix
from sklearn.cluster import SpectralClustering
from data.io import save_label


def run(project):
    # SETTINGS
    model_type = project.model

    settings = project.settings.processing
    min_cluster_area = settings.min_cluster_area
    min_split_area = settings.min_split_area
    max_dist = settings.split_max_dist
    alpha = settings.alpha
    min_edge_weight = settings.min_edge_weight
    split_similarity_threshold = settings.split_similarity_threshold
    n_clusters_to_test = settings.n_clusters


   
    # PATHS
    parameter_map_dir = project.parameter_map_dir
    split_dir = project.split_dir
    split_dir.mkdir(parents=True, exist_ok=True)

   
    # parameter MAP
    if model_type == "N":
        parameter_name = "N"

    elif model_type == "VFM":
        parameter_name = "polar_angle"

    map_lh = nib.load(
        parameter_map_dir
        / f"lh.{parameter_name}.mgh"
    ).get_fdata().squeeze()

    map_rh = nib.load(
        parameter_map_dir
        / f"rh.{parameter_name}.mgh"
    ).get_fdata().squeeze()

   
    # LOAD CLUSTERS
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters
    
    # HELPERS
    def normalize_vectors(v):

        norms = np.linalg.norm(
            v,
            axis=1,
            keepdims=True
        )

        out = np.full_like(v, np.nan)
        good = (norms.squeeze() > 1e-8)
        out[good] = (v[good]/norms[good])

        return out


    def direction_similarity(v1, v2):

        if np.any(np.isnan(v1)):
            return 0.0

        if np.any(np.isnan(v2)):
            return 0.0

        c = np.clipnp.dot(v1, v2), -1.0, 1.0

        return abs(c)

    def build_affinity_matrix(
        mean_vectors,
        map_values,
        roi_dist
    ):

        n = len(map_values)

        A = lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = roi_dist[i, j]
                if np.isnan(d):
                    continue
                if d > max_dist:
                    continue
                dir_sim = (
                    direction_similarity(
                        mean_vectors[i],
                        mean_vectors[j]
                    )
                )

                spatial_sim = np.exp(-(d / max_dist))
                w = ((dir_sim ** alpha))

                if w < min_edge_weight:
                    continue

                A[i, j] = w
                A[j, i] = w

        return A.tocsr()

    def evaluate_split(labels, affinity):
        rows, cols = affinity.nonzero()
        cross_weights = []

        for i, j in zip(rows, cols):
            if labels[i] != labels[j]:
                cross_weights.append(affinity[i, j])

        if len(cross_weights) == 0:
            return 0.0

        return np.mean(cross_weights)


    # SPLIT SUGGESTION
    def suggest_split(cluster, map_data):

        geo = project.geo[cluster.hemi]
        coords = geo.verts
        verts = np.asarray(cluster.valid_vertices)

        if cluster.metadata.get("surface_area") is  None:
            cluster.metadata["surface_area"] = geo.surface_area(cluster.valid_vertices)
            
        if cluster.metadata["surface_area"] < min_cluster_area:
            print(f"  Too small ({cluster.metadata['surface_area']:.1f} mm²)")
            return None

        map_values = map_data[verts]
        mean_vectors = (cluster.directions["topographic"])
        mean_vectors = (normalize_vectors(mean_vectors))
        valid = (~np.isnan(map_values)&~np.isnan(mean_vectors).any(axis=1))
        verts = verts[valid]
        map_values = (map_values[valid])
        mean_vectors = (mean_vectors[valid])
        roi_dist = (geo.roi_distance_matrix(verts))
        affinity = (build_affinity_matrix( mean_vectors, map_values, roi_dist))
        best_labels = None
        best_score = np.inf

        for k in n_clusters_to_test:
            if cluster.metadata["surface_area"] < (k * min_split_area):
                continue
            try:
                sc = (
                    SpectralClustering(
                        n_clusters=k,
                        affinity="precomputed",
                        assign_labels="kmeans",
                        random_state=0
                    )
                )

                labels = (sc.fit_predict(affinity))

            except Exception:
                continue

            valid_split = True
            for split_id in np.unique(labels):
                split_vertices = verts[labels == split_id]
                area = geo.surface_area(split_vertices)

                if area < min_split_area:
                    valid_split = False
                    break

            if not valid_split:
                continue

            score = evaluate_split(labels, affinity)

            if score < best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            return None

        if best_score > split_similarity_threshold:
            return None

        cluster_dir = (split_dir / f"{cluster.hemi}_cluster_{cluster.id:04d}")

        cluster_dir.mkdir(exist_ok=True)
        suggestions = []
        unique_labels = np.unique(best_labels)

        for split_id in unique_labels:
            split_vertices = verts[best_labels == split_id]

            split_label = (cluster_dir / f"split_{split_id:02d}.label")
            save_label(split_vertices, coords, split_label)
            suggestions.append(
                {
                    "split_id": int(split_id),
                    "vertices": split_vertices,
                    "label_file":str(split_label)
                }
            )

        return suggestions
    
    # MAIN LOOP
    n_suggestions = 0
    for cluster in [clusters[6]]:
        print(f"Cluster {cluster.id}")

        if cluster.hemi == "lh":
            map_data = map_lh
        else:
            map_data = map_rh

        cluster.metadata.pop("suggested_splits", None)
        suggestions = suggest_split(cluster, map_data)

        if suggestions is None:
            continue

        cluster.metadata["suggested_splits"] = suggestions
        n_suggestions += 1

        print(f"  Suggested {len(suggestions)} splits")

    
    # SAVE
    project.save_clusters(cluster_set)

    print()
    print(f"Clusters with suggestions: "f"{n_suggestions}")
