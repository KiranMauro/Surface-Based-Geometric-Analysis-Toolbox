import numpy as np
from scipy.stats import circmean
from pycircstat2.hypothesis import rayleigh_test
import pandas as pd
from core.compareTopographicVectors import compare_topographic_vectors
from statsmodels.stats.multitest import multipletests

def run(project):
    # PATHS
    statistics_dir = project.statistics_dir
    statistics_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

    def circular_std(angles_rad):

        R = np.abs(np.mean(np.exp(1j * angles_rad)))
        R = np.clip(R, 1e-12, 1)

        return np.sqrt(-2 * np.log(R))

    def compute_cluster_orientation_stats(cluster, normals):

        angles, _, _, n = (
            compare_topographic_vectors(
                cluster.directions["topographic"],
                cluster.directions["gyrus"],
                normals,
                cluster.hemi
            )
        )

        if len(angles) == 0:
            return None

        angles_rad = np.deg2rad(angles)
        mean_angle = np.rad2deg(circmean(angles_rad, high=2*np.pi))
        median_angle = np.rad2deg(np.median(angles_rad))
        std_angle = np.rad2deg(circular_std(angles_rad))
        rayleigh_result = rayleigh_test(angles_rad)

        R = float(rayleigh_result.r)
        z = float(rayleigh_result.z)
        p = float(rayleigh_result.pval)

        topographic_vectors = cluster.directions["topographic"]

        strength = np.nanmean(np.linalg.norm(topographic_vectors, axis=1))

        return {
            "angles":
                angles.tolist(),
            "mean_angle":
                float(mean_angle),
            "median_angle":
                float(median_angle),
            "std_angle":
                float(std_angle),
            "resultant_length":
                R,
            "rayleigh_z":
                z,
            "rayleigh_p":
                p,
            "rayleigh_log10_p":
                float(np.log10(p)) if p > 0 else -np.inf,
            "n":
                int(n),
            "topographic_strength":
                float(strength)
        }
        
    normals_lh = project.geo_lh.compute_vertex_normals()
    normals_rh = project.geo_rh.compute_vertex_normals()

    rows = []

    for cluster in clusters:
        print(f"Processing {cluster.name}")

        if ("topographic"not in cluster.directions):
            print("  missing topographic")
            continue

        if ("gyrus" not in cluster.directions):
            print("  missing gyrus")
            continue

        if cluster.hemi == "lh":
            normals = normals_lh[cluster.valid_vertices]

        else:normals = normals_rh[cluster.valid_vertices]

        stats = (compute_cluster_orientation_stats(cluster, normals))

        cluster.metadata["orientation_statistics"] = stats

        print(
            project.subject,
            cluster.name,
            id(cluster),
            "orientation_statistics" in cluster.metadata
        )

        rows.append(
            {
                "subject": project.subject,
                "model": project.model,
                "hemisphere": cluster.hemi,
                "cluster": cluster.name,
                "mean_angle": stats["mean_angle"],
                "median_angle": stats["median_angle"],
                "std_angle": stats["std_angle"],
                "resultant_length": stats["resultant_length"],
                "rayleigh_z": stats["rayleigh_z"],
                "rayleigh_p": stats["rayleigh_p"],
                "rayleigh_log10_p":stats["rayleigh_log10_p"],
                "n_vectors": stats["n"],
                "topographic_strength": stats["topographic_strength"],
            }
        )
    

   
    # FDR correction within this subject
    p_values = [row["rayleigh_p"] for row in rows]

    reject, p_fdr, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh"
    )

    for row, fdr, sig in zip(rows, p_fdr, reject):
        row["rayleigh_p_fdr"] = float(fdr)
        row["rayleigh_significant_fdr"] = bool(sig)

    for row, cluster in zip(rows, clusters):
        if cluster.metadata.get("orientation_statistics") is not None:

            cluster.metadata["orientation_statistics"]["rayleigh_p_fdr"] = row["rayleigh_p_fdr"]
            cluster.metadata["orientation_statistics"]["rayleigh_significant_fdr"] = row["rayleigh_significant_fdr"]

    project.save_clusters(cluster_set)

    pd.DataFrame(rows).to_csv(
        statistics_dir
        / "cluster_orientation_statistics.csv",
        index=False
    )