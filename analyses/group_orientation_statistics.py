import numpy as np
import pandas as pd
from pycircstat2.hypothesis import rayleigh_test
from statsmodels.stats.multitest import multipletests


def run(projects):
    #Paths
    output_dir = projects[0].project_root / "data" / "group_analysis"
    correspondence_path = output_dir / "correspondence_groups.npy"
    statistics_dir = output_dir / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)


    # Helpers
    def circular_mean(angles_deg):
        angles_rad = np.deg2rad(angles_deg)

        return (np.rad2deg(np.angle(np.mean(
                        np.exp(1j * angles_rad))))% 360)


    def weighted_circular_mean(angles_deg, weights):
        angles_rad = np.deg2rad(angles_deg)

        z = np.sum(weights * np.exp(1j * angles_rad)) / np.sum(weights)

        return np.rad2deg(np.angle(z)) % 360

    def resultant_length(angles_deg):
        angles_rad = np.deg2rad(angles_deg)

        return np.abs(np.mean(np.exp(1j * angles_rad)))

    groups = np.load(correspondence_path, allow_pickle=True).item()   

    print("Number of correspondence groups:", len(groups["correspondence"]))

    rows = []
    group_results = []
   
    for group in groups["correspondence"]:

        angles = []
        weights = []

        for subject, cluster in group.members.items():
            if "orientation_statistics" not in cluster.metadata:
                print(
                    "Missing:",
                    subject,
                    cluster.subject,
                    cluster.id,
                    cluster.name,
                    cluster.metadata.keys()
                )
                raise RuntimeError("Missing orientation statistics")

            stats = cluster.metadata["orientation_statistics"]
            angle = stats["mean_angle"]
            weights.append(stats["resultant_length"])

            angles.append(angle)
       
        angles = np.asarray(angles,dtype=float)
        weights = np.asarray(weights, float)
        angles_rad = np.deg2rad(angles)
        rayleigh_result = rayleigh_test(angles_rad)

        z = float(rayleigh_result.z)
        p = float(rayleigh_result.pval)

        group_stats = {
            "group_id":group.id,
            "n_subjects": len(angles),
            "mean_angle":circular_mean(angles),
            "weighted_mean_angle":weighted_circular_mean(angles,weights),
            "resultant_length":resultant_length(angles),
            "rayleigh_z": float(z),
            "rayleigh_p": float(p), 
            "rayleigh_log10_p":float(np.log10(p)) if p > 0 else -np.inf,
            "angles": angles.tolist()}

        group_results.append(group_stats)
        rows.append(
            {
                "group_id": group.id,
                "clusters": "; ".join(
                    f"{subject}:{cluster.name}"
                    for subject, cluster in group.members.items()
                ),
                "n_subjects": len(angles),
                "mean_angle": group_stats["mean_angle"],
                "weighted_mean_angle": weighted_circular_mean(angles, weights),
                "resultant_length": group_stats["resultant_length"],
                "rayleigh_z": group_stats["rayleigh_z"],
                "rayleigh_p": group_stats["rayleigh_p"],
                "rayleigh_log10_p":group_stats["rayleigh_log10_p"],
            }
        )

    p_values = [row["rayleigh_p"] for row in rows]

    reject, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    for row, group_stats, fdr, sig in zip(rows,group_results,p_fdr,reject):

        row["rayleigh_p_fdr"] = float(fdr)
        row["rayleigh_significant_fdr"] = bool(sig)
        group_stats["rayleigh_p_fdr"] = float(fdr)
        group_stats["rayleigh_significant_fdr"] = bool(sig)


    np.save(output_dir / "group_orientation_statistics.npy", group_results, allow_pickle=True)
    pd.DataFrame(rows).to_csv(statistics_dir / "group_orientation_statistics.csv", index=False)