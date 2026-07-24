import numpy as np
import pandas as pd
from data.io import load_clusters
from core.graph import GraphGeodesics
from core.correspondence import compute_cluster_matches
from pathlib import Path

def run(project):
    #Paths
    manual_roi_path = project.output_dir / "data_structures" / "manual_rois.npy"
    statistics_dir = project.statistics_dir
    statistics_dir.mkdir(parents=True, exist_ok=True)

    manual_set = load_clusters(manual_roi_path)
    cluster_set = project.cluster_set

    manual_rois = manual_set.final_clusters
    clusters = cluster_set.final_clusters

    #Helpers
    def compute_validation_metrics(manual_roi,cluster):
        manual = set(manual_roi.label_vertices)
        predicted = set(cluster.label_vertices)
        intersection = len(manual & predicted)
        precision = (intersection / len(predicted) if len(predicted) else np.nan)
        recall = (intersection / len(manual) if len(manual) else np.nan)
        area_manual = (manual_roi.metadata["surface_area"])
        area_cluster = (cluster.metadata["surface_area"])

        return {
            "precision":
                precision,
            "recall":
                recall,
            "area_manual":
                area_manual,
            "area_cluster":
                area_cluster,
            "area_difference":
                area_cluster
                - area_manual,
        }

    rows = []

    for hemi in ["lh", "rh"]:
        geo = project.geo[hemi]

        roi_hemi = [r for r in manual_rois if r.hemi == hemi]
        cluster_hemi = [c for c in clusters if c.hemi == hemi]
        print(f"{hemi}: {len(roi_hemi)} manual ROIs")
        print(f"{hemi}: {len(cluster_hemi)} predicted clusters")
        # Compute native-space centroids once
        # Only manual ROIs need a centroid computed
        print('Computing ROI centroids')
        for roi in roi_hemi:
            roi_dist = geo.roi_distance_matrix(roi.label_vertices)
            roi.centroid = int(
                geo.centroid_from_distance_matrix(
                    roi.label_vertices,
                    roi_dist
                )
            )
            
        matches = compute_cluster_matches(
            roi_hemi,
            cluster_hemi,
            geo,
            vertex_set="label_vertices",
            centroid_attr="centroid",
        )

        print(f"{hemi}: {len(matches)} matches")
            
        for match in matches:
            roi = match["cluster1"]
            cluster = match["cluster2"]
            metrics = compute_validation_metrics(roi, cluster)
            rows.append(
                {
                    "subject":
                        project.subject,
                    "model":
                        project.model,
                    "hemisphere":
                        hemi,
                    "roi":
                        roi.name,
                    "matched_cluster":
                        cluster.name,
                    "dice":
                        match["dice"],
                    "precision":
                        metrics["precision"],
                    "recall":
                        metrics["recall"],
                    "centroid_distance_mm":
                        match["centroid_distance"],
                    "matching_similarity":
                        match["similarity"],
                    "area_manual_mm2":
                        metrics["area_manual"],
                    "area_cluster_mm2":
                        metrics["area_cluster"],
                    "area_difference_mm2":
                        metrics["area_difference"],
                }
            )

    pd.DataFrame(rows).to_csv(statistics_dir / "manual_roi_statistics.csv", index=False)