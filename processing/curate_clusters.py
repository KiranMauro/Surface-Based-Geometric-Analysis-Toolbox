import os
import glob
import subprocess
import numpy as np
from copy import deepcopy
import shutil
from PyQt5.QtWidgets import QApplication
from visualization.curation_gui import ClusterCuratorDialog
from visualization.overlay_colours import (hsv_overlay_string, polar_angle_cmap, r2_scale)
from data.structures import Cluster
from data.io import save_label
from PyQt5.QtWidgets import QMessageBox


def run(project):
    model_type = project.model

    # Paths
    surf_dir = project.surf_dir
    parameter_map_dir = project.parameter_map_dir
    final_dir = project.final_dir
    deleted_dir = project.deleted_dir

    final_dir.mkdir(parents=True, exist_ok=True)

   
    # Load Clusters
    cluster_set = project.cluster_set

    lh_clusters = [
        c for c in cluster_set.final_clusters
        if c.hemi == "lh"
    ]

    rh_clusters = [
        c for c in cluster_set.final_clusters
        if c.hemi == "rh"
    ]

   
    # Helpers
    CLUSTER_COLORS = [
            "#FF00EA", 
            "#000000", 
            "#FFFFFF", 
            "#E93159",
            "#5F3600",  
            "#115750", 
            "#E0CF34",
            "#15361B",  
            "#FA8072", 
        ]

    def launch_freeview(clusters):
        hemi = clusters[0].hemi

        if model_type == "N":

            parameter_map = (
                parameter_map_dir
                / f"{hemi}.N_display.mgh"
            )

            parameter_threshold = "1.1,7"

            parameter_cmap = (
                hsv_overlay_string()
            )

        else:
            parameter_map = (
                parameter_map_dir
                / f"{hemi}.polar_angle.mgh"
            )

            parameter_threshold = "0,6.28318"

            parameter_cmap = (
                polar_angle_cmap()
            )

        r2_map = (
            parameter_map_dir
            / f"{hemi}.r2_{model_type}.mgh"
        )

        r2_cmap = (
            r2_scale()
        )

        base = (

            f"{surf_dir}/{hemi}.inflated:"

            f"overlay={parameter_map}:"

            "overlay_color=custom:"

            f"overlay_custom={parameter_cmap}:"

            f"overlay_threshold={parameter_threshold}:"

            f"overlay={r2_map}:"

            "overlay_color=custom:"

            f"overlay_custom={r2_cmap}:"

            "overlay_threshold=0.15,1:"
        )

        for i, cluster in enumerate(clusters):

            colour = CLUSTER_COLORS[
                i % len(CLUSTER_COLORS)
            ]

            base += (

                f"label={cluster.metadata['label_file']}:"

                "label_outline=1:"

                f"label_color={colour}:"

            )

        cmd = [

            "freeview",

            "-f",

            base

        ]

        return subprocess.Popen(cmd)

    def merge_clusters(cluster_list):

        vertices = np.unique(

            np.concatenate(
                [
                    c.valid_vertices
                    for c in cluster_list
                ]
            )
        )

        label_vertices = np.unique(

            np.concatenate(

                [
                    c.label_vertices
                    for c in cluster_list
                ]

            )

        )

        cluster = Cluster(
            id=-1,
            hemi=cluster_list[0].hemi,
            subject = project.subject,
            valid_vertices=vertices,
            label_vertices= label_vertices
        )

        cluster.history = {
        "overall_merge": {
            "merged_from": [c.id for c in cluster_list],
            "source_histories": [deepcopy(c.history) for c in cluster_list]
        }
    }
        return cluster

    def are_adjacent(cluster1, cluster2, geo):

        verts2 = set(cluster2.label_vertices)
        for v in cluster1.label_vertices:

            neighbors = set(geo.graph[v].indices)

            if neighbors & verts2:
                return True

        return False

    def group_is_connected(cluster_group, geo):

        n = len(cluster_group)
        if n <= 1:
            return True

        visited = {0}
        stack = [0]

        while stack:
            i = stack.pop()
            for j in range(n):
                if j in visited:
                    continue

                if are_adjacent(cluster_group[i], cluster_group[j], geo):
                    visited.add(j)
                    stack.append(j)
        return len(visited) == n

   
    # Initialize 
    new_final_clusters = []
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    for clusters in [lh_clusters, rh_clusters]:
        valid_partition = False
        while not valid_partition:
            freeview_process = launch_freeview(clusters)
            dialog = ClusterCuratorDialog(clusters)
            dialog.exec_()
            result = dialog.get_result()
            freeview_process.terminate()
            valid_partition = True

          
            for group in result["groups"].values():

                cluster_group = [
                    clusters[i]
                    for i in group
                ]

                geo = project.geo[cluster_group[0].hemi]

                if not group_is_connected(
                    cluster_group,
                    geo
                ):
                    reply = QMessageBox.warning(
                        None,
                        "Warning",
                        "These clusters are not adjacent.\n\nMerge anyway?",
                        QMessageBox.Yes,
                        QMessageBox.No

                    )
                    if reply == QMessageBox.No:
                        valid_partition = False
                        break

       
        # Commit accepted partition
        for idx in result["deleted"]:

            cluster = clusters[idx]
            cluster.history["deleted_at"] = ("overall_curation")
            cluster.history["deletion_reason"] = ("manual")

            old_label = cluster.metadata["label_file"]
            if os.path.exists(old_label):
                new_label = deleted_dir / os.path.basename(old_label)
                shutil.move(old_label, new_label)
                cluster.metadata["label_file"] = str(new_label)
            cluster_set.deleted_clusters.append(cluster)

        for group in result["groups"].values():

            cluster_group = [
                clusters[i]
                for i in group
            ]

            merged = merge_clusters(
                cluster_group
            )

            new_final_clusters.append(
                merged
            )
            
    next_id = 1

    for cluster in new_final_clusters:
        cluster.id = next_id
        next_id += 1

        cluster_set.final_clusters = (
            new_final_clusters
        )

    for f in glob.glob(str(final_dir/ "*.label")):
        os.remove(f)

    for cluster in cluster_set.final_clusters:
        coords = project.geo[cluster.hemi].verts
        label_file = final_dir / f"{cluster.hemi}_final_{cluster.id:04d}.label"
        save_label(cluster.label_vertices, coords, label_file)

        cluster.metadata["label_file"] = str(label_file)

    project.save_clusters(cluster_set)