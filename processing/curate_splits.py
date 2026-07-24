import numpy as np
import subprocess
from data.structures import Cluster
from data.io import save_label
from visualization.curation_gui import SplitCuratorDialog
from PyQt5.QtWidgets import QApplication
from visualization.overlay_colours import hsv_overlay_string, polar_angle_cmap, r2_scale

def run(project):
    model_type = project.model
  
    surf_dir = project.surf_dir
    parameter_map_dir = project.parameter_map_dir
    final_dir = project.final_dir
    final_dir.mkdir(parents=True, exist_ok=True)
   
    # LOAD
    cluster_set = project.cluster_set

    final_clusters = cluster_set.final_clusters
    new_final_clusters = []
    
    deleted_clusters = cluster_set.deleted_clusters

    
    # HELPERS
    def launch_freeview(
        cluster,
        surf_dir,
        parameter_map_dir,
        model_type
    ):

        splits = (
            cluster.metadata["suggested_splits"]
        )

        colours =[
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

     
        # ========================================================
        # parameter MAPS
        # ========================================================
        if model_type == "N":
            lh_map = (parameter_map_dir/ "lh.N_display.mgh")
            rh_map = (parameter_map_dir/ "rh.N_display.mgh")
            parameter_threshold = "1.1,7"
            parameter_cmap = (hsv_overlay_string())

        else:
            lh_map = (parameter_map_dir/ "lh.polar_angle.mgh")
            rh_map = (parameter_map_dir/ "rh.polar_angle.mgh")

            parameter_threshold = "0,6.28318"

            # temporary until we build a proper circular VFM map

            parameter_cmap = (polar_angle_cmap())


        lh_r2 = (parameter_map_dir/ f"lh.r2_{model_type}.mgh")
        rh_r2 = (parameter_map_dir/ f"rh.r2_{model_type}.mgh")

        r2_cmap = r2_scale()

       
        # LH SURFACE
      

        if model_type == "N":
            base_lh = (

                f"{surf_dir}/lh.inflated:"

                f"overlay={lh_map}:"

                "overlay_color=custom:"

                f"overlay_custom={parameter_cmap}:"

                f"overlay_threshold={parameter_threshold}:"

                f"overlay={lh_r2}:"

                "overlay_color=custom:"

                f"overlay_custom={r2_cmap}:"

                "overlay_threshold=0.15,1:"

            )

        else:

            base_lh = (

                f"{surf_dir}/lh.inflated:"

                f"overlay={lh_map}:"

                "overlay_color=custom:"

                f"overlay_custom={parameter_cmap}:"

                f"overlay_threshold={parameter_threshold}:"

                f"overlay={lh_r2}:"

                "overlay_color=custom:"

                f"overlay_custom={r2_cmap}:"

                "overlay_threshold=0.15,1:"
            )

  
        # RH SURFACE
        if model_type == "N":
            base_rh = (

                f"{surf_dir}/rh.inflated:"

                f"overlay={rh_map}:"

                "overlay_color=custom:"

                f"overlay_custom={parameter_cmap}:"

                f"overlay_threshold={parameter_threshold}:"

                f"overlay={rh_r2}:"

                "overlay_color=custom:"

                f"overlay_custom={r2_cmap}:"

                "overlay_threshold=0.15,1:"
            )

        else:
            base_rh = (

                f"{surf_dir}/rh.inflated:"

                f"overlay={rh_map}:"

                "overlay_color=heat:"

                f"overlay_threshold={parameter_threshold}:"

                f"overlay={rh_r2}:"

                "overlay_color=custom:"

                f"overlay_custom={r2_cmap}:"

                "overlay_threshold=0.15,1:"
            )

       
        # SPLITS
        for i, split in enumerate(splits):
            colour = colours[i % len(colours)]
            label_file = (split["label_file"])

            if cluster.hemi == "lh":
                base_lh += (
                    f"label={label_file}:"

                    "label_outline=1:"

                    f"label_color={colour}:"
                )

            else:
                base_rh += (
                    f"label={label_file}:"

                    "label_outline=1:"

                    f"label_color={colour}:"
                )

       
        # LAUNCH
        if cluster.hemi == "lh":
            cmd = [
                "freeview",
                "-f",
                base_lh
            ]

        else:
            cmd = [
                "freeview",
                "-f",
                base_rh
            ]

        return subprocess.Popen(cmd)


    def build_cluster_from_vertices(source_cluster, vertices):

        return Cluster(
            id=-1,
            hemi=source_cluster.hemi,
            subject = project.subject,
            valid_vertices=np.asarray(
                vertices,
                dtype=np.int32
            ),
            label_vertices=vertices.copy(),
            metadata={"model_type": model_type}
        )

    next_cluster_id = 1

    app = QApplication.instance()

    if app is None:
        app = QApplication([])
    
    # CURATION LOOP
    for cluster in final_clusters:

        if ("suggested_splits"not in cluster.metadata):
            
            cluster.history.update({
                "candidate_cluster_id": cluster.id,
                "source_splits": None,
            })

            cluster.id = next_cluster_id
            next_cluster_id += 1

            new_final_clusters.append(cluster)
            continue

        freeview_process = launch_freeview(
            cluster=cluster,
            surf_dir=surf_dir,
            parameter_map_dir=parameter_map_dir,
            model_type=model_type
        )

        splits = (
            cluster.metadata[
                "suggested_splits"
            ]
        )

        dialog = SplitCuratorDialog(cluster)

        dialog.exec_()

        result = dialog.get_result()

        freeview_process.terminate()

        if result["action"] == "keep_original":

            cluster.history = {
                "candidate_cluster_id": cluster.id,
                "source_splits": None,
            }

            cluster.id = next_cluster_id
            next_cluster_id += 1

            new_final_clusters.append(cluster)
            continue

        elif result["action"] == "delete_cluster":

            cluster.history = {

                "candidate_cluster_id":
                    cluster.id,

                "source_splits":
                    None,

                "deleted_at":
                    "curation",

                "deletion_reason":
                    "manual"

            }

            deleted_clusters.append(
                cluster
            )

            continue

        elif result["action"] == "partition":
            groups = result["groups"]

            deleted = result["deleted"]

            splits = cluster.metadata[
                "suggested_splits"
            ]

            for split_ids in groups.values():

                merged_vertices = []

                for split_id in split_ids:

                    merged_vertices.extend(

                        splits[
                            split_id
                        ]["vertices"]

                    )

                merged_vertices = np.unique(
                    merged_vertices
                )

                new_cluster = (
                    build_cluster_from_vertices(
                        cluster,
                        merged_vertices
                    )
                )

                new_cluster.history = {

                    "candidate_cluster_id":
                        cluster.id,

                    "source_splits":
                        split_ids,

                }

                new_cluster.id = (
                    next_cluster_id
                )

                new_final_clusters.append(new_cluster)

                next_cluster_id += 1

            for split_id in deleted:

                deleted_cluster = (
                    build_cluster_from_vertices(
                        cluster,
                        splits[
                            split_id
                        ]["vertices"]
                    )
                )

                deleted_cluster.history = {

                "candidate_cluster_id":
                    cluster.id,

                "source_splits":
                    [split_id],

                "deleted_at":
                    "curation",

                "deletion_reason":
                    "manual"

            }

                deleted_clusters.append(
                    deleted_cluster
                )

    
    # Generate labels
    cluster_set.final_clusters = new_final_clusters

    for cluster in new_final_clusters:
        geo = project.geo[cluster.hemi]
        coords = geo.verts

        label_file = (final_dir/f"{cluster.hemi}_final_{cluster.id:04d}.label")
        save_label(cluster.valid_vertices, coords, label_file)
        cluster.metadata["label_file"] = str(label_file)


    # SAVE

    for cluster in new_final_clusters:
        cluster.metadata.pop("suggested_splits", None)

    for cluster in deleted_clusters:
        cluster.metadata.pop("suggested_splits", None)

    project.save_clusters(cluster_set)

    print()
    print("=" * 60)
    print()

    print(
        f"Final clusters: "
        f"{len(final_clusters)}"
    )

    print(
        f"Deleted clusters: "
        f"{len(deleted_clusters)}"
    )
