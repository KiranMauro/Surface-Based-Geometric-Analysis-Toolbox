import subprocess
import numpy as np
from data.io import save_label
from visualization.overlay_colours import (hsv_overlay_string, polar_angle_cmap, r2_scale)

def run(project, show_clusters = True, show_centroids = True, parameter_threshold = None):
    model_type = project.model
    surf_dir = project.surf_dir
    parameter_map_dir = project.parameter_map_dir
    temp_dir = project.output_dir / "temp_visualization"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "N":
        parameter_cmap = hsv_overlay_string()
        lh_parameter = parameter_map_dir / "lh.N_display.mgh"
        rh_parameter = parameter_map_dir / "rh.N_display.mgh"

        if not parameter_threshold:
            parameter_threshold = "1.1,7"



    else:
        parameter_cmap = (polar_angle_cmap())
        lh_parameter = (parameter_map_dir / "lh.polar_angle.mgh")
        rh_parameter = (parameter_map_dir / "rh.polar_angle.mgh")
        if not parameter_threshold:
            parameter_threshold = "0,6.28318"

   

    base_lh = (
        f"{surf_dir}/lh.white:"

        f"overlay={lh_parameter}:"

        "overlay_color=custom:"

        f"overlay_custom={parameter_cmap}:"

        f"overlay_threshold={parameter_threshold}:"
        
        "overlay_opacity=1:"

        f"overlay={parameter_map_dir / 'lh.r2_display.mgh'}:"

        "overlay_color=custom:"

        f"overlay_custom={r2_scale()}:"

        "overlay_threshold=0.15,1:"
    )

    base_rh = (
        f"{surf_dir}/rh.inflated:"

        f"overlay={rh_parameter}:"

        "overlay_color=custom:"

        f"overlay_custom={parameter_cmap}:"

        f"overlay_threshold={parameter_threshold}:"

        "overlay_opacity=1:"

        f"overlay={parameter_map_dir / 'rh.r2_display.mgh'}:"

        "overlay_color=custom:"

        f"overlay_custom={r2_scale()}:"

        "overlay_threshold=0.15,1:"
    )

    if show_clusters:
        cluster_set = project.cluster_set
        clusters = cluster_set.final_clusters
        for cluster in clusters:
            if cluster.hemi == "lh":
                base_lh += (

                    f"label={cluster.metadata['label_file']}:"

                    "label_outline=1:"

                    "label_color=black:"
                )

            else:
                base_rh += (

                    f"label={cluster.metadata['label_file']}:"

                    "label_outline=1:"

                    "label_color=black:"
                )

    if show_centroids:
        for cluster in clusters:
            geo = project.geo[cluster.hemi]
            centroid_file = temp_dir / f"{cluster.name}_centroid.label"
        
            save_label(np.array([cluster.metadata["centroid_vertex"]], dtype=np.int32), geo.verts, centroid_file)
            if cluster.hemi == "lh":
                base_lh += (

                    f"label={centroid_file}:"

                    "label_outline=10:"

                    "label_color=black:"
                )

            else:
                base_rh += (

                    f"label={centroid_file}:"

                    "label_outline=10:"

                    "label_color=black:"
                )


    cmd = [
        "freeview",
        "-f",
        base_lh,
        base_rh
    ]

    subprocess.run(cmd, check=True)

    for f in temp_dir.glob("*.label"):
        f.unlink()
    
    temp_dir.rmdir()


