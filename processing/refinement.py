import os
import glob
import subprocess
import nibabel as nib
import numpy as np
import shutil
from PyQt5.QtWidgets import QApplication
from visualization.curation_gui import (RefinementDialog, PreviewDecisionDialog)
from visualization.overlay_colours import (hsv_overlay_string,polar_angle_cmap,r2_scale)
from data.io import save_label
from PyQt5.QtWidgets import QMessageBox
from scipy.sparse.csgraph import connected_components

def run(project):
    # SETTINGS
    model_type = project.model
    preview_counter = 0
    settings = project.settings.processing
    default_erode = settings.default_erode
    default_dilate = settings.default_dilate
    default_min_component_area = settings.min_component_area
    
    # PATHS
    out_dir = project.output_dir
    surf_dir = project.surf_dir
    parameter_map_dir = project.parameter_map_dir
    final_dir = project.final_dir
    temp_dir = out_dir / "temp_refinement"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # LOAD CLUSTERS
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

    # HELPERS
    def read_label(path):
        return np.asarray(
            nib.freesurfer.read_label(
                str(path)
            ),

            dtype=np.int32

        )

    def cluster_to_label(
        cluster,
        geo,
        outfile
    ):

        save_label(
            cluster.label_vertices,
            geo.verts,
            outfile
        )



    def erode_label(
        label_file,
        hemi,
        steps,
        base
    ):

        out_file = (
            temp_dir
            / f"{base}_eroded.label"
        )

        if steps == 0:
            shutil.copy(
                label_file,
                out_file
            )

            return out_file

        cmd = [
            "mris_label_calc",
            "erode",
            str(steps),
            str(label_file),
            str(surf_dir/ f"{hemi}.white"),
            str(out_file)
        ]

        subprocess.run(cmd, check=True)

        return out_file

    def dilate_label(
        label_file,
        hemi,
        steps,
        base
    ):

        out_file = (temp_dir/ f"{base}_dilated.label")

        if steps == 0:
            shutil.copy(
                label_file,
                out_file
            )

            return out_file

        cmd = [
            "mris_label_calc",
            "dilate",
            str(steps),
            str(label_file),
            str(surf_dir/ f"{hemi}.white"),
            str(out_file)
        ]

        subprocess.run(cmd, check=True)

        return out_file

    def remove_small_components(
        vertices,
        geo,
        min_area
    ):

        vertices = np.asarray(
            vertices,
            dtype=np.int32
        )

        if len(vertices) == 0:

            return vertices

        subgraph = geo.graph[
            vertices
        ][
            :,
            vertices
        ]

        n_components, labels = (
            connected_components(
                subgraph,
                directed=False
            )
        )

        cleaned = []

        for component_id in range(
            n_components
        ):

            component_vertices = vertices[
                labels == component_id
            ]

            area = geo.surface_area(component_vertices)

            if area >= min_area:
                cleaned.append(component_vertices)

        if len(cleaned) == 0:

            return np.array(
                [],
                dtype=np.int32
            )

        return np.concatenate(
            cleaned
        )

    def generate_preview(
        cluster,
        geo,
        erode_steps,
        dilate_steps,
        min_component_area,
        preview_counter
    ):

        base = (
            f"cluster_"
            f"{cluster.id}_"
            f"{preview_counter}"
        )

        original_label = (
            temp_dir
            / f"{base}_original.label"
    )

        cluster_to_label(
            cluster,
            geo,
            original_label
        )

   
        # Erode
        eroded_label = (
            erode_label(
                original_label,
                cluster.hemi,
                erode_steps,
                base
            )
        )
        eroded_vertices = read_label(
            eroded_label
        )

        
        # Remove detached pieces
        cleaned_vertices = remove_small_components(
            eroded_vertices,
            geo,
            min_component_area,
        )

        if len(cleaned_vertices) == 0:

            QMessageBox.warning(

                None,

                "Cluster disappeared",

                (
                    "The current refinement parameters removed the entire cluster.\n\n"
                    "Try reducing the erosion or decreasing the minimum component size."
                )

            )

            return (
                None,
                None,
                None
            )

        cleaned_label = (
            temp_dir
            / f"{base}_cleaned.label"
        )

        save_label(
            cleaned_vertices,
            geo.verts,
            cleaned_label
        )

        
        # Dilate surviving component
        dilated_label = (
            dilate_label(
                cleaned_label,
                cluster.hemi,
                dilate_steps,
                base
            )
        )

        preview_vertices = read_label(
            dilated_label
        )


        preview_label = (
            temp_dir
            / f"{base}_preview.label"
        )

        save_label(

            preview_vertices,

            geo.verts,

            preview_label

        )

        return (

            preview_vertices,

            original_label,

            preview_label

        )

    def launch_preview(
        cluster,
        label
    ):

        hemi = cluster.hemi

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

            f"label={label}:"

            "label_outline=1:"

            "label_color=black:"
        )

        cmd = [
            "freeview",
            "-f",
            base
        ]

        return subprocess.Popen(cmd)


    def resolve_overlaps(clusters):
    
        for cluster in clusters:
            cluster.label_vertices = set(cluster.label_vertices)

       
        vertex_to_clusters = {}
        for cluster in clusters:
            hemi = cluster.hemi
            for v in cluster.label_vertices:

                key = (hemi, int(v))
                vertex_to_clusters.setdefault(key,[]).append(cluster)

        distance_maps = {}
        def get_distance_map(cluster):
            key = (cluster.hemi, cluster.id)
            if key in distance_maps:
                return distance_maps[key]

            geo = project.geo[cluster.hemi]

            vertices = np.asarray(
                sorted(cluster.label_vertices),
                dtype=np.int32
            )

            roi = geo.roi_distance_matrix(vertices)
            centroid = geo.centroid_from_distance_matrix(vertices, roi)
            dist = geo.distance_from_vertex(centroid)

            distance_maps[key] = dist

            return dist

        n_overlaps = 0
        for (hemi, vertex), owners in vertex_to_clusters.items():
            if len(owners) == 1:
                continue

            n_overlaps += 1
            if n_overlaps == 1:
                print(
                    f"Resolving vertex overlap found in {hemi}_clusters: ",
                    [c.id for c in owners]
                )

            best_cluster = min(
                owners,
                key=lambda c:
                    get_distance_map(c)[vertex]
            )

            for cluster in owners:
                if cluster is best_cluster:
                    continue
                cluster.label_vertices.discard(vertex)

       
        # Convert back to arrays
        for cluster in clusters:
            cluster.label_vertices = np.asarray(
                sorted(cluster.label_vertices), dtype=np.int32)

        print(f"Resolved {n_overlaps} overlapping vertices.")

    
    # MAIN LOOP
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    for cluster in clusters:
        geo = project.geo[cluster.hemi]

        erode_steps = default_erode
        dilate_steps = default_dilate
        min_component_area = default_min_component_area

        accepted = False
        freeview_process = None

        original_label = (temp_dir/ "original.label")

        cluster_to_label(cluster, geo, original_label)

        freeview_process = launch_preview(cluster, original_label)

        while not accepted:
            dialog = RefinementDialog(
                cluster,
                erode_steps,
                dilate_steps,
                min_component_area
            )

            dialog.exec_()
            result = dialog.get_result()
            erode_steps = result["erode"]
            dilate_steps = result["dilate"]
            min_component_area = result["min_component_area"]

            preview_counter += 1
            (preview_vertices, original_label, preview_label) = generate_preview(
                cluster,
                geo,
                erode_steps,
                dilate_steps,
                min_component_area,
                preview_counter
            )

            if preview_vertices is None:
                continue

            if freeview_process is not None:
                freeview_process.terminate()

                try:
                    freeview_process.wait(timeout=5)
                except:
                    pass             

            freeview_process = launch_preview(cluster, preview_label)

            # Accept or adjust
           
            decision = PreviewDecisionDialog()
            decision.exec_()

            if decision.get_result() == "accept":
                if freeview_process is not None:
                    freeview_process.terminate()
                    try:
                        freeview_process.wait(timeout=5)
                    except:
                        pass

                cluster.label_vertices = preview_vertices
                cluster.history["refinement"] = {
                    "erode": erode_steps,
                    "dilate": dilate_steps, 
                    "min_component_area": min_component_area
                }

                accepted = True
            else:
                continue

    resolve_overlaps(cluster_set.final_clusters)

    map_lh = nib.load(parameter_map_dir / f"lh.{model_type}.mgh").get_fdata().squeeze()
    map_rh = nib.load(parameter_map_dir / f"rh.{model_type}.mgh").get_fdata().squeeze()

    for cluster in clusters:
        if cluster.hemi == 'lh':
            valid_vertices = cluster.label_vertices[~np.isnan(map_lh[cluster.label_vertices])]
        else:
            valid_vertices = cluster.label_vertices[~np.isnan(map_rh[cluster.label_vertices])]
        cluster.valid_vertices = valid_vertices
    
    # SAVE LABELS
    for f in glob.glob(str(final_dir/ "*.label")):
        os.remove(f)

    for cluster in cluster_set.final_clusters:
        geo = project.geo[cluster.hemi]
        label_file = (final_dir/f"{cluster.hemi}_final_{cluster.id:04d}.label")
        save_label(cluster.label_vertices, geo.verts, label_file)
        cluster.metadata["label_file"] = str(label_file)

    # SAVE CLUSTERS
    project.save_clusters(cluster_set)

    print(f"Saved {len(cluster_set.final_clusters)} refined clusters")

    for f in temp_dir.glob("*"):
        f.unlink()

    temp_dir.rmdir()
