import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from data.io import load_clusters, save_label


def run(projects, group_ids=None, show_centroids=True, color_mode="group"):
    
    project_dirs = {p.subject: p.data_dir for p in projects}
    data_root = projects[0].project_root / "data"
    groups = load_clusters(data_root / "group_analysis" / "correspondence_groups.npy")['correspondence']
    fsavg_dir = projects[0].data_dir / "fsaverage" / "surf"

    temp_dir = data_root / "group_analysis" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def get_clusters(groups, group_ids):
        clusters = []

        for group in groups:
            if group_ids is not None and group.id not in group_ids:
                continue
            clusters.extend(group.members.values())

        return clusters

    

    coords = {
        "lh": nib.freesurfer.read_geometry(
            fsavg_dir / "lh.white"
        )[0],

        "rh": nib.freesurfer.read_geometry(
            fsavg_dir / "rh.white"
        )[0]

    }

    GROUP_COLORS = [
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "orange",
    ]

    SUBJECT_COLORS = {
        project.subject: GROUP_COLORS[i % len(GROUP_COLORS)]
        for i, project in enumerate(projects)
    }

    def get_color(cluster):

        if color_mode == "group":
            return GROUP_COLORS[(cluster.group_id - 1) % len(GROUP_COLORS)]

        return SUBJECT_COLORS[cluster.subject]


    def add_cluster(base, cluster, color):
        label = (project_dirs[cluster.subject]/ "fsaverage_results"/ f"{cluster.subject}_{cluster.name}.label")

        if not label.exists():
            print("Missing:", label)
            return base

        print("Found:", label)

        base += (
            f"label={label}:"
            "label_outline=1:"
            f"label_color={color}:"
        )
        return base

    def add_centroid(base, cluster):

        centroid_file = (temp_dir/ f"{cluster.subject}_{cluster.name}_centroid.label")

        save_label(np.array([cluster.fsaverage_centroid], dtype=np.int32),coords[cluster.hemi],centroid_file
                   )
        base += (
            f"label={centroid_file}:"

            "label_outline=10:"

            "label_color=black:"
        )
        return base


    base_lh = (f"{fsavg_dir}/lh.inflated:curvature=on:")
    base_rh = (f"{fsavg_dir}/rh.inflated:curvature=on:")
    
    clusters = get_clusters(groups, group_ids)
   
    for cluster in clusters:

        color = get_color(cluster)

        if cluster.hemi == "lh":

            base_lh = add_cluster(
                base_lh,
                cluster,
                color
            )

            if show_centroids:

                base_lh = add_centroid(
                    base_lh,
                    cluster
                )

        else:

            base_rh = add_cluster(
                base_rh,
                cluster,
                color
            )

            if show_centroids:

                base_rh = add_centroid(
                    base_rh,
                    cluster
                )

    cmd = [

        "freeview",
        "-f",
        base_lh,
        base_rh
    ]

    subprocess.run(cmd,check=True)

    for f in temp_dir.glob("*.label"):
        f.unlink()