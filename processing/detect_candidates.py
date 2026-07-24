import glob
import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from data.structures import Cluster
from data.structures import ClusterSet

def run(project):
    settings = project.settings.processing
    threshold = settings.thmin
    min_area = settings.min_area
    GOF_threshold = settings.gof_threshold
    min_fraction = settings.min_fraction

    # ============================================================
    # PATHS
    # ============================================================
    parameter_map_dir = project.parameter_map_dir
    candidate_label_dir = project.candidate_dir
    summary_dir = (candidate_label_dir / "summary")
    deleted_label_dir = project.deleted_dir

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(project.data_dir)

    candidate_label_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    deleted_label_dir.mkdir(parents=True, exist_ok=True)

    geo = project.geo

    # ============================================================
    # MODEL-SPECIFIC FILES
    # ============================================================

    if project.model == "N":
        parameter_name = "N"

    elif project.model == "VFM":
        parameter_name = "polar_angle"

    else:
        raise ValueError(
            f"Unknown MODEL_TYPE: {project.model}"
        )

    lh_parameter_path = (parameter_map_dir/ f"lh.{parameter_name}.mgh")

    rh_parameter_path = (parameter_map_dir / f"rh.{parameter_name}.mgh")

    lh_r2_path = (parameter_map_dir/ f"lh.r2_{project.model}.mgh")

    rh_r2_path = (parameter_map_dir/ f"rh.r2_{project.model}.mgh")

   
    # LOAD GOF MAPS
    gof_lh = nib.load(lh_r2_path).get_fdata().squeeze()

    gof_rh = nib.load(rh_r2_path).get_fdata().squeeze()

    # RUN SURFCLUSTER
    def run_cluster(hemi,in_file,out_name):

        cmd = [

            "mri_surfcluster",

            "--in",
            str(in_file),

            "--subject",
            "s1",

            "--hemi",
            hemi,

            "--surf",
            "white",

            "--thmin",
            str(threshold),

            "--annot",
            "aparc",

            "--minarea",
            str(min_area),

            "--o",
            str(
                summary_dir / out_name
            ),

            "--sum",
            str(
                summary_dir
                / f"{out_name}.summary"
            ),

            "--ocn",
            str(
                summary_dir
                / f"{out_name}.cluster_numbers"
            ),

            "--olab",
            str(
                candidate_label_dir
                / f"{out_name}_label"
            ),
        ]

        subprocess.run(cmd,check=True,env=env)


    # BUILD CLUSTERS

    def build_candidates(label_files,gof_map, hemi):

        candidates = []

        for label_file in label_files:

            vertices = np.asarray(

                nib.freesurfer.read_label(
                    label_file
                ),

                dtype=np.int32
            )

            vertices = np.unique(vertices)

            if len(vertices) == 0:
                continue

            cluster_gof = gof_map[vertices]

            cluster_gof = cluster_gof[~np.isnan(cluster_gof)]

            good_fraction = np.mean(cluster_gof > GOF_threshold)

            cluster = Cluster(
                id=-1,
                hemi=hemi,
                subject = project.subject,
                valid_vertices=vertices,
                label_vertices = vertices.copy()
            )

            cluster.metrics["n_vertices"] = int(len(vertices))
            cluster.metadata["surface_area"] = (geo[hemi].surface_area(vertices))
            cluster.metrics["good_fraction"] = float(good_fraction)
            cluster.metadata["model_type"] = project.model

            cluster.metadata["label_file"] = str(label_file)
            
            if good_fraction < min_fraction:

                new_path = (deleted_label_dir/Path(label_file).name)
                os.rename(label_file, new_path)
                cluster.metadata["label_file"] = str(new_path)

                cluster.history["deleted_at"] = "candidate"
                cluster.history["deletion_reason"] = "low_gof"
                print('Goodness of fit too low')
                cluster_set.deleted_clusters.append(cluster)

                continue

            candidates.append(cluster)
        return candidates


    # RUN CLUSTERING
    cluster_set = ClusterSet()

    cluster_set.metadata = {

        "subject": project.subject,

        "model_type": project.model,

        "cluster_threshold": threshold,

        "min_area": min_area,

        "gof_threshold": GOF_threshold,

        "min_fraction": min_fraction

    }

    run_cluster("lh", lh_parameter_path, "lh_candidate")
    run_cluster("rh", rh_parameter_path, "rh_candidate")

    print("\nInitial clustering complete")


    # LOAD LABEL FILES
    label_files_lh = sorted(glob.glob(str(candidate_label_dir/ "lh_candidate_label-*.label")))
    label_files_rh = sorted(glob.glob(str(candidate_label_dir/ "rh_candidate_label-*.label")))

    
    # BUILD CANDIDATS
  

    clusters_lh = build_candidates(

        label_files_lh,

        gof_lh,

        "lh"
    )

    clusters_rh = build_candidates(

        label_files_rh,

        gof_rh,

        "rh"
    )

    candidate_clusters = (
        clusters_lh
        +
        clusters_rh
    )

    for cluster_id, cluster in enumerate(candidate_clusters, start=1):

        cluster.id = cluster_id

        old = Path(cluster.metadata["label_file"])
        new = old.parent / f"{cluster.hemi}_candidate_{cluster.id:04d}.label"

        os.rename(old, new)

        cluster.metadata["label_file"] = str(new)

    # ============================================================
    # SAVE
    # ============================================================


    cluster_set.candidate_clusters = (
        candidate_clusters
    )

    cluster_set.final_clusters = candidate_clusters.copy()

    project.save_clusters(cluster_set)

    print()

    print(
        f"Saved {len(candidate_clusters)} candidates"
    )

    print()

    print("DONE")