import os
import subprocess
import nibabel as nib
import numpy as np
from core.graph import GraphGeodesics
from scipy.spatial import cKDTree

def run(project):
    # Paths
    data_dir = project.data_dir
    fsavg_dir = data_dir / "fsaverage_results"
    fsavg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(project.data_dir)

    # Load
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters


    # Load fsaverage geometry
    verts_fsavg_lh, faces = nib.freesurfer.read_geometry(
        f"{data_dir}/fsaverage/surf/lh.white"
    )

    verts_fsavg_rh, faces = nib.freesurfer.read_geometry(
        f"{data_dir}/fsaverage/surf/rh.white"
    )

    geo_fsavg_lh = GraphGeodesics(
        f"{data_dir}/fsaverage/surf/lh.white"
    )

    geo_fsavg_rh = GraphGeodesics(
        f"{data_dir}/fsaverage/surf/rh.white"
    )

    surfaces = {
        "lh": len(nib.freesurfer.read_geometry(f"{data_dir}/s1/surf/lh.white")[0]),
        "rh": len(nib.freesurfer.read_geometry(f"{data_dir}/s1/surf/rh.white")[0])
    }

    FSAVG = {
        "lh": {
            "coords": verts_fsavg_lh,
            "geo": geo_fsavg_lh,
        },

        "rh": {
            "coords": verts_fsavg_rh,
            "geo": geo_fsavg_rh,
        }

    }

   
    # Helpers
    def label_to_map(vertices, n_vertices):

        surf_map = np.zeros(n_vertices, dtype=np.float32)
        surf_map[vertices] = 1
        return surf_map


    def save_mgh(data, outfile):
        img = nib.MGHImage(data[:, None, None], affine=np.eye(4))
        nib.save(img, outfile)


    def save_label(vertices, coords, outfile):
        with open(outfile, "w") as f:
            f.write("#!ascii label\n")
            f.write(f"{len(vertices)}\n")

            for v in vertices:
                x, y, z = coords[v]
                f.write(f"{v} {x} {y} {z} 0\n")

    
    # Mapping to fsaverage
    def map_to_fsaverage(in_file, out_file, hemi):

        cmd = [
            "mri_surf2surf",
            "--srcsubject", "s1",
            "--trgsubject", "fsaverage",
            "--hemi", hemi,
            "--sval", str(in_file),
            "--tval", str(out_file),
            "--mapmethod", "nnf"
        ]

        subprocess.run(cmd, check=True, env=env)


    def process_cluster(cluster):

        n_vertices = surfaces[cluster.hemi]
        name = cluster.name
        temp_map = fsavg_dir / f"{name}.mgh"
        out_map = fsavg_dir / f"{name}_fsavg.mgh"
        surf_map = label_to_map(cluster.label_vertices, n_vertices)

        save_mgh(surf_map, temp_map)

        map_to_fsaverage(temp_map, out_map, cluster.hemi)

        data = nib.load(out_map).get_fdata().squeeze()
        vertices = np.where(data > 0.5)[0]

        cluster.fsaverage_vertices = vertices.astype(np.int32)

        geo = FSAVG[cluster.hemi]["geo"]
        coords = FSAVG[cluster.hemi]["coords"]

        roi = geo.roi_distance_matrix(cluster.fsaverage_vertices)

        cluster.fsaverage_centroid = geo.centroid_from_distance_matrix(cluster.fsaverage_vertices, roi)


        label_file = (fsavg_dir / f"{cluster.subject}_{cluster.name}.label")

        save_label(cluster.fsaverage_vertices, coords, label_file)

        temp_map.unlink(missing_ok=True)
        out_map.unlink(missing_ok=True)

        print("Mapped:", name)


    # Run transform
    for cluster in clusters:
        process_cluster(cluster)

    print("\nAll clusters mapped to fsaverage\n")

    # Surface mapping
 
    def compute_surface_mapping(hemi):
        fsavg_sphere = (f"{data_dir}/fsaverage/surf/{hemi}.sphere.reg")
        subj_sphere = (f"{data_dir}/s1/surf/{hemi}.sphere.reg")
        fsavg_verts, _ = nib.freesurfer.read_geometry(fsavg_sphere)
        subj_verts, _ = nib.freesurfer.read_geometry(subj_sphere)

        tree = cKDTree(subj_verts)
        dist, idx = tree.query(fsavg_verts)

        return idx

   
    # Save fsaverage ↔ subject mapping
    lh_map = compute_surface_mapping("lh")
    rh_map = compute_surface_mapping("rh")

    np.save(fsavg_dir / f"lh_fsavg_to_{project.subject}.npy", lh_map)
    np.save(fsavg_dir / f"rh_fsavg_to_{project.subject}.npy", rh_map)

    project.save_clusters(cluster_set)

