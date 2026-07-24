import glob
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from data.structures import (Cluster, ClusterSet)
from data.io import save_label, save_clusters

def run(project):
    #PATHS
    roi_dir = project.data_dir / "manual_rois"
    coords_path = project.input_dir / "coords.mat"
    label_dir = project.label_dir / "manual_rois"
    label_dir.mkdir(parents=True, exist_ok=True)

    cluster_out = project.output_dir / "data_structures" / "manual_rois.npy"

    #LOAD
    vertex_data = project.vertex_data
    mat = loadmat(coords_path)

    coords_all = (mat["coords"].T.astype(np.int32))

    coord_lookup = {tuple(coord): i for i, coord in enumerate(coords_all)}
    cluster_set = ClusterSet()
    manual_clusters = []

    roi_files = sorted(glob.glob(str(roi_dir / "*.mat")))

    for cluster_id, roi_file in enumerate(roi_files, start=1):
        
        mat = loadmat(roi_file)
        coords = (mat["ROI"][0][0][2] .T .astype(np.int32))
        voxel_indices = []
        coords_set = {tuple(c) for c in coords_all}
        matches = sum(tuple(c) in coords_set for c in coords)

        print(f"{matches} / {len(coords)} exact matches")

        for coord in coords:
            idx = coord_lookup.get(tuple(coord))

            if idx is not None:
                voxel_indices.append(idx)

        voxel_indices = np.asarray(voxel_indices, dtype=np.int32)
        vertices = np.where(np.isin(vertex_data.voxel_index, voxel_indices))[0]

        n_lh = len(project.geo_lh.verts)

        lh_vertices = vertices[vertices < n_lh]
        rh_vertices = vertices[vertices >= n_lh] - n_lh
        
        if len(lh_vertices):
            hemi = "lh"
            verts = lh_vertices
            geo = project.geo_lh

        elif len(rh_vertices):
            hemi = "rh"
            verts = rh_vertices
            geo = project.geo_rh

        else:
            continue


        cluster = Cluster(
            id=cluster_id,
            subject=project.subject,
            hemi=hemi,
            valid_vertices=verts,
            label_vertices=verts.copy(),
        )

        cluster.name = (Path(roi_file).stem)
        roi_dist = geo.roi_distance_matrix(verts)
        centroid = geo.centroid_from_distance_matrix(verts, roi_dist)
        cluster.metadata["surface_area"] = (geo.surface_area(verts))
        cluster.centroid = int(centroid)

        label_path = label_dir / f"{cluster.name}.label"
        save_label(verts, geo.verts, label_path)

        cluster.metadata["label_file"] = str(label_path)

        manual_clusters.append(cluster)

    cluster_set.final_clusters = manual_clusters

    save_clusters(cluster_set, cluster_out)