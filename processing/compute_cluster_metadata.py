
from pathlib import Path
import os
import nibabel as nib
import numpy as np
from collections import defaultdict

def run(project):
    # PATHS
    data_dir = project.data_dir
    final_dir = project.final_dir
    
    # LOAD CLUSTERS
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

    labels_l, _, names_l = nib.freesurfer.read_annot(
        data_dir
        / "s1"
        / "label"
        / "lh.aparc.annot"
    )

    labels_r, _, names_r = nib.freesurfer.read_annot(
        data_dir
        / "s1"
        / "label"
        / "rh.aparc.annot"
    )

    
    #HELPERS
    def assign_cluster_names(clusters, parameter):

        groups = defaultdict(list)
        for c in clusters:
            groups[(c.hemi, c.metadata["parcel"])].append(c)

        for (hemi, parcel), group in groups.items():
            coords = np.array([project.geo[c.hemi].verts[c.centroid] for c in group])
            labels = recursive_labels(coords)
            
            for c, label in zip(group, labels):
                if label:
                    c.name = (f"{hemi}_{parameter}_{parcel.lower()}_{label}")
                else:
                    c.name = (f"{hemi}_{parameter}_{parcel.lower()}")

    def axis_labels(axis):
        if axis == 0:
            return ["L", "C", "M"]

        elif axis == 1:
            return ["P", "C", "A"]

        else:
            return ["I", "C", "S"]

    def recursive_labels(coords, axes=None):
        n = len(coords)
        if n == 1:
            return [""]

        spread = coords.max(axis=0) - coords.min(axis=0)

        if axes is not None:
            spread[axes] = -1

        axis = np.argmax(spread)
        base = axis_labels(axis)

        if n == 2:
            order = np.argsort(coords[:, axis])
            labels = [""] * n
            labels[order[0]] = base[0]
            labels[order[1]] = base[-1]
            return labels

        if n == 3:
            order = np.argsort(coords[:, axis])
            labels = [""] * n
            labels[order[0]] = base[0]
            labels[order[1]] = base[1]
            labels[order[2]] = base[2]

            return labels

        # n >= 4
        order = np.argsort(coords[:, axis])

        half = n // 2

        idx1 = order[:half]
        idx2 = order[half:]

        sub1 = recursive_labels(coords[idx1], axes=[axis])
        sub2 = recursive_labels(coords[idx2], axes=[axis])

        labels = [""] * n
        for i, lab in zip(idx1, sub1):
            labels[i] = base[0] + lab

        for i, lab in zip(idx2, sub2):
            labels[i] = base[-1] + lab

        return labels

    # Add Metadata
    vertex_data = project.vertex_data

    for cluster in clusters:
        geo = project.geo[cluster.hemi]

        if cluster.hemi == "lh":
            labels = labels_l
            names = names_l

        else:
            labels = labels_r
            names = names_r


        roi_dist = geo.roi_distance_matrix(cluster.valid_vertices)
        centroid_vertex = (geo.centroid_from_distance_matrix(cluster.valid_vertices, roi_dist))

        cluster.centroid = int(centroid_vertex)
        cluster.metadata["surface_area"] = geo.surface_area(cluster.valid_vertices)
    

        label_index = labels[centroid_vertex]
        parcel = names[label_index].decode()
        cluster.metadata["parcel"] = parcel
        cluster.metadata["voxels"] = vertex_data.voxel_index[cluster.valid_vertices]


    assign_cluster_names(clusters, project.model)

    for cluster in clusters:

        old_label = Path(cluster.metadata["label_file"])
        new_label = (final_dir / f"{cluster.name}.label")
        os.rename(old_label, new_label)

        cluster.metadata["label_file"] = str(new_label)


    # Save Clusters
    project.save_clusters(cluster_set)


