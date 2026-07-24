import time
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from core.surfacePathToVector import  surfacePathToVector_cached
from data.io import select_clusters

def run(project, cluster_ids = None, cluster_names = None,
        mode='full', local_dist=15, n_global_samples=400, n_jobs=8, random_seed = 0):
   
    # SETTINGS
    model_type = project.model   
    random_seed = 0

    # PATHS
    parameter_map_dir = project.parameter_map_dir 

   
    # PARAMETER SELECTION
    if model_type == "N":
        parameter_name = "N"
        
    elif model_type == "VFM":
        parameter_name = "polar_angle"
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    
    # LOAD DATA
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

    clusters = select_clusters(clusters, cluster_ids, cluster_names)

    map_lh = nib.load(
        parameter_map_dir
        / f"lh.{parameter_name}.mgh"
    ).get_fdata().squeeze()

    map_rh = nib.load(
        parameter_map_dir
        / f"rh.{parameter_name}.mgh"
    ).get_fdata().squeeze()


   
    # UTILITIES
  
    def select_targets(cluster_vertices, distances,rng):
        valid_mask = np.isfinite(distances)
        valid_vertices = cluster_vertices[valid_mask]
        valid_distances = distances[valid_mask]

        if mode == "full":
            return valid_vertices

        local_mask = valid_distances <= local_dist
        local_vertices = valid_vertices[local_mask]
        far_vertices = valid_vertices[~local_mask]

        if len(far_vertices) > n_global_samples:
            sampled_far = rng.choice(
                far_vertices,
                size=n_global_samples,
                replace=False
            )

        else:
            sampled_far = far_vertices

        return np.concatenate([local_vertices, sampled_far])


    def process_vertex(
        i,
        source,
        cluster_vertices,
        map_lookup,
        geo,
        verts,
        normal
    ):
        
        rng = np.random.default_rng(
            random_seed + i
        )

        dist, pred = geo.distance_from_vertex(source, return_predecessors = True)

        start = verts[source]
        delta = verts - start         

        euclid = np.linalg.norm(delta, axis=1)
        direction_table = np.zeros_like(delta)
        valid = euclid > 0
        direction_table[valid] = delta[valid] / euclid[valid, None]

        cross = np.cross(direction_table, normal)
        cluster_dist = dist[cluster_vertices]

        targets = select_targets(cluster_vertices, cluster_dist, rng)

        vectors = []
        for target in targets:
            if target == source:
                continue

            d = dist[target]

            if d == 0:
                continue

            delta = (map_lookup[target] - map_lookup[source])

            if np.isnan(delta):
                continue
                
            path = geo.reconstruct_path(pred, source, target,)

            if path is None:
                continue

            direction = surfacePathToVector_cached(
                path,
                direction_table,
                cross,
                euclid
            )

            if direction is None:
                continue

            vec = (delta / d) * direction
            vectors.append(vec)
    
        if len(vectors) == 0:
            return i, None

        vectors = np.asarray(vectors, dtype=np.float32)

        mean_vec = np.nanmean(vectors, axis=0)

        return i, mean_vec


    def compute_cluster_directions(cluster, geo, full_map):

        cluster_vertices = np.asarray(cluster.valid_vertices, dtype=np.int32)
        map_values = full_map[cluster_vertices]
        valid_mask = ~np.isnan(map_values)
        cluster_vertices = cluster_vertices[valid_mask]
        map_values = map_values[valid_mask]

        normals_all = geo.compute_vertex_normals()
        normals = normals_all[cluster_vertices]

        map_lookup = np.full(full_map.shape, np.nan, dtype=np.float32)
        map_lookup[cluster_vertices] = map_values
        verts = geo.verts.astype(np.float32)

        results = Parallel(
            n_jobs=n_jobs,
            prefer="processes"
        )(
            delayed(process_vertex)(
                i=i,
                source=source,
                cluster_vertices=cluster_vertices,
                map_lookup=map_lookup,
                geo=geo,
                verts=verts,
                normal=normals[idx]
            )


            for idx, (i, source)
            in enumerate(
                tqdm(
                    list(
                        enumerate(
                            cluster_vertices
                        )
                    ),
                    total=len(
                        cluster_vertices
                    ),
                    desc=f"Cluster {cluster.id}"
                )
            )
        )

        
        total = {
            "dijkstra": 0,
            "target_selection": 0,
            "path": 0,
            "direction": 0,
            "vector": 0,
            "mean": 0,
        }

        n_paths = 0

        print(f'len: {len(cluster_vertices)}')

        mean_vectors = np.full((len(cluster_vertices), 3), np.nan, dtype=np.float32)

        for i, vec in results:
            if vec is not None:
                mean_vectors[i] = vec

        return mean_vectors


    # MAIN LOOP
    for cluster in clusters:
        print(f"\nCluster {cluster.id}")

        geo = project.geo[cluster.hemi]
            
        if cluster.hemi == "lh":
            full_map = map_lh

        else:
            full_map = map_rh

        mean_vectors = (compute_cluster_directions(cluster=cluster, geo=geo, full_map=full_map))
        cluster.directions["topographic"] = mean_vectors


    # SAVE
    project.save_clusters(cluster_set)
