import numpy as np
from pathlib import Path
from core.surfacePathToVector import surfacePathToVector
from scipy.sparse.csgraph import dijkstra
from data.io import select_clusters
from analyses import ridge_paths

def run(project,cluster_ids = None , cluster_names = None, ridge_name = "G_precentral"):
    #PATHS
    final_dir = project.final_dir
    ridge_dir = project.anatomy_dir / "ridge"

    #LOAD
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

    if not any(ridge_dir.glob("*.label")):ridge_paths.run(project)

    #Helpers
    def extract_ridge_name(filepath):
        stem = Path(filepath).stem
        parts = stem.split(".")
        return parts[1]

    def load_label_vertices(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()[2:]
            return np.array([int(line.split()[0]) for line in lines], dtype=np.int32)
        
   
    def classify_ridge_cluster_interaction(ridge_vertices, cluster_vertices):
        cluster_set = set(cluster_vertices)
        inside = np.isin(ridge_vertices,list(cluster_set))

        if np.any(inside):
            if inside[0] or inside[-1]:
                return "ends_inside"
            return "crosses"
        
        return "outside"
        
    #------------------------------------------------------
    def detect_crossing_segments(ridge_vertices, cluster_vertices):

        cluster_set = set(cluster_vertices)
        inside = np.isin(ridge_vertices, list(cluster_set))

        segments = []
        in_segment = False

        for i, is_inside in enumerate(inside):
            # entering cluster
            if is_inside and not in_segment:
                entry_idx = i
                in_segment = True

            # leaving cluster
            elif not is_inside and in_segment:
                exit_idx = i - 1
                segments.append({"entry_idx": entry_idx, "exit_idx": exit_idx,})

                in_segment = False

        if len(segments) > 0:
            segments = {"entry_idx": segments[0]["entry_idx"], "exit_idx": segments[-1]["exit_idx"]}

        return segments

    def wrap_ridge_around_cluster(ridge_vertices, cluster_vertices,coords, geo):
        seg = detect_crossing_segments(ridge_vertices,cluster_vertices)
    
        graph = geo.graph
        graph = graph.copy().tolil()

        for v in cluster_vertices:

            # remove outgoing edges
            graph[v, :] = 0

            # remove incoming edges
            graph[:, v] = 0

        new_graph = graph.tocsr()  

        reroutes = []
    
        start_vertex = ridge_vertices[seg["entry_idx"] - 1]

        dist, pred = dijkstra(new_graph, indices=start_vertex, return_predecessors=True)
        
        candidate_indices = np.arange(seg["exit_idx"], min(len(ridge_vertices), seg["exit_idx"] + 30))
        candidate_targets = ridge_vertices[candidate_indices]

        best_local_idx = np.argmin(dist[candidate_targets])
        best_target = candidate_targets[best_local_idx]
        best_target_idx = candidate_indices[best_local_idx]
        
        reroute_path = geo.reconstruct_path(pred,start_vertex,best_target)

        reroute_path = reroute_path[::-1]  #Flip the path to be correct within the ridge

        reroutes.append({
            "entry_idx": seg["entry_idx"],
            "best_target_idx": best_target_idx,
            "reroute_path": reroute_path,
            })
        
        wrapped_ridge = []

        current_idx = 0
        if len(reroutes) == 0:
            return ridge_vertices
        
        for reroute in reroutes:
            wrapped_ridge.extend(ridge_vertices[current_idx : reroute["entry_idx"]])
            wrapped_ridge.extend(reroute["reroute_path"])
            current_idx = reroute["best_target_idx"]

        wrapped_ridge.extend(ridge_vertices[current_idx:])
        wrapped_ridge = np.array(wrapped_ridge,dtype=np.int32)
        wrapped_ridge = np.array([v for i, v in enumerate(wrapped_ridge) if i == 0 or v != wrapped_ridge[i - 1]])

        return wrapped_ridge

    
    def compute_cluster_ridge_directions(hemi):
        geo = project.geo[hemi]
        coords = geo.verts
        normals = geo.compute_vertex_normals()
        
        hemi_clusters = [c for c in clusters if c.hemi == hemi]
            
        if hemi == "lh":
            ridge_labels = ridge_labels_lh
        else:
            ridge_labels = ridge_labels_rh


        for cluster in hemi_clusters:
            print(f"Determining direction for {cluster.name}")
            current_ridge_name = ridge_name
            ridge_vertices = ridge_labels[current_ridge_name].copy()
            label_file = final_dir / f"{cluster.name}.label"

            if not label_file.exists():
                print(f"Missing label file: {label_file}")
                continue

            cluster_vertices_dilated = load_label_vertices(label_file)
            cluster_vertices = cluster.valid_vertices
            interaction = classify_ridge_cluster_interaction(ridge_vertices, cluster_vertices_dilated)

            if interaction == "ends_inside":
                ridge_vertices = trim_ridge_end(
                    ridge_vertices,
                    cluster_vertices_dilated,
                    margin=5
                )

                if len(ridge_vertices) < 2:
                    print(f"Could not trim ridge for {cluster.name}")
                    continue

            elif interaction == 'crosses':
                ridge_vertices = wrap_ridge_around_cluster(ridge_vertices, cluster_vertices_dilated, coords, geo)

            directions = np.full((len(cluster_vertices), 3),np.nan)
            for i, v in enumerate(cluster_vertices):
                dist, pred = geo.distance_from_vertex(v, return_predecessors=True)
                ridge_dist = dist[ridge_vertices]
                nearest_idx = np.argmin(ridge_dist)
                nearest_vertex = ridge_vertices[nearest_idx]
                path = geo.reconstruct_path(pred, v, nearest_vertex)

                direction = surfacePathToVector(path, coords, normals[v])
                directions[i] = direction
            cluster.directions['gyrus'] = directions
            cluster.metadata["reference_gyrus"] = ridge_name
            
    def trim_ridge_end(ridge_vertices, cluster_vertices, margin=15):
        cluster_set = set(cluster_vertices)

        inside = np.isin(ridge_vertices, list(cluster_set))

        if not np.any(inside):
            return ridge_vertices

        # Trim from the beginning
        if inside[0]:
            last_inside = np.where(inside)[0][-1]
            start = min(last_inside + margin + 1, len(ridge_vertices) - 1)
            return ridge_vertices[start:]

        # Trim from the end
        if inside[-1]:
            first_inside = np.where(inside)[0][0]
            end = max(first_inside - margin, 1)
            return ridge_vertices[:end]

        return ridge_vertices

  
    # Main
    clusters = select_clusters(clusters, cluster_ids, cluster_names)
   
    ridge_files = list(ridge_dir.glob("*.label"))

    ridge_labels_lh = {}
    ridge_labels_rh = {}

    for f in ridge_files:
        verts = load_label_vertices(f)
        name = extract_ridge_name(f)

        if f.name.startswith("lh."):
            ridge_labels_lh[name] = verts
        elif f.name.startswith("rh."):
            ridge_labels_rh[name] = verts
    
    for hemi in ["lh", "rh"]:
        compute_cluster_ridge_directions(hemi)
    project.save_clusters(cluster_set)

