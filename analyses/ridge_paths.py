import os
import heapq
import numpy as np
import nibabel as nib
from collections import defaultdict


def run(project):
    #PATHS
    settings = project.settings.processing
    concavity_thr = settings.concavity_threshold
    depth_threshold_sulcus = settings.depth_threshold_sulcus
    depth_threshold_gyrus = settings.depth_threshold_gyrus

    data_dir = project.data_dir
    surf_dir = project.surf_dir
    anatomy_dir = project.anatomy_dir
    label_dir = data_dir / 's1' / 'label'

    (anatomy_dir / "ridge").mkdir(parents=True, exist_ok=True)
    (anatomy_dir / "fundus").mkdir(parents=True, exist_ok=True)
  
    # Helpers
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    def edge_weight(u, v):
        d = edge_length[(u, v)]

        depth_term = np.exp(-6 * depth[v])
        concavity_term = np.exp(-2.0 * concavity[v])

        return d * depth_term * concavity_term 

    def dijkstra(start, valid_set):
        dist = {start: 0}
        prev = {}
        heap = [(0, start)]

        while heap:
            curr_dist, u = heapq.heappop(heap)

            if curr_dist > dist[u]:
                continue

            for v in neighbors[u]:
                if v not in valid_set:
                    continue  

                w = edge_weight(u, v)
                new_dist = curr_dist + w

                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return dist, prev

    def find_longest_path(valid_set):

        indices = list(valid_set)
        if len(indices) < 10:
            return None, None, None

        seed = indices[0]

        dist1, _ = dijkstra(seed, valid_set)
        A = max(dist1, key=dist1.get)

        dist2, prev = dijkstra(A, valid_set)
        B = max(dist2, key=dist2.get)

        return A, B, prev

    def extract_path(prev, start, end):
        path = []
        v = end

        while v != start:
            path.append(v)
            v = prev.get(v)
            if v is None:
                return []

        path.append(start)
        return path[::-1]
    
  
    # Main
    for hemi in ["lh", "rh"]:
        curv_path = os.path.join(surf_dir, f"{hemi}.curv")
        sulc_path = os.path.join(surf_dir, f"{hemi}.sulc")
        annot_path = os.path.join(label_dir, f"{hemi}.aparc.a2009s.annot")

        curv = nib.freesurfer.read_morph_data(curv_path)
        sulc = nib.freesurfer.read_morph_data(sulc_path)
        labels, _, names = nib.freesurfer.read_annot(annot_path)
        names = [n.decode("utf-8") for n in names]
        
        geo = project.geo[hemi]
        coords = geo.verts
        faces = geo.faces

        neighbors = defaultdict(list)
        edge_length = {}

        for tri in faces:
            for i in range(3):
                u = tri[i]
                v = tri[(i + 1) % 3]

                neighbors[u].append(v)
                neighbors[v].append(u)

                d = np.linalg.norm(coords[u] - coords[v])
                edge_length[(u, v)] = d
                edge_length[(v, u)] = d


        for i in range(2):
            depth = normalize((-1)**i * sulc)        # deeper = higher 
            concavity = normalize((-1) ** (i+1) *curv)   # concave = higher

            for fold_type, path_type, depth_thr in [
                (
                    "S_",
                    "fundus",
                    depth_threshold_sulcus
                ),

                (
                    "G_",
                    "ridge",
                    depth_threshold_gyrus
                )
            ]:
                
                for label_id, region_name in enumerate(names):
                    if not region_name.startswith(fold_type) and not region_name.startswith('G_and_S'):
                        continue

                    region_mask = labels == label_id
                    
                    if np.sum(region_mask) < 100:
                        continue

                    valid = region_mask & (depth > depth_thr) & (concavity > concavity_thr)
                    valid_set = set(np.where(valid)[0]) 

                    if len(valid_set) < 20:
                        continue

                    print(f"\n[Region] {region_name} ({len(valid_set)} valid verts)")

                    start, end, prev = find_longest_path(valid_set)

                    if start is None:
                        print("skipped")
                        continue

                    path = extract_path(prev, start, end)

                    if len(path) < 10:
                        continue

                    print(f"path length: {len(path)}")

                    out_path = os.path.join(f"{anatomy_dir}/{path_type}", f"{hemi}.{region_name}.{path_type}.label")

                    with open(out_path, "w") as f:
                        f.write("#!ascii label, from fundus extraction\n")
                        f.write(f"{len(path)}\n")

                        for i in path:
                            x, y, z = coords[i]
                            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} 1.0\n")
