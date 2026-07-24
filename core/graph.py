import numpy as np
import nibabel as nib
import scipy.sparse

from scipy.sparse.csgraph import dijkstra
from joblib import Parallel, delayed


class GraphGeodesics:
    def __init__(self, surf_path):
        verts, faces = nib.freesurfer.read_geometry(surf_path)
        self.verts = np.asarray(verts,dtype=np.float64)
        self.faces = np.asarray(faces,dtype=np.int32)
        self.graph = self._build_graph()

        # Compute once
        self.vertex_area = self._compute_vertex_areas()

    def _build_graph(self):
        faces = self.faces
        verts = self.verts
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])

        edges = np.sort(edges,axis=1)
        edges = np.unique(edges,axis=0)
        lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]],axis=1)

        n = len(verts)
        A = scipy.sparse.coo_matrix((lengths,(edges[:, 0],edges[:, 1])), shape=(n, n))
        return (A + A.T).tocsr()

    def distance_from_vertex(self, source, return_predecessors=False):
        kwargs = {"indices": source, "return_predecessors": return_predecessors}
        result = dijkstra(self.graph, **kwargs)

        return result
    
    def roi_distance_matrix(self, cluster_vertices, max_dist=None, n_jobs=4):

        cluster_vertices = np.asarray(cluster_vertices, dtype=np.int32)
        def worker(v):
            if max_dist is None:
                dist = dijkstra(self.graph, indices=v)

            else:
                dist = dijkstra(self.graph, indices=v, limit=max_dist)

            return dist[cluster_vertices]

        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(worker)(v) for v in cluster_vertices)

        roi_dist = np.asarray(results,dtype=np.float32)

        return roi_dist


    def centroid_from_distance_matrix(self, cluster_vertices, roi_dist):

        scores = np.sum(roi_dist ** 2, axis=1)
        best_index = np.argmin(scores)

        return cluster_vertices[best_index]
    

    def compute_vertex_normals(self):
        normals = np.zeros_like(self.verts)
        for tri in self.faces:
            v0, v1, v2 = self.verts[tri]
            n = np.cross(v1 - v0, v2 - v0)
            normals[tri] += n

        normals /= (np.linalg.norm(normals, axis=1, keepdims=True,) + 1e-8)

        return normals
    
    def _compute_vertex_areas(self):
        v0 = self.verts[self.faces[:, 0]]
        v1 = self.verts[self.faces[:, 1]]
        v2 = self.verts[self.faces[:, 2]]

        face_areas = (np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0)

        vertex_area = np.zeros(len(self.verts), dtype=np.float64)

        third = face_areas / 3.0

        np.add.at(vertex_area, self.faces[:, 0], third)
        np.add.at(vertex_area, self.faces[:, 1], third)
        np.add.at(vertex_area, self.faces[:, 2], third)

        return vertex_area

    def surface_area(self, vertices):
        return float(self.vertex_area[vertices].sum())

    def reconstruct_path(self,predecessors,start,end):
        path = [end]
        current = end

        while current != start:
            current = predecessors[current]
            if current < 0:
                return None
            path.append(current)
        return path
    
