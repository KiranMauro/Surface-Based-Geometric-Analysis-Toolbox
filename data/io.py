import numpy as np

def save_vertex_data(vertex_data, path):
    np.save(path, vertex_data, allow_pickle=True)


def load_vertex_data(path):
    return np.load(path, allow_pickle=True).item()

def save_clusters(clusters, path):
    np.save(path, clusters, allow_pickle=True)


def load_clusters(path):
    return np.load(path, allow_pickle=True).item()

def save_parameter_map(parameter_map, path):
    np.save(path, parameter_map, allow_pickle=True)

def load_parameter_map(path):
    return np.load(path, allow_pickle=True).item()

def save_label(vertices, coords, outfile):
    with open(outfile, "w") as f:
        f.write("#!ascii label\n")
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            x, y, z = coords[v]
            f.write(f"{v} {x} {y} {z} 0\n")

def select_clusters(clusters, cluster_ids=None, cluster_names=None,):
    if cluster_ids is not None:
        clusters = [c for c in clusters if c.id in cluster_ids]

    if cluster_names is not None:
        clusters = [c for c in clusters if c.name in cluster_names]

    return clusters

