import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy.io import loadmat
from data.structures import VertexData



def run(project):
    settings = project.settings.processing
    method = settings.method
    max_dist = settings.max_dist
    model_type = project.model

    rm_path = project.input_dir / "responseModel.mat"
    coord_path = project.input_dir / "coords_tkrRAS.npz"

    #Load data
    voxel_coords = np.load(coord_path)
    coords = voxel_coords["tkrRAS"]

    rm = loadmat(rm_path)
    response_model = rm['model'][0][0]

    lh_verts= project.geo_lh.verts
    rh_verts= project.geo_rh.verts

    all_verts = np.vstack([lh_verts, rh_verts])
    V = len(all_verts)

    print("Number of vertices:", V)

    #Load parameters
    x0 = response_model['x0'][0][0].squeeze()
    y0 = response_model['y0'][0][0].squeeze()
    sigma = response_model['sigma'][0][0]
    sigma_major = sigma['major'][0][0].squeeze()
    sigma_minor = sigma['minor'][0][0].squeeze()
    rss = response_model['rss'][0][0].squeeze()
    raw_rss = response_model['rawrss'][0][0].squeeze()

    # Filter valid voxels
    if model_type == 'N':
        indices = [i for i, x in enumerate(x0) if x != 0]
    else:
        indices = [x for x in range(len(x0))]

    coords = coords[indices]
    x0 = x0[indices]
    y0 = y0[indices]
    sigma_major = sigma_major[indices]
    sigma_minor = sigma_minor[indices] 
    rss = rss[indices]
    raw_rss = raw_rss[indices]

    # Method A: Voxel → Vertex
    def run_method_A():
        print("\nRunning Method A (voxel → vertex)")

        surface_data = {
            "voxels": [list() for _ in range(V)],
            "dist": np.full(V, np.nan),
            "x0": np.full(V, np.nan),
            "y0": np.full(V, np.nan),
            "sigma_major": np.full(V, np.nan),
            "sigma_minor": np.full(V, np.nan),
            "rss": np.full(V, np.nan),
            "raw_rss": np.full(V, np.nan),
        }

        tree = cKDTree(all_verts)
        distances, nn_idx = tree.query(coords, k=1)

        voxels_per_vertex = defaultdict(list)
        for vox_idx, vtx_idx in enumerate(nn_idx):
            voxels_per_vertex[vtx_idx].append(vox_idx)

        for vtx, vox_list in voxels_per_vertex.items():
            vox_list = np.asarray(vox_list)

            d = distances[vox_list]
            good = d < max_dist
            if not np.any(good):
                continue

            v = vox_list[good]

            surface_data["voxels"][vtx] = v.tolist()
            surface_data["dist"][vtx] = distances[v].mean()
            surface_data["x0"][vtx] = x0[v].mean()
            surface_data["y0"][vtx] = y0[v].mean()
            surface_data["sigma_major"][vtx] = sigma_major[v].mean()
            surface_data["sigma_minor"][vtx] = sigma_minor[v].mean()
            surface_data["rss"][vtx] = rss[v].mean()
            surface_data["raw_rss"][vtx] = raw_rss[v].mean()

        print("Vertices with data:", np.sum(~np.isnan(surface_data["x0"])))
        return surface_data

    # Method B: Vertex -> Voxel
    def run_method_B():
        print("\nRunning Method B (vertex → voxel)")

        surface_data = {
            "voxels": np.full(V, np.nan),
            "dist": np.full(V, np.nan),
            "x0": np.full(V, np.nan),
            "y0": np.full(V, np.nan),
            "sigma_major": np.full(V, np.nan),
            "sigma_minor": np.full(V, np.nan),
            "rss": np.full(V, np.nan),
            "raw_rss": np.full(V, np.nan),
        }

        voxel_tree = cKDTree(coords)
        dist_v, vox_idx = voxel_tree.query(all_verts, k=1)

        for vtx in range(V):
            if dist_v[vtx] > max_dist:
                continue

            v = vox_idx[vtx]

            surface_data["voxels"][vtx] = indices[v]
            surface_data["dist"][vtx] = dist_v[vtx]
            surface_data["x0"][vtx] = x0[v]
            surface_data["y0"][vtx] = y0[v]
            surface_data["sigma_major"][vtx] = sigma_major[v]
            surface_data["sigma_minor"][vtx] = sigma_minor[v]
            surface_data["rss"][vtx] = rss[v]
            surface_data["raw_rss"][vtx] = raw_rss[v]

        print("Vertices with data:", np.sum(~np.isnan(surface_data["x0"])))
        return surface_data

    # Run method
    if method == "A":
        surface_data = run_method_A()
    elif method == "B":
        surface_data = run_method_B()
    else:
        raise ValueError("METHOD must be 'A' or 'B'")
    # Compute r2

    rss_arr = surface_data["rss"]
    raw_rss_arr = surface_data["raw_rss"]

    r2 = np.full_like(
        rss_arr,
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(rss_arr)
        & np.isfinite(raw_rss_arr)
        & (raw_rss_arr > 0)
    )

    r2[valid] = (
        1.0
        - (rss_arr[valid] / raw_rss_arr[valid])
    )

    # Build VertexData

    metrics = {

        "x0": surface_data["x0"],

        "y0": surface_data["y0"],

        "sigma_major": surface_data["sigma_major"],

        "sigma_minor": surface_data["sigma_minor"],

        "dist": surface_data["dist"],

        "rss": surface_data["rss"],

        "raw_rss": surface_data["raw_rss"],

        "r2": r2,
    }

    # VFM-specific derived metrics
    if model_type == "VFM":

        x0 = surface_data["x0"]
        y0 = surface_data["y0"]

        polar_angle = np.mod(
            np.arctan2(y0, x0),
            2 * np.pi
        )

        eccentricity = np.sqrt(
            x0**2 + y0**2
        )

        metrics["polar_angle"] = polar_angle

        metrics["eccentricity"] = eccentricity

    vertex_data = VertexData(
        coords=all_verts,

        voxel_index=surface_data["voxels"],

        metrics=metrics,

        metadata={
            "method": method,
            "max_dist": max_dist,
            "model_type": model_type,
            "subject": project.subject,
        }
    )

    project.save_vertex_data(vertex_data)


