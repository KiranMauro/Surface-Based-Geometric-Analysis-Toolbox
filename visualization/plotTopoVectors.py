import numpy as np
import nibabel as nib
import pyvista as pv
import colorsys
from pathlib import Path
from matplotlib.colors import ListedColormap


def run(project):
    surf_dir = project.surf_dir
    parameter_map_dir = project.parameter_map_dir
       
    lh_map_path = parameter_map_dir / "lh.N_display.mgh"
    rh_map_path = parameter_map_dir / "rh.N_display.mgh"

    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters


    map_lh = nib.load(
        lh_map_path
    ).get_fdata().squeeze()

    map_rh = nib.load(
        rh_map_path
    ).get_fdata().squeeze()


    lh_white, lh_faces = nib.freesurfer.read_geometry(
        surf_dir / "lh.white"
    )

    rh_white, rh_faces = nib.freesurfer.read_geometry(
        surf_dir / "rh.white"
    )

    lh_inflated, _ = nib.freesurfer.read_geometry(
        surf_dir / "lh.inflated"
    )

    rh_inflated, _ = nib.freesurfer.read_geometry(
        surf_dir / "rh.inflated"
    )

    geo_lh = project.geo_lh
    geo_rh = project.geo_rh

    normals_lh_all = geo_lh.compute_vertex_normals()
    normals_rh_all = geo_rh.compute_vertex_normals()
 

    USE_INFLATED = True

    HSV_LOW = 0.001
    HSV_HIGH = 8
    DISPLAY_MAX = 6.5


    def build_truncated_hsv_cmap(
        low=HSV_LOW,
        high=HSV_HIGH,
        cutoff=DISPLAY_MAX,
        n=256,
    ):
        vals = np.linspace(low, cutoff, n)
        colors = []
        for v in vals:

            hue = (v - low) / (high - low)
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(rgb)

        return ListedColormap(colors)
   
    def load_label_vertices(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()[2:]
            return np.array([int(line.split()[0]) for line in lines], dtype=np.int32)


    def plot_cluster_arrows(
        verts,
        vectors,
        gyrus_vectors,
        normals,
        values,
        surface,
        ridge_points,
        cluster_name="cluster",
        subsample=10,
        arrow_scale=2.0,
    ):

        idx = np.arange(
            0,
            len(verts),
            subsample
        )

        verts = verts[idx]
        vectors = vectors[idx]
        gyrus_vectors = gyrus_vectors[idx]
        normals = normals[idx]
        values = values[idx]


        vectors = (
            vectors
            - (vectors * normals).sum(
                axis=1,
                keepdims=True
            ) * normals
        )

        
        mesh = pv.PolyData(verts)
        mesh["vectors"] = vectors
        mesh["values"] = values
        mesh["magnitude"] = (np.linalg.norm(vectors, axis=1)* 100)

        arrows = mesh.glyph(
            orient="vectors",
            scale=False,
            factor=arrow_scale
        )

        gyrus_vectors = (
            gyrus_vectors
            - (gyrus_vectors * normals).sum(
                axis=1,
                keepdims=True
            ) * normals
        )

        gyrus_mag = np.linalg.norm(
            gyrus_vectors,
            axis=1,
            keepdims=True
        )

        gyrus_dirs = (
            gyrus_vectors
            / (gyrus_mag + 1e-8)
        )

        offset = 0
        verts_plot = verts + offset * normals
        mesh2 = pv.PolyData(verts_plot)
        mesh2["vectors"] = gyrus_dirs
        arrows2 = mesh2.glyph(
            orient="vectors",
            scale=False,
            factor=arrow_scale * 0.7
        )

        plotter = pv.Plotter()
        plotter.add_mesh(
            surface,
            color="lightgray",
            opacity=0,
            smooth_shading=True,
        )

        spline = pv.Spline(
            ridge_points,
            len(ridge_points) * 10
        )

        plotter.add_mesh(
            spline,
            color="red",
            line_width=10,
        )

        plotter.add_title(cluster_name)

        cmap = build_truncated_hsv_cmap()

        mean_actor = plotter.add_mesh(
            arrows,
            scalars="values",
            cmap=cmap,
            clim=[HSV_LOW, DISPLAY_MAX],
            show_scalar_bar=False,
            scalar_bar_args={
                "title": "Preferred Numerosity",
                "vertical": False,
                "position_x": 0.2,
                "position_y": 0.02,
                "width": 0.6,
                "height": 0.08,
                "n_labels": 7,
                "fmt": "%.0f",
            }
        )


        gyrus_actor = plotter.add_mesh(
            arrows2,
            color="black"
        )

        
        def toggle_mean(flag):

            mean_actor.SetVisibility(flag)

        def toggle_gyrus(flag):

            gyrus_actor.SetVisibility(flag)

        
        plotter.add_checkbox_button_widget(
            toggle_mean,
            value=True,
            position=(10, 10),
            size=25,
            color_on="green",
            color_off="gray",
        )

        plotter.add_text(
            "Topographic Direction",
            position=(45, 10),
            font_size=10
        )

        plotter.add_checkbox_button_widget(
            toggle_gyrus,
            value=True,
            position=(10, 50),
            size=25,
            color_on="green",
            color_off="gray",
        )

        plotter.add_text(
            "Ridge Direction",
            position=(45, 50),
            font_size=10
        )

        
        plotter.add_key_event(
            "t",
            lambda: mean_actor.SetVisibility(
                not mean_actor.GetVisibility()
            )
        )

        plotter.add_key_event(
            "g",
            lambda: gyrus_actor.SetVisibility(
                not gyrus_actor.GetVisibility()
            )
        )

        print("\nKeyboard shortcuts:")
        print("  t = toggle topographic direction")
        print("  g = toggle gyrus direction")


        plotter.show()

    ridge = False
    for cluster in clusters:

        print(f"\nPlotting cluster: {cluster.name}")

        if cluster.hemi == "lh":

            verts_vis = lh_inflated if USE_INFLATED else lh_white
            faces = lh_faces

            normals_all = normals_lh_all
            full_map = map_lh

        else:

            verts_vis = rh_inflated if USE_INFLATED else rh_white
            faces = rh_faces

            normals_all = normals_rh_all
            full_map = map_rh

        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)

        surface = pv.PolyData(
            verts_vis,
            faces_pv.ravel()
        )

        cluster_vertices = np.asarray(
            cluster.valid_vertices,
            dtype=int
        )
        ridge_name = cluster.metadata["reference_gyrus"]

        label = project.anatomy_dir / "ridge" / f"{cluster.hemi}.{ridge_name}.ridge.label"

        ridge_vertices = load_label_vertices(label)

        ridge_points = verts_vis[ridge_vertices]

       

        map_vals = full_map[cluster_vertices]

        valid = ~np.isnan(map_vals)

        cluster_vertices = cluster_vertices[valid]
        map_vals = map_vals[valid]

        verts_cluster = verts_vis[cluster_vertices]

        normals_cluster = normals_all[cluster_vertices]

        vecs_cluster = cluster.directions['topographic'][valid]

        gyrus_vecs = cluster.directions['gyrus'][valid]

        
        plot_cluster_arrows(
            verts=verts_cluster,
            vectors=vecs_cluster,
            gyrus_vectors=gyrus_vecs,
            normals=normals_cluster,
            values=map_vals,
            surface=surface,
            ridge_points=ridge_points,
            cluster_name=(
                cluster.name
                + (
                    " (inflated)"
                    if USE_INFLATED
                    else " (white)"
                )
            ),
            subsample=1,
            arrow_scale=2,
        )