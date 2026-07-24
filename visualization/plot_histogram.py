import numpy as np
import matplotlib.pyplot as plt
from core.compareTopographicVectors import compare_topographic_vectors



def run(project):
    #LOAD CLUSTERS
    cluster_set = project.cluster_set
    clusters = cluster_set.final_clusters

   
    # HELPERS
    def circular_mean_vector(vectors):
        mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            return None

        return mean / norm
    
    def ridge_angle_from_view(vector):
        # Project into the lateral-view plane (YZ)
        yz = vector.copy()
        yz[0] = 0

        norm = np.linalg.norm(yz)
        if norm < 1e-8:
            return 0.0

        yz /= norm

        # 0°   = anterior (-Y)
        # 90°  = superior (+Z)
        # 180° = posterior (+Y)
        # 270° = inferior (-Z)

        angle = np.degrees(np.arctan2(yz[2], -yz[1])) % 360
        return angle
    
    def plot_polar_histogram(angles, mean_angle, ridge_angle, title=""):
        angles = angles[~np.isnan(angles)]
        if len(angles) == 0:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")

        angles_rad = np.deg2rad(angles)
        counts, bins, patches = ax.hist(angles_rad, bins=30)
        ax.set_theta_offset(np.deg2rad(ridge_angle))
        mean_rad = np.deg2rad(mean_angle)
        rmax = ax.get_rmax()
        

        ax.plot(
            [mean_rad, mean_rad],
            [0, rmax],
            color="orange",
            linewidth=2,
            zorder=100
        )

        ax.set_title(
            f"{title}\nMean={mean_angle:.1f}°, N={len(angles)}"
        )

        print("Gridlines:", len(ax.yaxis.get_gridlines()))
        for g in ax.yaxis.get_gridlines():
            print(g.get_visible(), g.get_alpha(), g.get_zorder())

        plt.show(block=True)
        plt.close(fig)

   
    # Main
    for cluster in clusters:

        geo = project.geo[cluster.hemi]
        normals = geo.compute_vertex_normals()

        print(f"\nProcessing: {cluster.name}")

        if cluster.directions['topographic'] is None:
            print("Skipping: mean_direction is None")
            continue

        if cluster.directions['gyrus'] is None:
            print("Skipping: gyrus_direction is None")
            continue
    
        vectors1 = []
        vectors2 = []
        normals_list = []
  
        cluster_vertices = np.asarray(cluster.valid_vertices, dtype=int)
        topo = cluster.directions.get("topographic")
        gyrus = cluster.directions.get("gyrus")

        if topo.shape != gyrus.shape:
            print("Shape mismatch")
            continue
        
        for i, v in enumerate(cluster_vertices):
            v1 = topo[i]
            v2 = gyrus[i]

            if np.any(np.isnan(v1)):
                continue

            if np.any(np.isnan(v2)):
                continue

            vectors1.append(v1)
            vectors2.append(v2)

            normals_list.append(normals[v])

    
        vectors1 = np.array(vectors1)
        vectors2 = np.array(vectors2)
        normals_list = np.array(normals_list)

        gyrus_proj = vectors2.copy()
        gyrus_proj[:, 0] = 0

        reference = circular_mean_vector(gyrus_proj)
        print(reference)
        
        ridge_angle = ridge_angle_from_view(reference)
        print(ridge_angle)
        if len(vectors1) == 0:
            print("No valid vectors")
            continue

    
        angles, mean_ang, std_ang, n = compare_topographic_vectors(
            vectors1,
            vectors2,
            normals_list,
            hemi=cluster.hemi
        )

    
        print(f"\n{cluster.name}")
        print("Mean:", mean_ang)
        print("Std:", std_ang)
        print("N:", n)

        print(cluster.name)
        print("ridge_angle:", ridge_angle)
        print("mean_angle:", mean_ang)

        print("reference:", reference)
        print("norm reference:", np.linalg.norm(reference))

        assert np.isfinite(ridge_angle), ridge_angle
        assert np.isfinite(mean_ang), mean_ang


        plt.figure(figsize=(4,4))

        plt.axhline(0, color="gray")
        plt.axvline(0, color="gray")

    
        plt.quiver(
            0, 0,
            -reference[1],   # anterior is right
            reference[2],    # superior is up
            angles='xy',
            scale_units='xy',
            scale=1,
            color='red'
        )

        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
        plt.gca().set_aspect("equal")

        plt.xlabel("Posterior ←   → Anterior")
        plt.ylabel("Inferior ↓   ↑ Superior")
        plt.show()
        plot_polar_histogram(
            angles,
            mean_ang,
            ridge_angle,
            title=f"{cluster.name}: Topo vs Ridge"
        )

