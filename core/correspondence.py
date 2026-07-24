import numpy as np
from scipy.optimize import linear_sum_assignment


def dice_similarity(a, b):
    a = set(a)
    b = set(b)

    inter = len(a & b)
    if len(a) + len(b) == 0:
        return 0.0

    return (2 * inter/(len(a) + len(b)))


def cluster_similarity(
    cluster1,
    cluster2,
    geo,
    vertex_set="fsaverage_vertices",
    centroid_attr="fsaverage_centroid",
    sigma=20,
):

    verts1 = getattr(cluster1,vertex_set)
    verts2 = getattr(cluster2,vertex_set)

    dice = dice_similarity(verts1,verts2)

    centroid1 = getattr(cluster1, centroid_attr)
    centroid2 = getattr(cluster2, centroid_attr)
    
    dist = geo.distance_from_vertex(centroid1)[centroid2]
    similarity = (dice*np.exp(-dist / sigma))

    return {
        "dice": float(dice),
        "centroid_distance": float(dist),
        "similarity": float(similarity),
    }


def build_similarity_matrix(
    reference_clusters,
    target_clusters,
    geo,
    vertex_set="fsaverage_vertices",
    centroid_attr="fsaverage_centroid", 
    sigma=20,
):

    similarity = np.zeros((len(reference_clusters),len(target_clusters)))
    dice = np.zeros_like(similarity)
    distance = np.zeros_like(similarity)

    for i, c1 in enumerate(reference_clusters):
        for j, c2 in enumerate(target_clusters):
            result = cluster_similarity(
                c1,
                c2,
                geo,
                vertex_set=vertex_set,
                centroid_attr=centroid_attr,
                sigma=sigma,
            )

            similarity[i, j] = (result["similarity"])
            dice[i, j] = (result["dice"])
            distance[i, j] = (result["centroid_distance"])

    return (similarity, dice, distance)


def match_clusters(similarity):

    cost = 1 - similarity
    return linear_sum_assignment(cost)


def compute_cluster_matches(
    reference_clusters,
    target_clusters,
    geo,
    vertex_set="fsaverage_vertices",
    centroid_attr="fsaverage_centroid",
    sigma=20,
    similarity_threshold=0.1
):

    (similarity, dice, distance) = build_similarity_matrix(
        reference_clusters,
        target_clusters,
        geo,
        vertex_set=vertex_set,
        centroid_attr=centroid_attr,
        sigma=sigma,
    )

    rows, cols = match_clusters(similarity)
    matches = []

    for r, c in zip(rows, cols):
        if similarity[r, c] < similarity_threshold:
            continue

        matches.append({
            "cluster1":
                reference_clusters[r],
            "cluster2":
                target_clusters[c],
            "similarity":
                similarity[r, c],
            "dice":
                dice[r, c],
            "centroid_distance":
                distance[r, c],
        })

    print(f"\nReturning {len(matches)} matches")

    for m in matches:
        print(
            f"{m['cluster1'].name} -> "
            f"{m['cluster2'].name} "
            f"(similarity={m['similarity']:.3f}, "
            f"dice={m['dice']:.3f}, "
            f"distance={m['centroid_distance']:.1f} mm)"
        )

    return matches