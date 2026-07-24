import numpy as np


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cross_cross(v, n):
    cross1 = np.cross(v, n)
    v_tangent = np.cross(n, cross1)
    return normalize(v_tangent)


def vecangle360(v1, v2, normal):
    v1 = normalize(v1)
    v2 = normalize(v2)

    cross = np.cross(v1, v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

    angle = np.arctan2(np.dot(cross, normal), dot)
    angle = np.degrees(angle)

    if angle < 0:
        angle += 360

    return angle


def compare_topographic_vectors(vectors1, vectors2, normals, hemi):
    angles = []

    for i in range(len(vectors1)):

        v1 = vectors1[i]
        v2 = vectors2[i]
        n = normals[i]

        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            continue

        v1 = cross_cross(v1, n)
        v2 = cross_cross(v2, n)

        if hemi == "rh":
            angle = vecangle360(v1, v2, n)
        else:  
            angle = vecangle360(v2, v1, n)

        angles.append(angle)

    angles = np.array(angles)

    if len(angles) == 0:
        return angles, np.nan, np.nan, np.nan, 0

    angles_rad = np.deg2rad(angles)

    mean_angle = (np.rad2deg(np.angle(
                np.mean(np.exp(1j * angles_rad))))% 360)
    std_angle = np.rad2deg(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * angles_rad))))))

   

    return angles, mean_angle, std_angle, len(angles)