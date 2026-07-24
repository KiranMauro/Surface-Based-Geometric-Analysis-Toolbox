import numpy as np


def surfacePathToVector(path, coords, normal):
    if len(path) == 2:

        p0 = coords[path[-1]]
        p1 = coords[path[0]]

        v = p1 - p0

        norm = np.linalg.norm(v)

        if norm == 0:
            return None

        v = v / norm

        return v
    
    n_steps = len(path) - 1
    if n_steps <= 0:
        return None

    angle_cross = np.full(n_steps, np.nan)
    distances = np.zeros(n_steps)
    step_vec = np.zeros((3, n_steps))

    start = path[-1]
    first_cross = None

    for idx, step in enumerate(range(len(path) - 1, 0, -1)):

        p = path[step]

        ai = coords[p, 0] - coords[start, 0]  
        bj = coords[p, 1] - coords[start, 1] 
        ck = coords[p, 2] - coords[start, 2] 

        d = (ai*ai + bj*bj + ck*ck) ** 0.5
        if d == 0:
            continue

        distances[idx] = d  
        step_vec[:, idx] = np.array([ai/d, bj/d, ck/d])

        cx = step_vec[1, idx]*normal[2] - step_vec[2, idx]*normal[1]
        cy = step_vec[2, idx]*normal[0] - step_vec[0, idx]*normal[2]
        cz = step_vec[0, idx]*normal[1] - step_vec[1, idx]*normal[0]

        crossproduct = np.array([cx, cy, cz])

        if first_cross is None:
            first_cross = crossproduct
            first_cross_norm = np.linalg.norm(first_cross)

        cross_norm = (cx*cx + cy*cy + cz*cz) ** 0.5

        denom = first_cross_norm * cross_norm
        if denom == 0:
            continue

        dot = (first_cross[0]*crossproduct[0] + 
               first_cross[1]*crossproduct[1] + 
               first_cross[2]*crossproduct[2]
               )

        cos_angle = dot / denom
        cos_angle = np.clip(cos_angle, -1, 1)

        angle_cross[idx] = np.degrees(np.arccos(cos_angle))

    valid = np.where(angle_cross < 90)[0]
    if len(valid) == 0:
        return None

    best_idx = valid[np.argmax(distances[valid])]

    return step_vec[:, best_idx]


def surfacePathToVector_cached(path,directions,crosses,euclid):

    if len(path) < 2:
        return None

    start = path[-1]

    first_cross = None
    best_dist = -1.0
    best_vec = None

    for step in range(len(path)-1, 0, -1):

        p = path[step]
        d = euclid[p]
        if d == 0:
            continue

        cross = crosses[p]
        if first_cross is None:
            first_cross = cross
            if np.all(first_cross == 0):
                continue

        if np.dot(first_cross, cross) > 0:

            if d > best_dist:

                best_dist = d
                best_vec = directions[p]

    return best_vec