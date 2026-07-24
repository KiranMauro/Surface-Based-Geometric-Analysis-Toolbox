import colorsys
import numpy as np

def hsv_to_rgb255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return int(r*255), int(g*255), int(b*255)


def hsv_overlay_string(low=1.1, high=8, n=64):
    vals = np.linspace(low, high, n)
    rows = []

    for v in vals:
        hue = 360 * (v - low) / (high - low)
        r, g, b = hsv_to_rgb255(hue, 1.0, 1.0)
        rows.append(f"{v:.6f},{r},{g},{b}")

    return ",".join(rows)

def polar_angle_cmap(low=0.0, high=2 * np.pi,n=128):
    vals = np.linspace(low, high, n)
    rows = []
    for v in vals:
        hue = (360*(v - low)/(high - low))
        r, g, b = hsv_to_rgb255(hue, 1.0, 1.0)
        rows.append(f"{v:.6f},{r},{g},{b}")
    return ",".join(rows)

def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def r2_scale(low=0.00001, high=1.0, n=64):
    anchors = [
        (0.0,  (0, 0, 255)),    
        (0.1,  (50, 0, 200)),   
        (0.2, (100, 0, 150)),   
        (0.3,  (150, 0, 100)),     
        (0.4,  (200, 0, 50)),   
        (0.5, (255, 0, 0)), 
        (0.6,  (255, 50, 0)),    
        (0.7,  (255, 100, 0)),   
        (0.8, (255, 150, 0)), 
        (0.9,  (255, 200, 0)),   
        (1,  (255, 255, 0)),   
    ]

    vals = np.linspace(low, high, n)
    rows = []

    for v in vals:
        t = (v - low) / (high - low)

        for i in range(len(anchors) - 1):
            t0, c0 = anchors[i]
            t1, c1 = anchors[i + 1]
            if t0 <= t <= t1:
                local_t = (t - t0) / (t1 - t0)
                r, g, b = lerp(c0, c1, local_t)
                rows.append(f"{v:.6f},{r},{g},{b}")
                break

    return ",".join(rows)


