import numpy as np


def get_detected_points(x, xr=20, max_num_pts=25, pct_thresh=85):
    thresh = np.percentile(np.ravel(x), pct_thresh)
    pts = np.nonzero(x > thresh)
    pts = map(tuple, zip(*pts))
    pts = sorted(pts, key=x.__getitem__, reverse=True)
    detected_points = []
    for pt in pts:
        if len(detected_points) > max_num_pts:
            break
        pta = np.array(pt)[::-1]
        valid = True
        for dp in detected_points:
            if np.linalg.norm(pta - dp) < xr:
                valid = False
                break
        if valid:
            detected_points.append(pta)
    detected_points = np.array(detected_points)
    return detected_points


def maxf(C, max_pts=100, min_dist=10, mode='reflect', cval=0.):
    '''Return coordinates of 8-neighborhood locally maximal points in C'''
    MF = ndimage.maximum_filter(C, size=(3, 3))
    maxima_coords = (MF == C).nonzero()
    return maxima_coords
