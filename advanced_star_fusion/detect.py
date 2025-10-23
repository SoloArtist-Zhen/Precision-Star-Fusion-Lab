
import numpy as np

def select_top_by_brightness(px, mags, topk=28):
    scores = 1.0/(mags+1e-6)
    order = np.argsort(-scores)
    return order[:topk]

def centroid_subpixel(px_noisy):
    return px_noisy.copy()
