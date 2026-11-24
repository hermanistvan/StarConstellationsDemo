import cv2
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple

def extract_stars_and_triangulate(image: np.ndarray, threshold_val: int = 200, max_stars: int = 15):
    """
    Detects the brightest star centroids in an image and computes their Delaunay triangulation.
    """
    # 1. Convert to grayscale and reduce sensor noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thresholding to isolate bright stars
    _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # 3. Find contours and calculate centroids + brightness/area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    star_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1.0:  # Filter out single-pixel noise
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                star_candidates.append((cx, cy, area))

    # Keep only the brightest/largest N stars so background noise doesn't ruin the mesh
    star_candidates.sort(key=lambda s: s[2], reverse=True)
    top_stars = np.array([(s[0], s[1]) for s in star_candidates[:max_stars]], dtype=np.float32)

    # 4. Compute Delaunay triangulation (requires at least 3 non-collinear points)
    triangles = None
    if len(top_stars) >= 3:
        tri = Delaunay(top_stars)
        triangles = tri.simplices  # Indices of points forming each triangle

    return top_stars, triangles


def draw_delaunay_debug(image: np.ndarray, points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Draws the detected stars and the Delaunay triangulation mesh on a copy of the image."""
    vis_img = image.copy()

    if triangles is not None:
        for simplex in triangles:
            pt1 = tuple(points[simplex[0]].astype(int))
            pt2 = tuple(points[simplex[1]].astype(int))
            pt3 = tuple(points[simplex[2]].astype(int))
            
            # Draw triangle edges in green
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
            cv2.line(vis_img, pt2, pt3, (0, 255, 0), 1)
            cv2.line(vis_img, pt3, pt1, (0, 255, 0), 1)

    # Draw star centroids in red
    for pt in points:
        cv2.circle(vis_img, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

    return vis_img

