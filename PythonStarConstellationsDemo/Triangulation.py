import cv2
import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, Optional

from DataLoader import load_dataset_split


def extract_stars_and_triangulate(
    roi_image: np.ndarray, 
    max_stars: int = 10
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Detects star centroids and their relative brightness in a clean ROI using 
    Morphological Top-Hat and a noise-floor threshold, then computes Delaunay triangulation.

    Returns:
        (top_stars, star_brightnesses, triangles)
    """
    if roi_image.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), None

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Morphological Top-Hat isolates point-like stars and suppresses background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Use a consistent threshold above the median noise floor
    bg_median = float(np.median(tophat))
    thresh_val = bg_median + 18.0

    _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    star_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out large non-star artifacts
        if w > 15 or h > 15:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        
        star_patch = tophat[y:y+h, x:x+w]
        # Integrated flux (sum of intensities) gives a much more accurate brightness ranking than single peak
        integrated_brightness = float(np.sum(star_patch)) if star_patch.size > 0 else 0.0
        
        star_candidates.append((integrated_brightness, cx, cy))

    # Rank by brightness descending and keep top N stars
    star_candidates.sort(reverse=True)
    
    if not star_candidates:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), None

    selected = star_candidates[:max_stars]
    top_stars = np.array([(cx, cy) for _, cx, cy in selected], dtype=np.float32)
    
    raw_brightnesses = np.array([b for b, _, _ in selected], dtype=np.float32)
    max_b = float(np.max(raw_brightnesses))
    star_brightnesses = raw_brightnesses / max_b if max_b > 0 else raw_brightnesses

    triangles = None
    if len(top_stars) >= 3:
        try:
            tri = Delaunay(top_stars)
            triangles = tri.simplices
        except Exception:
            triangles = None

    return top_stars, star_brightnesses, triangles


def draw_delaunay_on_full_image(
    full_image: np.ndarray,
    offset_x: int,
    offset_y: int,
    local_points: np.ndarray,
    triangles: Optional[np.ndarray],
    box_coords: Tuple[int, int, int, int],
    class_name: str
) -> None:
    """
    Draws the detected star centroids and Delaunay triangles first,
    then overlays a thin, subtle bounding box and constellation label.
    """
    if local_points.size == 0 or local_points.ndim != 2:
        return

    global_points = local_points + np.array([offset_x, offset_y], dtype=np.float32)

    if triangles is not None:
        for idx1, idx2, idx3 in triangles:
            pt1 = tuple(global_points[idx1].astype(int))
            pt2 = tuple(global_points[idx2].astype(int))
            pt3 = tuple(global_points[idx3].astype(int))
            
            cv2.line(full_image, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.line(full_image, pt2, pt3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.line(full_image, pt3, pt1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    for pt in global_points:
        cv2.circle(full_image, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

    x_min, y_min, x_max, y_max = box_coords
    subtle_color = (140, 100, 70)

    cv2.rectangle(full_image, (x_min, y_min), (x_max, y_max), subtle_color, 1, lineType=cv2.LINE_AA)
    cv2.putText(
        full_image,
        class_name.upper(),
        (x_min, max(15, y_min - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        subtle_color,
        1,
        lineType=cv2.LINE_AA
    )


if __name__ == "__main__":
    train_dataset, _ = load_dataset_split("train")
    
    window_name = "Constellation Delaunay Triangulation (Press ANY key for next, ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for item in train_dataset:
        if not item.annotations:
            continue

        display_image = item.image.load()
        clean_image = display_image.copy()
        img_height, img_width = display_image.shape[:2]

        for box in item.annotations:
            if box.class_name.lower() == "moon":
                continue

            box_w = int(box.width * img_width)
            box_h = int(box.height * img_height)
            x_min = max(0, int((box.x_center * img_width) - (box_w / 2)))
            y_min = max(0, int((box.y_center * img_height) - (box_h / 2)))
            x_max = min(img_width, x_min + box_w)
            y_max = min(img_height, y_min + box_h)

            roi = clean_image[y_min:y_max, x_min:x_max]
            points, _, triangles = extract_stars_and_triangulate(roi, max_stars=10)

            draw_delaunay_on_full_image(
                full_image=display_image,
                offset_x=x_min,
                offset_y=y_min,
                local_points=points,
                triangles=triangles,
                box_coords=(x_min, y_min, x_max, y_max),
                class_name=box.class_name
            )

        cv2.imshow(window_name, display_image)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
