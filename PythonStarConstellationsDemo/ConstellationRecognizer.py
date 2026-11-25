import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

# Import your existing modules
from DataLoader import load_dataset_split, BoundingBox
from Triangulation import extract_stars_and_triangulate
from Fingerprint_builder import load_fingerprints_from_json, compute_triangle_descriptors
from ConstellationFingerprint import ConstellationFingerprint, find_best_matching_constellation


@dataclass
class DetectedConstellation:
    """Represents a constellation found live on an unannotated sky image."""
    class_name: str
    score: float
    center_x: float
    center_y: float
    matched_triangles: np.ndarray  # Subset of Delaunay simplices forming this constellation
    all_points: np.ndarray         # Reference to all detected star points in the image


def _filter_long_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    max_edge_px: float = 185.0
) -> np.ndarray:
    """
    Removes Delaunay triangles that span unrealistically large distances across the sky,
    keeping only compact, local constellation-scale triangles.
    """
    valid_triangles = []
    for v1, v2, v3 in triangles:
        p1, p2, p3 = points[v1], points[v2], points[v3]
        d1 = float(np.linalg.norm(p1 - p2))
        d2 = float(np.linalg.norm(p2 - p3))
        d3 = float(np.linalg.norm(p3 - p1))
        if max(d1, d2, d3) <= max_edge_px:
            valid_triangles.append([v1, v2, v3])

    if not valid_triangles:
        return np.empty((0, 3), dtype=triangles.dtype)
    return np.array(valid_triangles, dtype=triangles.dtype)


def _get_local_triangle_neighborhood(
    anchor_star_idx: int,
    triangles: np.ndarray,
    hops: int = 1
) -> np.ndarray:
    """
    Extracts a compact local sub-mesh of Delaunay triangles around a given anchor star.
    """
    selected_star_indices: Set[int] = {anchor_star_idx}
    selected_tri_indices: Set[int] = set()

    for _ in range(hops):
        new_stars: Set[int] = set()
        for tri_idx, (v1, v2, v3) in enumerate(triangles):
            if (v1 in selected_star_indices or 
                v2 in selected_star_indices or 
                v3 in selected_star_indices):
                selected_tri_indices.add(tri_idx)
                new_stars.update((int(v1), int(v2), int(v3)))
        selected_star_indices.update(new_stars)

    if not selected_tri_indices:
        return np.empty((0, 3), dtype=triangles.dtype)

    return triangles[sorted(list(selected_tri_indices))]


def recognize_constellations_in_image(
    image: np.ndarray,
    fingerprints: Dict[str, ConstellationFingerprint],
    max_global_stars: int = 75,
    min_score_threshold: float = 0.22
) -> Tuple[List[DetectedConstellation], np.ndarray, Optional[np.ndarray]]:
    """
    Performs blind constellation recognition on a full sky image without bounding boxes,
    enforcing spatial scale limits so distant unrelated stars are not grouped together.
    """
    points, brightnesses, raw_global_triangles = extract_stars_and_triangulate(
        roi_image=image,
        max_stars=max_global_stars
    )

    if raw_global_triangles is None or len(points) < 4:
        return [], points, raw_global_triangles

    # 1. Prune unrealistically long edges spanning across different constellations
    global_triangles = _filter_long_triangles(points, raw_global_triangles, max_edge_px=185.0)
    if len(global_triangles) < 3:
        return [], points, global_triangles

    raw_detections: List[DetectedConstellation] = []

    # 2. Slide a compact 1-hop neighborhood window over every detected star
    for star_idx in range(len(points)):
        local_triangles = _get_local_triangle_neighborhood(
            anchor_star_idx=star_idx,
            triangles=global_triangles,
            hops=1
        )

        if len(local_triangles) < 2:
            continue

        local_descriptors = compute_triangle_descriptors(
            points=points,
            brightnesses=brightnesses,
            triangles=local_triangles
        )

        match = find_best_matching_constellation(
            query_descriptors=local_descriptors,
            fingerprints=fingerprints,
            distance_tolerance=0.065,
            min_score_threshold=min_score_threshold
        )

        if match is not None:
            class_name, score, matched_local_indices = match
            # Require at least 3 distinct triangles to confirm a constellation pattern
            if len(matched_local_indices) < 2:
                continue

            matched_simplices = local_triangles[matched_local_indices]
            unique_star_ids = np.unique(matched_simplices)
            matched_coords = points[unique_star_ids]
            cx = float(np.mean(matched_coords[:, 0]))
            cy = float(np.mean(matched_coords[:, 1]))

            raw_detections.append(
                DetectedConstellation(
                    class_name=class_name,
                    score=score,
                    center_x=cx,
                    center_y=cy,
                    matched_triangles=matched_simplices,
                    all_points=points
                )
            )

    final_detections = _suppress_overlapping_detections(raw_detections, min_distance_px=120.0)
    return final_detections, points, global_triangles



def _suppress_overlapping_detections(
    detections: List[DetectedConstellation],
    min_distance_px: float = 130.0
) -> List[DetectedConstellation]:
    """
    Non-Maximum Suppression (NMS): keeps only the highest-scoring detection 
    when multiple neighboring anchor stars trigger nearby.
    """
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept_detections: List[DetectedConstellation] = []

    for candidate in sorted_dets:
        is_duplicate = False
        for existing in kept_detections:
            dist = float(np.hypot(candidate.center_x - existing.center_x, candidate.center_y - existing.center_y))
            if (candidate.class_name == existing.class_name and dist < min_distance_px) or (dist < min_distance_px * 0.6):
                is_duplicate = True
                break
        if not is_duplicate:
            kept_detections.append(candidate)

    return kept_detections


def render_recognition_results(
    image: np.ndarray,
    detections: List[DetectedConstellation],
    all_points: np.ndarray,
    ground_truth_boxes: List[BoundingBox]
) -> np.ndarray:
    """
    Renders both Ground Truth annotations (Magenta) and Predicted constellations (Cyan/Green)
    for clear visual comparison.
    """
    vis_img = image.copy()
    img_height, img_width = vis_img.shape[:2]

    # 1. Draw all background detected stars as small dim red dots
    for pt in all_points:
        cv2.circle(vis_img, tuple(pt.astype(int)), 2, (0, 0, 180), -1)

    # 2. Draw Ground Truth bounding boxes in MAGENTA (purple-pink)
    gt_color = (220, 50, 220)
    for box in ground_truth_boxes:
        if box.class_name.lower() == "moon":
            continue
        box_w = int(box.width * img_width)
        box_h = int(box.height * img_height)
        x_min = max(0, int((box.x_center * img_width) - (box_w / 2)))
        y_min = max(0, int((box.y_center * img_height) - (box_h / 2)))
        x_max = min(img_width, x_min + box_w)
        y_max = min(img_height, y_min + box_h)

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), gt_color, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis_img,
            f"GT: {box.class_name.upper()}",
            (x_min, min(img_height - 8, y_max + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            gt_color,
            1,
            lineType=cv2.LINE_AA
        )

    # 3. Draw Predicted constellations in GREEN (triangles) and CYAN (bounding box & label)
    pred_box_color = (255, 220, 0)  # Cyan-blue in BGR
    for det in detections:
        for idx1, idx2, idx3 in det.matched_triangles:
            pt1 = tuple(det.all_points[idx1].astype(int))
            pt2 = tuple(det.all_points[idx2].astype(int))
            pt3 = tuple(det.all_points[idx3].astype(int))

            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(vis_img, pt2, pt3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(vis_img, pt3, pt1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        unique_ids = np.unique(det.matched_triangles)
        constellation_pts = det.all_points[unique_ids]
        for pt in constellation_pts:
            cv2.circle(vis_img, tuple(pt.astype(int)), 4, (0, 220, 255), -1)

        x_min = max(0, int(np.min(constellation_pts[:, 0])) - 10)
        y_min = max(0, int(np.min(constellation_pts[:, 1])) - 10)
        x_max = min(img_width, int(np.max(constellation_pts[:, 0])) + 10)
        y_max = min(img_height, int(np.max(constellation_pts[:, 1])) + 10)

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), pred_box_color, 2, lineType=cv2.LINE_AA)
        label = f"PRED: {det.class_name.upper()} ({det.score:.2f})"
        cv2.putText(
            vis_img,
            label,
            (max(5, x_min), max(18, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            pred_box_color,
            1,
            lineType=cv2.LINE_AA
        )

    return vis_img


# --- Two-Phase Live Testing on 'test' Split (Runs ONLY when executed directly) ---
if __name__ == "__main__":
    print("Loading fingerprints and 'test' dataset...\n")
    fingerprints = load_fingerprints_from_json("constellation_fingerprints.json")
    test_dataset, _ = load_dataset_split("test")

    # Store results from Phase 1 so we don't recompute during Phase 2 visualization
    cached_results = []

    stats_per_class: Dict[str, Dict[str, int]] = {
        name: {"gt_total": 0, "true_positives": 0, "false_positives": 0}
        for name in sorted(list(fingerprints.keys()))
    }

    print(f"=== PHASE 1: EVALUATING ALL {len(test_dataset)} TEST IMAGES (HEADLESS) ===")
    for idx, item in enumerate(test_dataset):
        sky_image = item.image.load()
        img_h, img_w = sky_image.shape[:2]

        detections, all_stars, _ = recognize_constellations_in_image(
            image=sky_image,
            fingerprints=fingerprints,
            max_global_stars=75,
            min_score_threshold=0.22
        )

        cached_results.append((idx, item, sky_image, detections, all_stars))

        valid_gt_boxes = [b for b in item.annotations if b.class_name in stats_per_class]
        matched_gt_indices = set()

        for gt_box in valid_gt_boxes:
            stats_per_class[gt_box.class_name]["gt_total"] += 1

        # Match predictions against ground truth boxes (check class match + spatial proximity)
        for det in detections:
            if det.class_name not in stats_per_class:
                continue

            found_spatial_match = False
            for gt_idx, gt_box in enumerate(valid_gt_boxes):
                if gt_idx in matched_gt_indices:
                    continue
                if gt_box.class_name == det.class_name:
                    gt_cx = gt_box.x_center * img_w
                    gt_cy = gt_box.y_center * img_h
                    # Accept if predicted center is within 150px of ground truth center
                    if np.hypot(det.center_x - gt_cx, det.center_y - gt_cy) < 150.0:
                        stats_per_class[det.class_name]["true_positives"] += 1
                        matched_gt_indices.add(gt_idx)
                        found_spatial_match = True
                        break

            if not found_spatial_match:
                stats_per_class[det.class_name]["false_positives"] += 1

    # Print formatted statistics table
    print("\n=== BLIND TEST SET RECOGNITION STATISTICS ===")
    print(f"{'CONSTELLATION':<15} | {'RECALL (FOUND/GT)':<18} | {'FALSE POSITIVES':<15}")
    print("-" * 55)

    total_gt = 0
    total_tp = 0
    total_fp = 0

    for name, counts in stats_per_class.items():
        gt = counts["gt_total"]
        tp = counts["true_positives"]
        fp = counts["false_positives"]
        total_gt += gt
        total_tp += tp
        total_fp += fp
        recall_pct = (tp / gt * 100.0) if gt > 0 else 0.0
        print(f"{name:<15} | {tp:>3}/{gt:<3} ({recall_pct:5.1f}%)   | {fp:>5}")

    overall_recall = (total_tp / total_gt * 100.0) if total_gt > 0 else 0.0
    print("-" * 55)
    print(f"{'TOTAL':<15} | {total_tp:>3}/{total_gt:<3} ({overall_recall:5.1f}%)   | {total_fp:>5}\n")

    # --- PHASE 2: INTERACTIVE VISUALIZATION ---
    print("=== PHASE 2: INTERACTIVE VISUALIZATION ===")
    print("Legend: MAGENTA = Ground Truth | CYAN/GREEN = Predicted")
    print("Focus the image window and press ANY key for next image, or ESC to quit.\n")

    window_name = "Blind Constellation Recognition (MAGENTA: GT | CYAN: Pred)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx, item, sky_image, detections, all_stars in cached_results:
        gt_names = [b.class_name for b in item.annotations if b.class_name.lower() != "moon"]
        pred_names = [f"{d.class_name} ({d.score:.2f})" for d in detections]

        print(f"--- Image [{idx + 1}/{len(cached_results)}] ---")
        print(f"  Ground Truth : {', '.join(gt_names) if gt_names else 'None'}")
        print(f"  Predicted    : {', '.join(pred_names) if pred_names else 'None'}\n")

        vis_image = render_recognition_results(
            image=sky_image,
            detections=detections,
            all_points=all_stars,
            ground_truth_boxes=item.annotations
        )
        cv2.imshow(window_name, vis_image)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting visualization.")
            break

    cv2.destroyAllWindows()
