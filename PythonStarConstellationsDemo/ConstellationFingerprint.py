import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class ReferenceTriangle:
    """
    Represents a stable, invariant triangle descriptor enriched with relative scale
    and vertex brightness ordering:
      - r1 = a / c (shortest / longest side)
      - r2 = b / c (middle / longest side)
      - s  = c / c_max (relative scale within the constellation)
      - brightest_opp_idx: index (0, 1, or 2) of the side opposite to the brightest vertex
    """
    r1: float
    r2: float
    s: float
    brightest_opp_idx: int
    weight: float


@dataclass
class ConstellationFingerprint:
    """Stores the aggregated reference fingerprint for a single constellation class."""
    class_name: str
    sample_count: int
    reference_triangles: List[ReferenceTriangle]


def compute_similarity_score(
    query_descriptors: List[Tuple[float, float, float, int]],
    reference_fp: ConstellationFingerprint,
    distance_tolerance: float = 0.065,
    top_n_backbone: int = 6
) -> Tuple[float, List[int]]:
    """
    Computes a normalized similarity score [0.0, 1.0] between an observed set of 
    enriched triangle descriptors (r1, r2, s, brightest_opp_idx) and a stored reference fingerprint.
    """
    if not query_descriptors or not reference_fp.reference_triangles:
        return 0.0, []

    backbone_triangles = reference_fp.reference_triangles[:top_n_backbone]
    total_possible_weight = sum(ref_tri.weight for ref_tri in backbone_triangles)

    if total_possible_weight <= 1e-6:
        return 0.0, []

    matched_weight_sum = 0.0
    matched_query_indices = set()

    for ref_tri in reference_fp.reference_triangles:
        best_dist = float("inf")
        best_q_idx = -1

        for q_idx, (q_r1, q_r2, q_s, q_b_idx) in enumerate(query_descriptors):
            # Penalize brightness ordering mismatch while allowing slight tolerance for close-brightness stars
            brightness_penalty = 0.0 if q_b_idx == ref_tri.brightest_opp_idx else 0.045

            # Weighted 3D Euclidean distance in (r1, r2, 0.5 * s) space + brightness penalty
            geom_dist = float(np.sqrt(
                (q_r1 - ref_tri.r1) ** 2 +
                (q_r2 - ref_tri.r2) ** 2 +
                (0.5 * (q_s - ref_tri.s)) ** 2
            ))
            total_dist = geom_dist + brightness_penalty

            if total_dist < best_dist:
                best_dist = total_dist
                best_q_idx = q_idx

        if best_dist < distance_tolerance and best_q_idx != -1:
            proximity_factor = 1.0 - (best_dist / distance_tolerance) ** 2
            matched_weight_sum += ref_tri.weight * proximity_factor
            matched_query_indices.add(best_q_idx)

    similarity_score = min(1.0, matched_weight_sum / total_possible_weight)
    return float(similarity_score), sorted(list(matched_query_indices))


def find_best_matching_constellation(
    query_descriptors: List[Tuple[float, float, float, int]],
    fingerprints: Dict[str, ConstellationFingerprint],
    distance_tolerance: float = 0.065,
    min_score_threshold: float = 0.20
) -> Optional[Tuple[str, float, List[int]]]:
    """
    Compares an observed set of enriched triangle descriptors against all known constellation
    fingerprints and returns the best match if it exceeds `min_score_threshold`.
    """
    if not query_descriptors or not fingerprints:
        return None

    best_name: Optional[str] = None
    best_score: float = -1.0
    best_indices: List[int] = []

    for name, fp in fingerprints.items():
        score, matched_indices = compute_similarity_score(
            query_descriptors=query_descriptors,
            reference_fp=fp,
            distance_tolerance=distance_tolerance
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_indices = matched_indices

    if best_name is not None and best_score >= min_score_threshold:
        return best_name, best_score, best_indices

    return None


# --- Evaluation & Testing Script (Runs ONLY when executed directly) ---
if __name__ == "__main__":
    from DataLoader import load_dataset_split
    from Triangulation import extract_stars_and_triangulate
    from Fingerprint_builder import load_fingerprints_from_json, compute_triangle_descriptors

    print("Loading enriched reference fingerprints and 'train' dataset...")
    fingerprints = load_fingerprints_from_json("constellation_fingerprints.json")
    train_dataset, _ = load_dataset_split("train")

    ref_names = sorted(list(fingerprints.keys()))

    score_records: Dict[str, Dict[str, List[float]]] = {
        true_cls: {ref_cls: [] for ref_cls in ref_names}
        for true_cls in ref_names
    }

    accuracy_stats: Dict[str, Dict[str, int]] = {
        cls_name: {"correct": 0, "total": 0} for cls_name in ref_names
    }

    for item in train_dataset:
        if not item.annotations:
            continue

        valid_boxes = [b for b in item.annotations if b.class_name in fingerprints]
        if not valid_boxes:
            continue

        clean_image = item.image.load()
        img_height, img_width = clean_image.shape[:2]

        for box in valid_boxes:
            true_cls = box.class_name

            box_w = int(box.width * img_width)
            box_h = int(box.height * img_height)
            x_min = max(0, int((box.x_center * img_width) - (box_w / 2)))
            y_min = max(0, int((box.y_center * img_height) - (box_h / 2)))
            x_max = min(img_width, x_min + box_w)
            y_max = min(img_height, y_min + box_h)

            roi = clean_image[y_min:y_max, x_min:x_max]
            points, brightnesses, triangles = extract_stars_and_triangulate(roi, max_stars=10)
            query_descriptors = compute_triangle_descriptors(points, brightnesses, triangles)

            if not query_descriptors:
                continue

            # 1. Compare against EVERY reference fingerprint for the average score matrix
            for ref_cls in ref_names:
                score, _ = compute_similarity_score(
                    query_descriptors=query_descriptors,
                    reference_fp=fingerprints[ref_cls]
                )
                score_records[true_cls][ref_cls].append(score)

            # 2. Test best-match identification
            accuracy_stats[true_cls]["total"] += 1
            best_match = find_best_matching_constellation(
                query_descriptors=query_descriptors,
                fingerprints=fingerprints
            )

            if best_match is not None:
                predicted_cls, _, _ = best_match
                if predicted_cls == true_cls:
                    accuracy_stats[true_cls]["correct"] += 1

    # --- PART 1: Print Average Similarity Score Matrix ---
    print("\n=== 1. AVERAGE SIMILARITY SCORE MATRIX (Rows: Ground Truth | Cols: Reference JSON) ===")
    col_width = 7
    row_label_width = 14

    header = " " * row_label_width + " | " + " ".join(f"{name[:6]:>{col_width}}" for name in ref_names)
    print(header)
    print("-" * len(header))

    for true_cls in ref_names:
        row_cells = []
        for ref_cls in ref_names:
            scores = score_records[true_cls][ref_cls]
            avg_score = float(np.mean(scores)) if scores else 0.0
            row_cells.append(f"{avg_score:>{col_width}.3f}")
        print(f"{true_cls:<{row_label_width}} | " + " ".join(row_cells))

    # --- PART 2: Print Identification Accuracy per Constellation ---
    print("\n=== 2. IDENTIFICATION ACCURACY (find_best_matching_constellation) ===")
    total_correct = 0
    total_samples = 0

    for cls_name in ref_names:
        correct = accuracy_stats[cls_name]["correct"]
        total = accuracy_stats[cls_name]["total"]
        total_correct += correct
        total_samples += total
        percentage = (correct / total * 100.0) if total > 0 else 0.0
        print(f"  - {cls_name:<15}: {percentage:6.2f}% ({correct}/{total})")

    overall_acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    print(f"\nOVERALL ACCURACY: {overall_acc:.2f}% ({total_correct}/{total_samples})")
