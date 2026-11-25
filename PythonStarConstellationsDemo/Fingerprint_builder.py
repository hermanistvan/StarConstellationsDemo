import os
import json
import numpy as np
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional

# Import your existing modules
from DataLoader import load_dataset_split
from Triangulation import extract_stars_and_triangulate
from ConstellationFingerprint import ReferenceTriangle, ConstellationFingerprint


def compute_triangle_descriptors(
    points: np.ndarray,
    brightnesses: np.ndarray,
    triangles: Optional[np.ndarray]
) -> List[Tuple[float, float, float, int]]:
    """
    Converts raw Delaunay triangles into scale-, rotation-, and translation-invariant 
    descriptors enriched with relative scale and vertex brightness ordering:
      - r1 = a / c  (shortest side / longest side)
      - r2 = b / c  (middle side / longest side)
      - s  = c / c_max (relative scale of this triangle compared to the largest triangle in the set)
      - brightest_opp_idx: index (0, 1, or 2) of the side (a, b, or c) opposite to the brightest vertex.
    """
    if triangles is None or len(triangles) == 0 or len(points) < 3:
        return []

    raw_items = []
    max_longest_side = 0.0

    for idx1, idx2, idx3 in triangles:
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        b1, b2, b3 = float(brightnesses[idx1]), float(brightnesses[idx2]), float(brightnesses[idx3])

        # Side lengths paired with their opposite vertex brightness:
        # d1 (between p1-p2) is opposite to vertex p3 (brightness b3)
        # d2 (between p2-p3) is opposite to vertex p1 (brightness b1)
        # d3 (between p3-p1) is opposite to vertex p2 (brightness b2)
        d1 = float(np.linalg.norm(p1 - p2))
        d2 = float(np.linalg.norm(p2 - p3))
        d3 = float(np.linalg.norm(p3 - p1))

        # Sort sides ascending: a <= b <= c, keeping track of opposite vertex brightness
        sides_with_opp_b = sorted(
            [(d1, b3), (d2, b1), (d3, b2)],
            key=lambda pair: pair[0]
        )

        (a, opp_b_a), (b, opp_b_b), (c, opp_b_c) = sides_with_opp_b

        if c > 1e-5:
            r1 = a / c
            r2 = b / c
            opp_brightnesses = [opp_b_a, opp_b_b, opp_b_c]
            brightest_opp_idx = int(np.argmax(opp_brightnesses))

            if c > max_longest_side:
                max_longest_side = c

            raw_items.append((r1, r2, c, brightest_opp_idx))

    if max_longest_side <= 1e-5:
        return []

    # Normalize longest side 'c' by 'max_longest_side' to obtain relative scale 's' in (0.0, 1.0]
    descriptors: List[Tuple[float, float, float, int]] = [
        (r1, r2, c / max_longest_side, brightest_opp_idx)
        for r1, r2, c, brightest_opp_idx in raw_items
    ]

    return descriptors


def _cluster_and_aggregate_descriptors(
    raw_descriptors: List[Tuple[float, float, float, int]],
    sample_count: int,
    bin_size_ratio: float = 0.035,
    bin_size_scale: float = 0.08,
    merge_radius: float = 0.055,
    top_k: int = 12
) -> List[ReferenceTriangle]:
    """
    Aggregates noisy 4D triangle descriptors (r1, r2, s, brightest_opp_idx) across training samples
    by grouping them into density bins (separated by brightness ordering), merging split boundary
    clusters, and computing weighted cluster centroids.
    """
    if not raw_descriptors or sample_count == 0:
        return []

    # 1. Group descriptors into grid cells keyed by (brightest_opp_idx, bin_r1, bin_r2, bin_s)
    bins: Dict[Tuple[int, int, int, int], List[Tuple[float, float, float]]] = {}
    for r1, r2, s, b_idx in raw_descriptors:
        bin_x = int(r1 / bin_size_ratio)
        bin_y = int(r2 / bin_size_ratio)
        bin_z = int(s / bin_size_scale)
        key = (b_idx, bin_x, bin_y, bin_z)
        if key not in bins:
            bins[key] = []
        bins[key].append((r1, r2, s))

    # 2. Calculate initial centroid and stability weight for each grid cell
    clusters: List[Tuple[int, ReferenceTriangle]] = []
    for (b_idx, _, _, _), pts in bins.items():
        count = len(pts)
        if count < 2 and sample_count >= 5:
            continue

        pts_arr = np.array(pts, dtype=np.float32)
        mean_r1 = float(np.mean(pts_arr[:, 0]))
        mean_r2 = float(np.mean(pts_arr[:, 1]))
        mean_s = float(np.mean(pts_arr[:, 2]))

        weight = min(1.0, count / float(sample_count))
        clusters.append((
            count,
            ReferenceTriangle(
                r1=round(mean_r1, 5),
                r2=round(mean_r2, 5),
                s=round(mean_s, 5),
                brightest_opp_idx=b_idx,
                weight=round(weight, 4)
            )
        ))

    # Sort initial clusters by frequency descending so dominant centers absorb smaller neighbors
    clusters.sort(key=lambda item: item[0], reverse=True)

    # 3. Merge adjacent clusters that share the same brightest_opp_idx and lie within merge_radius
    merged_clusters: List[Tuple[int, ReferenceTriangle]] = []
    for count, ref_tri in clusters:
        merged = False
        for idx, (m_count, m_tri) in enumerate(merged_clusters):
            if ref_tri.brightest_opp_idx != m_tri.brightest_opp_idx:
                continue

            # Weighted Euclidean distance in (r1, r2, 0.5 * s) space
            dist = float(np.sqrt(
                (ref_tri.r1 - m_tri.r1) ** 2 +
                (ref_tri.r2 - m_tri.r2) ** 2 +
                (0.5 * (ref_tri.s - m_tri.s)) ** 2
            ))
            if dist < merge_radius:
                total_count = m_count + count
                new_r1 = (m_tri.r1 * m_count + ref_tri.r1 * count) / total_count
                new_r2 = (m_tri.r2 * m_count + ref_tri.r2 * count) / total_count
                new_s = (m_tri.s * m_count + ref_tri.s * count) / total_count
                new_weight = min(1.0, total_count / float(sample_count))

                merged_clusters[idx] = (
                    total_count,
                    ReferenceTriangle(
                        r1=round(new_r1, 5),
                        r2=round(new_r2, 5),
                        s=round(new_s, 5),
                        brightest_opp_idx=m_tri.brightest_opp_idx,
                        weight=round(new_weight, 4)
                    )
                )
                merged = True
                break

        if not merged:
            merged_clusters.append((count, ref_tri))

    # 4. Sort final merged clusters by frequency descending and return the top K
    merged_clusters.sort(key=lambda item: item[0], reverse=True)
    return [ref_tri for _, ref_tri in merged_clusters[:top_k]]


def build_and_save_fingerprints(
    output_json_path: str = "constellation_fingerprints.json",
    max_stars_per_roi: int = 10,
    top_k_triangles: int = 12
) -> Dict[str, ConstellationFingerprint]:
    """
    Iterates over the 'train' split, extracts brightness- and scale-aware Delaunay triangles
    inside each constellation's bounding box, aggregates them per class, and saves to JSON.
    """
    train_dataset, class_names = load_dataset_split("train")

    collected_descriptors: Dict[str, List[Tuple[float, float, float, int]]] = {
        name: [] for name in class_names if name.lower() != "moon"
    }
    sample_counts: Dict[str, int] = {name: 0 for name in collected_descriptors}

    for item in train_dataset:
        if not item.annotations:
            continue

        valid_boxes = [b for b in item.annotations if b.class_name in collected_descriptors]
        if not valid_boxes:
            continue

        clean_image = item.image.load()
        img_height, img_width = clean_image.shape[:2]

        for box in valid_boxes:
            box_w = int(box.width * img_width)
            box_h = int(box.height * img_height)
            x_min = max(0, int((box.x_center * img_width) - (box_w / 2)))
            y_min = max(0, int((box.y_center * img_height) - (box_h / 2)))
            x_max = min(img_width, x_min + box_w)
            y_max = min(img_height, y_min + box_h)

            roi = clean_image[y_min:y_max, x_min:x_max]
            points, brightnesses, triangles = extract_stars_and_triangulate(roi, max_stars=max_stars_per_roi)

            descriptors = compute_triangle_descriptors(points, brightnesses, triangles)
            if descriptors:
                collected_descriptors[box.class_name].extend(descriptors)
                sample_counts[box.class_name] += 1

    fingerprints: Dict[str, ConstellationFingerprint] = {}
    for class_name, raw_list in collected_descriptors.items():
        count = sample_counts[class_name]
        ref_triangles = _cluster_and_aggregate_descriptors(
            raw_descriptors=raw_list,
            sample_count=count,
            top_k=top_k_triangles
        )
        fingerprints[class_name] = ConstellationFingerprint(
            class_name=class_name,
            sample_count=count,
            reference_triangles=ref_triangles
        )

    serializable_data = {
        name: {
            "sample_count": fp.sample_count,
            "reference_triangles": [asdict(tri) for tri in fp.reference_triangles]
        }
        for name, fp in fingerprints.items()
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

    return fingerprints


def load_fingerprints_from_json(json_path: str = "constellation_fingerprints.json") -> Dict[str, ConstellationFingerprint]:
    """
    Reads and parses the saved reference fingerprints from a JSON file back into memory objects.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fingerprint file not found at: {os.path.abspath(json_path)}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    fingerprints: Dict[str, ConstellationFingerprint] = {}
    for class_name, data in raw_data.items():
        ref_triangles = [
            ReferenceTriangle(
                r1=t["r1"],
                r2=t["r2"],
                s=t["s"],
                brightest_opp_idx=t["brightest_opp_idx"],
                weight=t["weight"]
            )
            for t in data.get("reference_triangles", [])
        ]
        fingerprints[class_name] = ConstellationFingerprint(
            class_name=class_name,
            sample_count=data.get("sample_count", 0),
            reference_triangles=ref_triangles
        )

    return fingerprints


if __name__ == "__main__":
    output_file = "constellation_fingerprints.json"
    print("Building enriched reference fingerprints (r1, r2, scale, brightness) from 'train' dataset...")
    
    generated_fps = build_and_save_fingerprints(output_json_path=output_file)
    print(f"Saved enriched fingerprints to '{output_file}'.\n")

    loaded_fps = load_fingerprints_from_json(output_file)
    for name, fp in loaded_fps.items():
        print(f"Constellation: {name.upper()} (Samples: {fp.sample_count}, Triangles: {len(fp.reference_triangles)})")
        for tri in fp.reference_triangles[:3]:
            print(f"   -> r1={tri.r1:.4f}, r2={tri.r2:.4f}, s={tri.s:.4f}, bright_opp={tri.brightest_opp_idx}, weight={tri.weight:.2f}")
