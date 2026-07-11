"""Deterministic, writing-system-agnostic glyph segmentation and measurement.

The pipeline intentionally calls its output *candidates*: connected writing and
ligatures cannot always be split correctly without a script-specific model.  Such
regions are retained (never silently discarded) and receive uncertainty metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


FEATURE_VERSION = "glyph-geometry-v1"


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return round(value, 8) if math.isfinite(value) else 0.0
    return value


def _rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO-style column-major run-length encoding (counts start with zeros)."""
    pixels = (mask > 0).astype(np.uint8).flatten(order="F")
    counts, previous, run = [], 0, 0
    for pixel in pixels:
        pixel = int(pixel)
        if pixel != previous:
            counts.append(run)
            run, previous = 0, pixel
        run += 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


@dataclass(frozen=True)
class GlyphPipelineConfig:
    min_component_area_ratio: float = 0.00002
    max_component_area_ratio: float = 0.45
    clahe_clip_limit: float = 2.0
    adaptive_block_size: int = 31
    adaptive_c: float = 9.0
    close_kernel: int = 3
    normalized_size: int = 64
    radial_bins: int = 12
    projection_bins: int = 16
    fourier_terms: int = 10


class GlyphStructuralPipeline:
    """Segment an image into independently measured glyph candidates."""

    def __init__(self, config: Optional[GlyphPipelineConfig] = None):
        self.config = config or GlyphPipelineConfig()

    def analyze(self, image_path: str | Path, output_dir: str | Path | None = None,
                source_image_id: str | None = None) -> dict[str, Any]:
        path = Path(image_path)
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Cannot read image: {path}")
        source_id = source_image_id or hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        gray, binary, angle = self._preprocess(raw)
        candidates, line_count = self._segment(binary)
        records = []
        for index, candidate in enumerate(candidates):
            x, y, w, h = candidate["bbox"]
            mask = candidate["mask"]
            features, contour = self._features(mask)
            confidence, reasons = self._confidence(candidate, features, binary.shape)
            records.append(_finite({
                "glyph_id": f"{source_id}_g{index:04d}",
                "source_image_id": source_id,
                "line_index": candidate["line_index"],
                "reading_order_index": index,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "mask": {"encoding": "rle", **_rle(mask)},
                "contour_points": [[int(px + x), int(py + y)] for px, py in contour],
                "extracted_feature_vector": features,
                "feature_version": FEATURE_VERSION,
                "confidence_score": confidence,
                "uncertain": confidence < 0.65,
                "uncertainty_reasons": reasons,
                "component_count": candidate["component_count"],
                "optional_label": None,
            }))
        summary = self._summary(path, source_id, raw.shape, angle, line_count, records)
        result = {"segmentation_summary": summary, "glyphs": records}
        if output_dir is not None:
            result["exports"] = self.export(result, output_dir, binary, raw)
        return result

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
        gray = cv2.createCLAHE(self.config.clahe_clip_limit, (8, 8)).apply(gray)
        block = max(3, int(self.config.adaptive_block_size) | 1)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, self.config.adaptive_c)
        # Reject border-connected illumination fields and select ink polarity.
        if np.mean(mask > 0) > 0.48:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        angle = self._skew_angle(mask)
        if abs(angle) > 0.08:
            center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]),
                                  flags=cv2.INTER_CUBIC, borderValue=255)
            mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.config.close_kernel,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return gray, mask, angle

    @staticmethod
    def _skew_angle(mask: np.ndarray) -> float:
        points = np.column_stack(np.where(mask > 0))
        if len(points) < 20:
            return 0.0
        angle = cv2.minAreaRect(points[:, ::-1].astype(np.float32))[-1]
        angle = angle - 90.0 if angle > 45.0 else angle
        return float(-angle) if abs(angle) <= 15 else 0.0

    def _segment(self, binary: np.ndarray) -> tuple[list[dict[str, Any]], int]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        image_area = binary.size
        min_area = max(3, int(image_area * self.config.min_component_area_ratio))
        max_area = int(image_area * self.config.max_component_area_ratio)
        components = []
        for label_id in range(1, n):
            x, y, w, h, area = map(int, stats[label_id])
            if min_area <= area <= max_area and w < binary.shape[1] * 0.98 and h < binary.shape[0] * 0.98:
                components.append({"id": label_id, "bbox": (x, y, w, h), "area": area,
                                   "cx": float(centroids[label_id, 0]), "cy": float(centroids[label_id, 1])})
        if not components:
            return [], 0
        heights = np.array([c["bbox"][3] for c in components], dtype=float)
        body_h = max(2.0, float(np.percentile(heights, 65)))
        bodies = [c for c in components if c["bbox"][3] >= body_h * 0.42 or c["area"] >= np.median([v["area"] for v in components])]
        if not bodies:
            bodies = components[:]
        # Greedy baseline clustering is stable and works for rotated horizontal text.
        lines: list[list[dict[str, Any]]] = []
        for comp in sorted(bodies, key=lambda c: (c["cy"], c["cx"])):
            match = next((line for line in lines if abs(comp["cy"] - np.median([v["cy"] for v in line])) <= body_h * 0.65), None)
            if match is None:
                lines.append([comp])
            else:
                match.append(comp)
        groups = [[c] for c in bodies]
        body_ids = {c["id"] for c in bodies}
        # Attach punctuation/diacritics to a nearby body when vertically aligned;
        # otherwise retain them as independent symbols.
        for mark in [c for c in components if c["id"] not in body_ids]:
            ranked = sorted(range(len(groups)), key=lambda i: abs(mark["cx"] - groups[i][0]["cx"]))
            target = ranked[0] if ranked else None
            if target is not None and abs(mark["cx"] - groups[target][0]["cx"]) <= max(body_h * 0.55, groups[target][0]["bbox"][2] * 0.65):
                groups[target].append(mark)
            else:
                groups.append([mark])
        candidates = []
        line_centers = sorted(float(np.median([v["cy"] for v in line])) for line in lines)
        for group in groups:
            x0 = min(c["bbox"][0] for c in group); y0 = min(c["bbox"][1] for c in group)
            x1 = max(c["bbox"][0] + c["bbox"][2] for c in group); y1 = max(c["bbox"][1] + c["bbox"][3] for c in group)
            local = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for c in group:
                local[labels[y0:y1, x0:x1] == c["id"]] = 255
            cy = (y0 + y1) / 2
            line_index = int(np.argmin(np.abs(np.asarray(line_centers) - cy))) if line_centers else 0
            candidates.append({"bbox": (x0, y0, x1 - x0, y1 - y0), "mask": local,
                               "component_count": len(group), "line_index": line_index})
        candidates.sort(key=lambda c: (c["line_index"], c["bbox"][0], c["bbox"][1]))
        return candidates, max(1, len(lines))

    def _features(self, mask: np.ndarray) -> tuple[dict[str, Any], list[list[int]]]:
        binary = (mask > 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        external = [c for i, c in enumerate(contours) if hierarchy is None or hierarchy[0][i][3] < 0]
        contour = max(external, key=cv2.contourArea) if external else np.empty((0, 1, 2), np.int32)
        area = int(binary.sum()); h, w = binary.shape; perimeter = sum(cv2.arcLength(c, True) for c in external)
        hull = cv2.convexHull(np.vstack(external)) if external else contour
        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0.0
        holes = [c for i, c in enumerate(contours) if hierarchy is not None and hierarchy[0][i][3] >= 0]
        hole_area = sum(abs(cv2.contourArea(c)) for c in holes)
        skeleton = skeletonize(binary > 0)
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton
        endpoints = int(np.sum(skeleton & (neighbors == 1)))
        branchpoints = int(np.sum(skeleton & (neighbors >= 3)))
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        widths = 2.0 * distance[skeleton]
        moments = cv2.moments(binary)
        hu = cv2.HuMoments(moments).flatten()
        hu = [-math.copysign(math.log10(abs(v)), v) if abs(v) > 1e-30 else 0.0 for v in hu]
        normalized = cv2.resize(binary, (self.config.normalized_size,) * 2, interpolation=cv2.INTER_NEAREST)
        radial = self._radial(normalized, self.config.radial_bins)
        horizontal = self._bin_profile(normalized.sum(axis=1), self.config.projection_bins)
        vertical = self._bin_profile(normalized.sum(axis=0), self.config.projection_bins)
        fourier = self._fourier(contour, self.config.fourier_terms)
        eccentricity = self._eccentricity(binary)
        curvature = self._curvature(contour)
        opening_ratio = self._opening_ratio(binary)
        m00 = max(float(moments["m00"]), 1.0)
        features = {
            "geometric": {"area": area, "perimeter": perimeter, "contour_length": len(contour),
                "bounding_box_width": w, "bounding_box_height": h, "aspect_ratio": w / max(h, 1),
                "compactness": 4 * math.pi * area / max(perimeter * perimeter, 1.0),
                "solidity": area / max(hull_area, 1.0), "convex_hull_area_ratio": hull_area / max(area, 1),
                "eccentricity": eccentricity, "extent": area / max(w * h, 1)},
            "stroke": {"average_stroke_width": float(np.mean(widths)) if len(widths) else 0.0,
                "stroke_width_variance": float(np.var(widths)) if len(widths) else 0.0,
                "stroke_density": area / max(w * h, 1), "skeleton_length": int(skeleton.sum()),
                "skeleton_length_normalized": float(skeleton.sum()) / max(area, 1),
                "endpoints": endpoints, "branch_points": branchpoints, "curvature": curvature},
            "structural": {"hole_count": len(holes), "hole_to_area_ratio": hole_area / max(area, 1),
                "opening_ratio": opening_ratio, "connected_components_inside_glyph": int(cv2.connectedComponents(binary, 8)[0] - 1),
                "euler_number": int(cv2.connectedComponents(binary, 8)[0] - 1 - len(holes)),
                "symmetry_score": self._symmetry(binary)},
            "shape": {"moments": [moments[k] / m00 for k in ("m10", "m01", "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03")],
                "hu_moments_log": hu, "radial_distribution": radial,
                "horizontal_projection": horizontal, "vertical_projection": vertical,
                "fourier_descriptors": fourier},
        }
        features["flat_vector"] = self._flatten(features)
        features["flat_vector_names"] = self._flatten(features, names=True)
        return _finite(features), contour.reshape(-1, 2).tolist()

    @staticmethod
    def _flatten(features: dict[str, Any], names: bool = False) -> list[Any]:
        output = []
        for section in ("geometric", "stroke", "structural", "shape"):
            for key, value in features[section].items():
                values = value if isinstance(value, list) else [value]
                output.extend([f"{section}.{key}[{i}]" if len(values) > 1 else f"{section}.{key}" for i in range(len(values))] if names else values)
        return output

    @staticmethod
    def _eccentricity(binary: np.ndarray) -> float:
        ys, xs = np.where(binary > 0)
        if len(xs) < 3: return 0.0
        eig = np.sort(np.linalg.eigvalsh(np.cov(np.vstack((xs, ys)))))[::-1]
        return math.sqrt(max(0.0, 1.0 - eig[1] / max(eig[0], 1e-9)))

    @staticmethod
    def _curvature(contour: np.ndarray) -> dict[str, float]:
        pts = contour.reshape(-1, 2).astype(float)
        if len(pts) < 5: return {"mean": 0.0, "variance": 0.0, "maximum": 0.0}
        step = max(1, len(pts) // 64); pts = pts[::step]
        a = np.roll(pts, 1, axis=0) - pts; b = np.roll(pts, -1, axis=0) - pts
        cos = np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-9)
        values = np.pi - np.arccos(np.clip(cos, -1, 1))
        return {"mean": float(values.mean()), "variance": float(values.var()), "maximum": float(values.max())}

    @staticmethod
    def _symmetry(binary: np.ndarray) -> float:
        flip = np.fliplr(binary); inter = np.sum((binary > 0) & (flip > 0)); union = np.sum((binary > 0) | (flip > 0))
        return float(inter / max(union, 1))

    @staticmethod
    def _opening_ratio(binary: np.ndarray) -> float:
        padded = np.pad(binary, 1); inv = (padded == 0).astype(np.uint8)
        flood = inv.copy(); flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 2)
        boundary_bg = flood[1:-1, 1:-1] == 2
        return float(np.sum(boundary_bg) / max(np.sum(binary == 0), 1))

    @staticmethod
    def _bin_profile(profile: np.ndarray, bins: int) -> list[float]:
        chunks = np.array_split(profile.astype(float), bins); total = max(float(profile.sum()), 1.0)
        return [float(c.sum() / total) for c in chunks]

    @staticmethod
    def _radial(binary: np.ndarray, bins: int) -> list[float]:
        ys, xs = np.where(binary > 0)
        if not len(xs): return [0.0] * bins
        cx, cy = xs.mean(), ys.mean(); distances = np.hypot(xs-cx, ys-cy); scale = max(distances.max(), 1.0)
        hist, _ = np.histogram(distances / scale, bins=bins, range=(0, 1))
        return (hist / max(hist.sum(), 1)).astype(float).tolist()

    @staticmethod
    def _fourier(contour: np.ndarray, terms: int) -> list[float]:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3: return [0.0] * terms
        z = pts[:, 0] + 1j * pts[:, 1]; coeff = np.fft.fft(z - z.mean()); scale = max(abs(coeff[1]), 1e-9)
        return [float(abs(coeff[i]) / scale) if i < len(coeff) else 0.0 for i in range(1, terms + 1)]

    @staticmethod
    def _confidence(candidate: dict[str, Any], features: dict[str, Any], shape: tuple[int, int]) -> tuple[float, list[str]]:
        geo, stroke = features["geometric"], features["stroke"]
        confidence, reasons = 0.95, []
        if geo["aspect_ratio"] > 2.5: confidence -= 0.25; reasons.append("wide_region_may_be_ligature")
        if stroke["branch_points"] > 18: confidence -= 0.12; reasons.append("complex_connected_strokes")
        if candidate["component_count"] > 3: confidence -= 0.12; reasons.append("compound_or_multiple_diacritics")
        if geo["area"] < shape[0] * shape[1] * 0.00008: confidence -= 0.18; reasons.append("very_small_symbol_or_noise")
        if geo["extent"] > 0.92: confidence -= 0.08; reasons.append("solid_region_may_be_artifact")
        return round(max(0.05, min(0.99, confidence)), 4), reasons

    @staticmethod
    def _summary(path: Path, source_id: str, shape: tuple[int, ...], angle: float,
                 lines: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        confidences = [g["confidence_score"] for g in records]
        return _finite({"source_image_id": source_id, "source_image_path": str(path.resolve()),
            "image_width": shape[1], "image_height": shape[0], "deskew_angle_degrees": angle,
            "line_count": lines, "glyph_count": len(records),
            "uncertain_glyph_count": sum(g["uncertain"] for g in records),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "feature_version": FEATURE_VERSION,
            "limitations": "Connected ligatures are retained as uncertain candidates; script-specific grapheme splitting requires a trained model."})

    def export(self, result: dict[str, Any], output_dir: str | Path,
               binary: np.ndarray | None = None, source: np.ndarray | None = None) -> dict[str, str]:
        out = Path(output_dir); glyph_dir = out / "glyphs"; glyph_dir.mkdir(parents=True, exist_ok=True)
        json_path = out / "glyphs.json"; csv_path = out / "glyphs.csv"; report_path = out / "report.json"
        json_path.write_text(json.dumps(result["glyphs"], indent=2, ensure_ascii=False), encoding="utf-8")
        report_path.write_text(json.dumps(result["segmentation_summary"], indent=2, ensure_ascii=False), encoding="utf-8")
        rows = []
        for glyph in result["glyphs"]:
            bbox = glyph["bounding_box"]; names = glyph["extracted_feature_vector"]["flat_vector_names"]
            values = glyph["extracted_feature_vector"]["flat_vector"]
            rows.append({"glyph_id": glyph["glyph_id"], "source_image_id": glyph["source_image_id"],
                         "x": bbox["x"], "y": bbox["y"], "width": bbox["width"], "height": bbox["height"],
                         "confidence_score": glyph["confidence_score"], "uncertain": glyph["uncertain"],
                         **dict(zip(names, values))})
            if binary is not None:
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                cv2.imwrite(str(glyph_dir / f'{glyph["glyph_id"]}_mask.png'), binary[y:y+h, x:x+w])
                if source is not None: cv2.imwrite(str(glyph_dir / f'{glyph["glyph_id"]}.png'), source[y:y+h, x:x+w])
            (glyph_dir / f'{glyph["glyph_id"]}.json').write_text(json.dumps(glyph, indent=2), encoding="utf-8")
        fields = list(rows[0]) if rows else ["glyph_id", "source_image_id", "confidence_score", "uncertain"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "glyph_directory": str(glyph_dir)}
