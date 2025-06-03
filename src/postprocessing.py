# src/postprocessing.py

import numpy as np
import math
from scipy.spatial import ConvexHull
from sklearn.neighbors import BallTree
from tqdm import tqdm
import logging

from src import config

logger = logging.getLogger(__name__)

class PortPostProcessor:
    """
    Takes a list of raw clusters (deduplicated) and:
    1) Merges very‐close clusters into a single “port”
    2) Computes convex hull, area in km², vessel density, etc.
    3) Assigns each port a size category and color (based on area)
    """

    def __init__(self, clusters):
        self.clusters = clusters
        self.final_ports = []

    def merge_clusters(self):
        """
        Merge any DBSCAN clusters whose centers lie within
        1.5 * max(eps_km of either cluster) (in kilometres).
        We first sort by (scale_priority, -point_count), then:
        for each cluster in that order, grab all neighbors within
        its merge_radius_km and mark them as “taken.”
        """
        if not self.clusters:
            return []

        # 1) Assign numeric priority to each cluster’s scale
        scale_priority = {
            'major_ports': 0,
            'regional_ports': 1,
            'local_ports': 2,
            'small_harbors': 3
        }

        n = len(self.clusters)
        coords_rad = np.zeros((n, 2))    # lat/lon in radians
        eps_arr = np.zeros(n, dtype=float)
        priorities = np.zeros(n, dtype=int)

        # 2) Fill arrays: coords_rad, eps_arr, priorities
        for i, c in enumerate(self.clusters):
            φ = math.radians(c["center_lat"])
            λ = math.radians(c["center_lon"])
            coords_rad[i, 0] = φ
            coords_rad[i, 1] = λ
            eps_arr[i] = c["eps_km"]
            priorities[i] = scale_priority.get(c["scale"], 99)

        # 3) Build BallTree on coords_rad with haversine metric
        tree = BallTree(coords_rad, metric="haversine")
        EARTH_RADIUS_KM = 6371.0

        # 4) Sort indices by (priority, -point_count)
        sorted_idx = sorted(
            range(n),
            key=lambda i: (priorities[i], -self.clusters[i]["point_count"])
        )

        taken = np.zeros(n, dtype=bool)
        merged = []

        for idx in sorted_idx:
            # If this cluster was already merged into another, skip it
            if taken[idx]:
                continue

            c1 = self.clusters[idx]
            base_phi, base_lambda = coords_rad[idx]

            # 5) Compute merge radius in radians (1.5 * eps_km / EARTH_RADIUS_KM)
            merge_radius_rad = 1.5 * eps_arr[idx] / EARTH_RADIUS_KM

            # 6) Query all neighbors within that radius (including idx itself)
            neighbors = tree.query_radius(
                coords_rad[idx].reshape(1, -1),
                r=merge_radius_rad
            )[0]  # returns a 1×k array, so [0] to get the list of indices

            # 7) Mark all those neighbors “taken”
            taken[neighbors] = True

            # 8) Collect all points from every neighbor cluster
            group_points = []
            total_point_count = 0
            group_scales = []
            for nbr in neighbors:
                group_points.append(self.clusters[nbr]["points"])
                total_point_count += self.clusters[nbr]["point_count"]
                group_scales.append((self.clusters[nbr]["scale"],
                                     self.clusters[nbr]["point_count"]))

            all_pts = np.vstack(group_points)
            # Pick the “highest‐priority, largest‐point_count” scale in this group:
            primary_scale = sorted(
                group_scales,
                key=lambda x: (scale_priority[x[0]], -x[1])
            )[0][0]

            merged.append({
                "center_lat": float(all_pts[:, 0].mean()),
                "center_lon": float(all_pts[:, 1].mean()),
                "points": all_pts,
                "point_count": int(total_point_count),
                "detected_scale": primary_scale,
                "dbscan_clusters": len(neighbors)
            })

        logger.info(f"Merged {len(self.clusters)} clusters → {len(merged)} ports")
        self.final_ports = merged
        return merged

    def compute_area_and_categorize(self):
        """
        For each merged port, compute:
         - convex hull area (approx km², using lat/lon correction)
         - max distance across hull
         - vessel density (point_count / area)
         - category & color (using config.PORT_SIZE_CATEGORIES)
        Then filter out anything outside MIN_PORT_AREA_KM2 … MAX_PORT_AREA_KM2.
        """
        if not self.final_ports:
            return []

        out_ports = []
        for port in tqdm(self.final_ports, desc="Calculating areas"):
            pts = port["points"]
            if pts.shape[0] < 3:
                # Too few points → hull fails; skip it
                continue

            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]

                # Approximate area: lat_range × (lon_range × cos(avg_lat)) × 111 × 111
                latr = np.ptp(hull_pts[:, 0])
                lonr = np.ptp(hull_pts[:, 1])
                avg_lat = hull_pts[:, 0].mean()
                correction = np.cos(np.radians(avg_lat))
                area_km2 = latr * (lonr * correction) * 111 * 111

                # Maximum pairwise distance (approx)
                max_d = 0.0
                for i in range(len(hull_pts)):
                    for j in range(i + 1, len(hull_pts)):
                        dlat = (hull_pts[i, 0] - hull_pts[j, 0]) * 111
                        dlon = (hull_pts[i, 1] - hull_pts[j, 1]) * 111 * correction
                        dist = np.hypot(dlat, dlon)
                        if dist > max_d:
                            max_d = dist

                density = port["point_count"] / area_km2 if area_km2 > 0 else 0.0

                # Assign category
                category = "Uncategorized"
                color    = config.PORT_SIZE_CATEGORIES["Uncategorized"]["color"]
                for cat, specs in config.PORT_SIZE_CATEGORIES.items():
                    if specs["min"] <= area_km2 <= specs["max"]:
                        category = cat
                        color    = specs["color"]
                        break

                # Keep only if within global area limits
                if config.MIN_PORT_AREA_KM2 <= area_km2 <= config.MAX_PORT_AREA_KM2:
                    out_ports.append({
                        "center_lat": port["center_lat"],
                        "center_lon": port["center_lon"],
                        "points": pts,
                        "point_count": port["point_count"],
                        "detected_scale": port["detected_scale"],
                        "dbscan_clusters": port["dbscan_clusters"],
                        "area_km2": area_km2,
                        "max_distance_km": max_d,
                        "vessel_density": density,
                        "category": category,
                        "color": color
                    })

            except Exception as e:
                # If hull computation fails for any reason, skip this port
                logger.warning(f"Skipping port at ({port['center_lat']:.4f}, {port['center_lon']:.4f}): {e}")
                continue

        # Sort descending by area
        out_ports.sort(key=lambda x: x["area_km2"], reverse=True)
        logger.info(f"After filtering by area: {len(out_ports)} ports remain")
        self.final_ports = out_ports
        return out_ports