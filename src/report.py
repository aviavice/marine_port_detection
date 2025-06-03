# src/report.py

import os
import time
import numpy as np
import logging

from src import config
from src.utils import ensure_directories_exist

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Builds a plain‐text report (multiscale_dbscan_report.txt) summarizing:
      • Preprocessing stats
      • DBSCAN configuration and detection counts
      • Port categorization details
      • Per‐port DBSCAN metrics
    """

    def __init__(self, final_ports, total_stationary):
        self.ports = final_ports
        self.total_stationary = total_stationary
        ensure_directories_exist(config.REPORTS_DIR)

    def write_report(self):
        total_ports = len(self.ports)
        vessel_counts = [p["point_count"] for p in self.ports]
        areas = [p["area_km2"] for p in self.ports] if self.ports else [0]

        stats = {
            "total_ports": total_ports,
            "total_vessels": sum(vessel_counts) if vessel_counts else 0,
            "largest_area": max(areas),
            "smallest_area": min(areas),
            "average_area": np.mean(areas) if areas else 0,
            "total_stationary": self.total_stationary
        }

        # Build the text
        lines = []
        lines.append("MULTI-SCALE DBSCAN MARINE PORT DETECTION REPORT")
        lines.append("===============================================")
        lines.append("")
        lines.append("CORE ALGORITHM: DBSCAN (density‐based spatial clustering)")
        lines.append("")
        lines.append("HIERARCHICAL CONFIGURATION:")
        for key, cfg in config.DBSCAN_CONFIGS.items():
            cnt = sum(1 for p in self.ports if p["detected_scale"] == key)
            lines.append(f"• {cfg['label']}: eps_km={cfg['eps_km']}, min_samples={cfg['min_samples']} → {cnt} ports")

        lines.append("")
        lines.append("PREPROCESSING FOR DBSCAN:")
        lines.append(f"• Stationary vessel filter: SOG ≤ {config.MAX_SOG} knots")
        lines.append(f"• Geographic bounds: lat {config.LATITUDE_BOUNDS}, lon {config.LONGITUDE_BOUNDS}")
        clustered_vessels = stats["total_vessels"]
        total_stat = stats["total_stationary"] or 1
        lines.append(f"• Filtering efficiency: {clustered_vessels:,} / {stats['total_stationary']:,} stationary vessels clustered ({(clustered_vessels/total_stat)*100:.1f}%)")
        lines.append("")
        lines.append("PORT DETECTION RESULTS:")
        lines.append(f"• Total ports detected: {stats['total_ports']}")
        lines.append(f"• Area range: {stats['smallest_area']:.3f} – {stats['largest_area']:.3f} km²")
        lines.append(f"• Average area: {stats['average_area']:.3f} km²")
        lines.append("")
        lines.append("PORT CATEGORIZATION:")
        # count per category
        cat_counts = {}
        for p in self.ports:
            cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
        for cat, cnt in cat_counts.items():
            specs = config.PORT_SIZE_CATEGORIES.get(cat, {"min": 0, "max": 0})
            lines.append(f"• {cat} ({specs['min']:.2f}–{specs['max']:.2f} km²): {cnt} ports")
        lines.append("")
        lines.append("PER-PORT DETAILS:")
        for idx, p in enumerate(self.ports, start=1):
            scale_lbl = config.DBSCAN_CONFIGS[p["detected_scale"]]["label"]
            lines.append(f"{idx:2d}. {p['category']} (Scale: {scale_lbl})")
            lines.append(f"    • Location: {p['center_lat']:.4f}°, {p['center_lon']:.4f}°")
            lines.append(f"    • Area: {p['area_km2']:.3f} km², Vessels: {p['point_count']}, Density: {p['vessel_density']:.0f}/km²")
            lines.append(f"    • DBSCAN clusters merged: {p['dbscan_clusters']}, Max distance: {p['max_distance_km']:.2f} km")
            lines.append("")

        lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        out_path = os.path.join(config.REPORTS_DIR, "multiscale_dbscan_report.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved report: {out_path}")
        return out_path