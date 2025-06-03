# src/config.py

import os

# ----------------------------------------
# 1A. Paths
# ----------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")

# Where to find the raw AIS CSV
AIS_INPUT_FILE = os.path.join(DATA_DIR, "aisdk-2024-05-04.csv")

# Subfolders in output (they will be created if missing)
FILTERED_CHUNKS_DIR = os.path.join(OUTPUT_DIR, "filtered_chunks")
PLOTS_DIR           = os.path.join(OUTPUT_DIR, "plots")
MAPS_DIR            = os.path.join(OUTPUT_DIR, "maps")
REPORTS_DIR         = os.path.join(OUTPUT_DIR, "reports")

# ----------------------------------------
# 1B. Geographic / filtering thresholds
# ----------------------------------------
LATITUDE_BOUNDS  = (54.5, 58.0)
LONGITUDE_BOUNDS = (8.0, 15.5)
MAX_SOG          = 0.5  # Stationary vessel threshold (knots)
MIN_CLUSTER_POINTS = 30  # absolute minimum to even write a chunk to disk

# ----------------------------------------
# 1C. DBSCAN parameter configurations (multi-scale)
# ----------------------------------------
DBSCAN_CONFIGS = {
    "major_ports": {
        "eps_km":      1.3,
        "min_samples": 200,
        "label":       "Major Commercial"
    },
    "regional_ports": {
        "eps_km":      1,
        "min_samples": 150,
        "label":       "Regional"
    },
    "local_ports": {
        "eps_km":      0.5,
        "min_samples": 60,
        "label":       "Local/Industrial"
    },
    "small_harbors": {
        "eps_km":      0.2,
        "min_samples": 30,
        "label":       "Small Harbor"
    },
}

# ----------------------------------------
# 1D. Port size‐category definitions (km²)
# ----------------------------------------
PORT_SIZE_CATEGORIES = {
    "Major Commercial":   {"min": 2.0,  "max": 15.0, "color": "red"},
    "Regional":           {"min": 0.5,  "max": 2.0,  "color": "orange"},
    "Local/Industrial":   {"min": 0.1,  "max": 0.5,  "color": "blue"},
    "Small Harbor":       {"min": 0.01, "max": 0.1,  "color": "green"},
    # Fallback bucket:
    "Uncategorized":      {"min": 0.0,  "max": 0.0,  "color": "gray"},
}

MIN_PORT_AREA_KM2 = 0.005   # 5 000 m²
MAX_PORT_AREA_KM2 = 20.0    # 20 km²

# ----------------------------------------
# 1E. Other thresholds
# ----------------------------------------
EARTH_RADIUS_KM = 6371.0