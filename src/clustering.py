# src/clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import math
from sklearn.neighbors import BallTree
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import time
import logging

from src import config

logger = logging.getLogger(__name__)

class Clusterer:
    """
    Performs multi-scale DBSCAN clustering on each filtered chunk (from DataPreprocessor),
    using a Haversine metric (great-circle distance). After clustering all chunks at
    each scale, it deduplicates overlapping clusters into a final list.
    """

    def __init__(self, filtered_chunks, n_cores=None):
        """
        :param filtered_chunks: list of dicts {'file': <csv_path>, 'records': <int>, 'chunk_id': <int>}
        :param n_cores: number of parallel processes to use; defaults to mp.cpu_count()
        """
        self.filtered_chunks = filtered_chunks
        self.n_cores = n_cores or mp.cpu_count()
        self.raw_clusters = []
        # Earth's radius in kilometres
        self.EARTH_RADIUS_KM = 6371.0

    def _cluster_one_chunk(self, chunk_info, scale_key, scale_config):
        """
        Apply DBSCAN (with Haversine metric) to a single chunk for the given scale.
        Returns a list of cluster dicts, each containing centroid, points, and metadata.
        """
        try:
            df = pd.read_csv(chunk_info["file"])
            if len(df) < scale_config["min_samples"]:
                return []  # Too few points to form any cluster

            # 1) Extract raw lat/lon in degrees
            coords_deg = df[["Latitude", "Longitude"]].to_numpy()

            # 2) Convert degrees → radians for Haversine
            coords_rad = np.radians(coords_deg)

            # 3) Convert eps_km → radians (eps_rad = eps_km / Earth's radius)
            eps_km = scale_config["eps_km"]
            eps_rad = eps_km / self.EARTH_RADIUS_KM

            # 4) Run DBSCAN with Haversine metric
            model = DBSCAN(
                eps=eps_rad,
                min_samples=scale_config["min_samples"],
                metric="haversine",
                n_jobs=1
            )
            labels = model.fit_predict(coords_rad)

            clusters = []
            for label in np.unique(labels):
                if label == -1:
                    continue  # skip noise

                mask = (labels == label)
                pts_deg = coords_deg[mask]
                center_lat = float(pts_deg[:, 0].mean())
                center_lon = float(pts_deg[:, 1].mean())

                clusters.append({
                    "chunk_id": chunk_info["chunk_id"],
                    "scale": scale_key,
                    "points": pts_deg,               # stored in degrees for plotting
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "point_count": int(len(pts_deg)),
                    "eps_km": eps_km,                # keep track of radius used
                    "min_samples_used": scale_config["min_samples"]
                })

            return clusters

        except Exception as e:
            logger.error(
                f"Error clustering chunk {chunk_info['chunk_id']} at scale {scale_key}: {e}"
            )
            return []

    def run_multiscale(self):
        """
        For each scale defined in config.DBSCAN_CONFIGS, spin up a Pool to cluster all chunks
        in parallel. After all scales are processed, merge & deduplicate clusters.
        Returns the final deduplicated list of clusters.
        """
        logger.info("Starting hierarchical DBSCAN clustering")
        all_scale_clusters = []

        for scale_key, scale_config in config.DBSCAN_CONFIGS.items():
            eps_km = scale_config["eps_km"]
            min_samples = scale_config["min_samples"]
            label = scale_config["label"]

            logger.info(
                f" → Scale: {label} (eps_km={eps_km:.2f} km, min_samples={min_samples})"
            )
            start = time.time()

            # Build argument tuples: (chunk_info, scale_key, scale_config)
            args = [
                (chunk_info, scale_key, scale_config)
                for chunk_info in self.filtered_chunks
            ]

            # Use starmap to avoid pickling a lambda
            with Pool(processes=self.n_cores) as pool:
                results = list(tqdm(
                    pool.starmap(self._cluster_one_chunk, args),
                    total=len(args),
                    desc=f"DBSCAN {label}"
                ))

            # Flatten the list of lists
            scale_clusters = [cluster for sublist in results for cluster in sublist]
            all_scale_clusters.extend(scale_clusters)

            elapsed = time.time() - start
            logger.info(
                f"   → {len(scale_clusters)} clusters found at this scale (took {elapsed:.1f}s)"
            )

        # Deduplicate / merge overlapping clusters
        logger.info(f"Total raw clusters (all scales): {len(all_scale_clusters)}")
        self.raw_clusters = self._deduplicate_clusters(all_scale_clusters)
        logger.info(f"After deduplication: {len(self.raw_clusters)} clusters")
        return self.raw_clusters

    def _deduplicate_clusters(self, clusters):
        """
        Use a BallTree on (lat_rad, lon_rad) to remove overlapping clusters efficiently.
        We sort clusters by priority (scale first, then -point_count), then:
        for each cluster in priority order, mark any neighbors within 1.5 * eps_km
        as “taken” so they cannot appear again.

        Returns a list of deduplicated clusters.
        """
        if not clusters:
            return []

        n = len(clusters)
        coords_rad = np.zeros((n, 2))
        eps_km_arr = np.zeros(n, dtype=float)
        priorities = np.zeros(n, dtype=int)

        # Define a priority mapping so that major_ports < regional_ports < local_ports < small_harbors
        priority_map = {
            "major_ports": 0,
            "regional_ports": 1,
            "local_ports": 2,
            "small_harbors": 3
        }

        # Build arrays of (lat_rad, lon_rad), eps_km, and priority for each cluster
        for i, c in enumerate(clusters):
            lat = c["center_lat"]
            lon = c["center_lon"]
            coords_rad[i, 0] = math.radians(lat)
            coords_rad[i, 1] = math.radians(lon)
            eps_km_arr[i] = c["eps_km"]
            priorities[i] = priority_map.get(c["scale"], 99)

        # Compute “merge radius” in radians = (1.5 * eps_km) / EarthRadius_km
        EARTH_RADIUS_KM = self.EARTH_RADIUS_KM  # 6371.0
        merge_radii = (1.5 * eps_km_arr) / EARTH_RADIUS_KM

        # Instead of np.argsort on a list of tuples, use Python’s sorted to get a list of ints
        sorted_idx = sorted(
            range(n),
            key=lambda i: (priorities[i], -clusters[i]["point_count"])
        )

        # Build a BallTree on coords_rad (using haversine metric)
        tree = BallTree(coords_rad, metric="haversine")

        taken = np.zeros(n, dtype=bool)
        deduped = []

        # Iterate in sorted (priority) order
        for idx in sorted_idx:
            # Now idx is a single Python int, so taken[idx] is a scalar bool
            if taken[idx]:
                continue

            # This cluster “survives” → add it to deduped
            deduped.append(clusters[idx])

            # Query all neighbors within the merge radius of this cluster (in radians)
            radius_rad = merge_radii[idx]
            neighbors = tree.query_radius(coords_rad[idx].reshape(1, -1), r=radius_rad)[0]

            # Mark each neighbor (including itself) as taken
            taken[neighbors] = True

        return deduped