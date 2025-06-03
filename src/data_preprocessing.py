# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from src import config
from src.utils import ensure_directories_exist

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Behavioral-based AIS data filtering with coastline proximity:
    1. SOG ≤ MAX_SOG (stationary/slow vessels)
    2. EXCLUDE obvious non-port activities (fishing, sailing, etc.)
    3. COG variance filtering (maneuvering behavior)
    4. COASTLINE DISTANCE filtering (≤5km from shore)
    5. Let DBSCAN find actual port clusters based on spatial behavior
    """

    def __init__(self, input_csv, chunk_size):
        self.input_csv = input_csv
        self.chunk_size = chunk_size
        self.filtered_chunks = []

        # EXCLUDE these statuses - definitely NOT port activities
        self.exclude_statuses = [
            'Engaged in fishing',  # 11.1% - fishing vessels at sea
            'Under way sailing',  # 1.5% - sailing vessels in transit
            'Restricted maneuverability'  # 4.3% - ships with limited movement
        ]

        # COG variance threshold - ships maneuvering vs steady course
        self.cog_variance_threshold = 35.0  # degrees²

        ensure_directories_exist(config.FILTERED_CHUNKS_DIR)

    def _calculate_cog_variance(self, chunk):
        """
        Calculate COG variance per vessel (MMSI).
        High variance = maneuvering behavior (typical at ports)
        """
        cog_variance = chunk.groupby('MMSI')['COG'].agg(
            lambda x: x.var() if len(x) > 1 else np.nan
        )
        return cog_variance

    def _apply_exclusion_filter(self, chunk):
        """
        EXCLUDE obvious non-port activities, but keep everything else.
        This preserves vessels that might be maneuvering in ports.
        """
        # Create exclusion mask
        exclude_mask = chunk['Navigational status'].isin(self.exclude_statuses)

        # Keep everything that's NOT excluded
        return chunk[~exclude_mask]

    def _apply_cog_behavioral_filter(self, chunk):
        """
        Filter based on COG behavior:
        - Keep vessels with missing COG (truly stationary)
        - Keep vessels with high COG variance (maneuvering - port behavior)
        - Remove vessels with very low variance (steady transit)
        """
        if chunk.empty:
            return chunk

        # Calculate COG variance per vessel
        cog_variance = self._calculate_cog_variance(chunk)

        # Create individual boolean masks to avoid pandas dtype issues
        missing_cog_mask = chunk['COG'].isna() | chunk['COG'].isnull()

        # Map variance and handle missing values
        variance_values = chunk['MMSI'].map(cog_variance)
        high_variance_mask = variance_values >= self.cog_variance_threshold
        no_variance_mask = variance_values.isna()

        # Combine masks using boolean operations
        cog_mask = missing_cog_mask | high_variance_mask | no_variance_mask

        return chunk[cog_mask]

    def _apply_coastline_distance_filter(self, chunk, max_distance_km=2.0):
        """
        Simple coastline distance filter for Danish waters.
        Remove ships more than ~5km from probable coastline areas.
        Uses geographic zones to approximate coastal proximity.
        """
        if chunk.empty:
            return chunk

        # Danish waters coastal zone approximation
        # This removes ships in deep water areas while keeping coastal vessels

        coastal_mask = (
            # West coast of Jutland (close to shore)
                (chunk['Longitude'] <= 8.8) |

                # East coast and Danish straits (Øresund, Great Belt, Little Belt)
                (chunk['Longitude'] >= 11.8) |

                # Kattegat shallow areas (between Jutland and Sweden)
                ((chunk['Latitude'] <= 57.5) &
                 (chunk['Longitude'].between(10.5, 12.5)) &
                 (chunk['Latitude'] >= 55.5)) |

                # Belt Sea areas (between Danish islands)
                ((chunk['Latitude'].between(54.8, 56.2)) &
                 (chunk['Longitude'].between(9.5, 11.5))) |

                # Coastal fjords and harbors (closer to shore)
                ((chunk['Latitude'] >= 56.8) &
                 (chunk['Longitude'].between(9.0, 11.0))) |

                # Copenhagen/Øresund area
                ((chunk['Latitude'].between(55.4, 56.2)) &
                 (chunk['Longitude'].between(12.2, 13.0))) |

                # Bornholm area (Danish island in Baltic)
                ((chunk['Latitude'].between(55.0, 55.4)) &
                 (chunk['Longitude'].between(14.5, 15.2)))
        )

        return chunk[coastal_mask]

    def run(self):
        """
        Enhanced behavioral filtering pipeline:
        1. Basic geographic + SOG filtering
        2. Exclude obvious non-port activities
        3. COG behavioral filtering
        4. Coastline distance filtering (NEW!)
        5. Let DBSCAN cluster based on spatial patterns
        """
        logger.info("Starting enhanced behavioral filtering (SOG + exclusions + COG + coastline)...")

        # Statistics tracking
        stats = {
            'original': 0,
            'after_basic': 0,
            'after_exclusion': 0,
            'after_cog': 0,
            'after_coastline': 0,
            'final': 0
        }
        chunk_id = 0

        # Required columns (added Navigational status and COG)
        columns_to_use = [
            "Latitude", "Longitude", "SOG", "MMSI",
            "Navigational status", "COG"
        ]

        reader = pd.read_csv(self.input_csv, usecols=columns_to_use, chunksize=self.chunk_size)

        for chunk in tqdm(reader, desc="Processing chunks"):
            stats['original'] += len(chunk)

            # 1) Basic filtering (geographic bounds + SOG threshold)
            chunk = chunk.dropna(subset=["Latitude", "Longitude", "SOG", "MMSI"])
            if chunk.empty:
                continue

            chunk = chunk[
                chunk["Latitude"].between(*config.LATITUDE_BOUNDS) &
                chunk["Longitude"].between(*config.LONGITUDE_BOUNDS) &
                (chunk["SOG"] <= config.MAX_SOG)
                ]
            if chunk.empty:
                continue
            stats['after_basic'] += len(chunk)

            # 2) Exclude obvious non-port activities (fishing, sailing, restricted)
            chunk = self._apply_exclusion_filter(chunk)
            if chunk.empty:
                continue
            stats['after_exclusion'] += len(chunk)

            # 3) COG behavioral filtering (maneuvering vs transit)
            chunk = self._apply_cog_behavioral_filter(chunk)
            if chunk.empty:
                continue
            stats['after_cog'] += len(chunk)

            # 4) NEW: Coastline distance filtering
            chunk = self._apply_coastline_distance_filter(chunk)
            if chunk.empty:
                continue
            stats['after_coastline'] += len(chunk)

            # 5) Save chunk if sufficient points for clustering
            if len(chunk) >= config.MIN_CLUSTER_POINTS:
                out_file = os.path.join(
                    config.FILTERED_CHUNKS_DIR,
                    f"stationary_chunk_{chunk_id:03d}.csv"
                )
                # Save only columns needed for clustering
                chunk.to_csv(
                    out_file,
                    index=False,
                    columns=["Latitude", "Longitude", "SOG", "MMSI"]
                )
                self.filtered_chunks.append({
                    "file": out_file,
                    "records": len(chunk),
                    "chunk_id": chunk_id
                })
                stats['final'] += len(chunk)
                chunk_id += 1

        # Log detailed statistics
        self._log_enhanced_statistics(stats)

        return self.filtered_chunks, stats['final']

    def _log_enhanced_statistics(self, stats):
        """Log detailed filtering statistics including coastline filter"""
        logger.info("Enhanced behavioral filtering results:")
        logger.info(f"  • Original records: {stats['original']:,}")

        if stats['original'] > 0:
            basic_pct = (stats['after_basic'] / stats['original']) * 100
            logger.info(f"  • After basic (geo + SOG ≤ {config.MAX_SOG}): {stats['after_basic']:,} ({basic_pct:.1f}%)")

            if stats['after_basic'] > 0:
                excl_pct = (stats['after_exclusion'] / stats['after_basic']) * 100
                logger.info(f"  • After exclusions: {stats['after_exclusion']:,} ({excl_pct:.1f}% of basic)")

                if stats['after_exclusion'] > 0:
                    cog_pct = (stats['after_cog'] / stats['after_exclusion']) * 100
                    logger.info(f"  • After COG behavior: {stats['after_cog']:,} ({cog_pct:.1f}% of excluded)")

                    if stats['after_cog'] > 0:
                        coast_pct = (stats['after_coastline'] / stats['after_cog']) * 100
                        logger.info(
                            f"  • After coastline filter: {stats['after_coastline']:,} ({coast_pct:.1f}% of COG)")

            final_pct = (stats['final'] / stats['original']) * 100
            logger.info(f"  • Final retention: {final_pct:.2f}% of original")

        logger.info(f"  • Chunks created for DBSCAN: {len(self.filtered_chunks)}")

        # Log what each filter removed
        excluded_count = stats['after_basic'] - stats['after_exclusion']
        logger.info(f"  • Excluded activities (fishing/sailing): {excluded_count:,}")

        cog_filtered = stats['after_exclusion'] - stats['after_cog']
        logger.info(f"  • Filtered by COG (steady transit): {cog_filtered:,}")

        coastline_filtered = stats['after_cog'] - stats['after_coastline']
        logger.info(f"  • Filtered by coastline (offshore vessels): {coastline_filtered:,}")

        logger.info(f"  • NEW: Coastline filter removed {coastline_filtered:,} offshore vessels!")