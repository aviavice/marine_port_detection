# src/cli.py

import os
import logging
import psutil

from src.utils import setup_logging, ensure_directories_exist
from src import config

from src.data_preprocessing import DataPreprocessor
from src.clustering         import Clusterer
from src.postprocessing     import PortPostProcessor
from src.visualisation      import Visualiser
from src.report             import ReportGenerator

def main():
    # 1) Setup
    logger = setup_logging("INFO")
    logger.info("=== Multi-Scale DBSCAN Port Detection (Modular Version) ===")

    # 2) Ensure output folders exist
    ensure_directories_exist(
        config.OUTPUT_DIR,
        config.FILTERED_CHUNKS_DIR,
        config.PLOTS_DIR,
        config.MAPS_DIR,
        config.REPORTS_DIR
    )

    # 3) Check input file
    if not os.path.exists(config.AIS_INPUT_FILE):
        logger.error(f"Input file not found: {config.AIS_INPUT_FILE}")
        return

    # 4) Determine chunk size based on available memory
    mem = psutil.virtual_memory().available / (1024 ** 3)
    chunk_size = 100_000 if mem < 3 else 200_000
    logger.info(f"Using chunk size = {chunk_size} rows per iteration")

    # 5) Run preprocessing
    preproc = DataPreprocessor(config.AIS_INPUT_FILE, chunk_size=chunk_size)
    filtered_chunks, total_stationary = preproc.run()

    if not filtered_chunks:
        logger.error("No stationary chunks were produced—exiting.")
        return

    # 6) Run DBSCAN clustering
    clusterer = Clusterer(filtered_chunks)
    raw_clusters = clusterer.run_multiscale()

    if not raw_clusters:
        logger.error("No clusters found—exiting.")
        return

    # 7) Merge clusters & compute port geometry
    postproc = PortPostProcessor(raw_clusters)
    merged_ports = postproc.merge_clusters()
    final_ports  = postproc.compute_area_and_categorize()

    if not final_ports:
        logger.error("No ports survived area/categorization filtering—exiting.")
        return

    # 8) Visualization (plots + map)
    viz = Visualiser(final_ports)
    viz.plot_summary()
    viz.make_interactive_map()

    # 9) Generate text report
    reporter = ReportGenerator(final_ports, total_stationary)
    reporter.write_report()

    logger.info("All tasks complete—check output/ directory for results.")

if __name__ == "__main__":
    main()