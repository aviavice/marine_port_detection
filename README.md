# Marine Port Detection

This repository implements a pipeline to detect marine ports from AIS data on the date 2024-05-04 using hierarchical DBSCAN clustering. The main entry point is `cli.py`.

## Prerequisites

1. Python 3.8+.
2. Install required Python packages. From the project root, run:

    ```bash
    pip install -r requirements.txt
    ```

## Project structure

```
structurised_group_project/
├── data/
│   └── aisdk-2024-05-04.csv
├── output/
│   ├── filtered_chunks/
│   ├── plots/
│   ├── maps/
│   └── reports/
└── src/
    ├── cli.py
    ├── clustering.py
    ├── config.py
    ├── data_preprocessing.py
    ├── postprocessing.py
    ├── report.py
    ├── utils.py
    └── visualisation.py
```

`config.py` defines:

- DATA_DIR (expects aisdk-<date>.csv under data/)
- OUTPUT_DIR, plus sub-folders (filtered_chunks/, plots/, maps/, reports/).

`cli.py` runs the steps:

1. Ensure output directories exist
2. Preprocess AIS data (slow/stationary filter, COG/coastline filtering) → saves chunks to output/filtered_chunks/
3. Run multi-scale DBSCAN on those chunks → produce raw clusters
4. Merge clusters and compute port geometry (area, density, category) → produce final port list
5. Generate summary plots (output/plots/), interactive map (output/maps/), and a plain-text report (output/reports/)

## To run

1. Place your AIS CSV in `data/`, named exactly as configured in `config.py`. By default:

```bash
AIS_INPUT_FILE = os.path.join(DATA_DIR, "aisdk-2024-05-04.csv")
```

If filename or location is different, either the CSV has to be renamed accordingly or `AIS_INPUT_FILE` in `config.py` has to be updated.

2. From the project root, invoke the pipeline:

```bash
python3 -m cli.py
```

This will print progress messages to stdout and populate the following on success:
- `output/filtered_chunks/` – stationary + filtered chunks saved as `stationary_chunk_XXX.csv`
- `output/plots/` – PNG plots (geographic scatter, size histograms, etc.)
- `output/maps/multiscale_dbscan_map.html` – interactive Folium map of detected ports
- `output/reports/multiscale_dbscan_report.txt` – text summary of port detection results

3. Inspect results:

- Open `output/maps/multiscale_dbscan_map.html` in a browser to see port markers colored by category.
- View the PNGs in `output/plots/` (e.g., `geographic_distribution.png, port_size_histogram.png`).
- Read the plain-text report at `output/reports/multiscale_dbscan_report.txt` for detailed statistics and per-port summaries.

## Adjusting parameters

- Geographic bounds / SOG: Change `LATITUDE_BOUNDS`, `LONGITUDE_BOUNDS`, or `MAX_SOG` in `config.py` for different regions or vessel speeds.
- DBSCAN scales: Under `DBSCAN_CONFIGS` in `config.py`, you can add/remove scales or tweak `eps_km` and `min_samples` per scale.
- Port size thresholds: Modify `PORT_SIZE_CATEGORIES`, `MIN_PORT_AREA_KM2`, and `MAX_PORT_AREA_KM2` to change how ports are categorized by area.