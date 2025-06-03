# src/visualisation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import folium
import logging

from src import config
from src.utils import ensure_directories_exist

logger = logging.getLogger(__name__)

class Visualiser:
    """
    Takes the final_ports list (from PortPostProcessor) and:
      1) Creates six separate PNGs under config.PLOTS_DIR
      2) Builds an interactive Folium map under config.MAPS_DIR
    """

    def __init__(self, final_ports):
        self.ports = final_ports
        ensure_directories_exist(config.PLOTS_DIR, config.MAPS_DIR)

    def plot_summary(self):
        """
        Instead of one 2×3 figure, save each panel as its own PNG.
        """

        # ────────────────────────────────────────────────────────────────
        # 1) Geographic distribution of ports by detected_scale
        # ────────────────────────────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        scale_colors = {
            'major_ports':   'red',
            'regional_ports':'orange',
            'local_ports':   'blue',
            'small_harbors': 'green'
        }
        for scale_key, color in scale_colors.items():
            pts = [p for p in self.ports if p['detected_scale'] == scale_key]
            if pts:
                lats = [p['center_lat'] for p in pts]
                lons = [p['center_lon'] for p in pts]
                areas = [p['area_km2'] for p in pts]
                ax1.scatter(
                    lons, lats,
                    s=[a * 100 for a in areas],
                    c=color,
                    alpha=0.7,
                    label=f"{scale_key.replace('_',' ').title()} (n={len(pts)})",
                    edgecolors='black',
                    linewidths=0.5
                )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('DBSCAN Multi-Scale Detection')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        out1 = os.path.join(config.PLOTS_DIR, "geographic_distribution.png")
        fig1.savefig(out1, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out1}")
        plt.close(fig1)

        # ────────────────────────────────────────────────────────────────
        # 2) Port size histogram by category
        # ────────────────────────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        category_data = {}
        for cat, specs in config.PORT_SIZE_CATEGORIES.items():
            cat_areas = [p['area_km2'] for p in self.ports if p['category'] == cat]
            if cat_areas:
                category_data[cat] = cat_areas

        if category_data:
            ax2.hist(
                category_data.values(),
                bins=15,
                alpha=0.7,
                label=list(category_data.keys()),
                color=[config.PORT_SIZE_CATEGORIES[c]['color'] for c in category_data.keys()]
            )
            ax2.set_xlabel('Port Area (km²)')
            ax2.set_ylabel('Number of Ports')
            ax2.set_title('Port Size Distribution by Category')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        out2 = os.path.join(config.PLOTS_DIR, "port_size_histogram.png")
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out2}")
        plt.close(fig2)

        # ────────────────────────────────────────────────────────────────
        # 3) Number of ports per DBSCAN scale (bar chart)
        # ────────────────────────────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        scale_counts = {
            scale_key.replace('_',' ').title():
            len([p for p in self.ports if p['detected_scale'] == scale_key])
            for scale_key in scale_colors
        }
        if scale_counts:
            bars = ax3.bar(
                scale_counts.keys(),
                scale_counts.values(),
                color=[scale_colors[k.lower().replace(' ','_')] for k in scale_counts.keys()],
                alpha=0.7,
                edgecolor='black'
            )
            for bar, val in zip(bars, scale_counts.values()):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val}",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        ax3.set_ylabel('Number of Ports')
        ax3.set_title('DBSCAN Scale Effectiveness')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, fontsize=9)
        out3 = os.path.join(config.PLOTS_DIR, "ports_per_scale.png")
        fig3.savefig(out3, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out3}")
        plt.close(fig3)

        # ────────────────────────────────────────────────────────────────
        # 4) Scatter of port area vs vessel density
        # ────────────────────────────────────────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        areas = [p['area_km2'] for p in self.ports]
        densities = [p['vessel_density'] for p in self.ports]
        colors = [config.PORT_SIZE_CATEGORIES[p['category']]['color'] for p in self.ports]
        ax4.scatter(areas, densities, c=colors, alpha=0.7, s=60, edgecolors='black')
        ax4.set_xlabel('Port Area (km²)')
        ax4.set_ylabel('Vessel Density (vessels/km²)')
        ax4.set_title('Port Area vs Vessel Density')
        ax4.grid(True, alpha=0.3)
        out4 = os.path.join(config.PLOTS_DIR, "area_vs_density.png")
        fig4.savefig(out4, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out4}")
        plt.close(fig4)

        # ────────────────────────────────────────────────────────────────
        # 5) Text summary box (DBSCAN stats)
        # ────────────────────────────────────────────────────────────────
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        ax5.axis('off')
        total_ports = len(self.ports)
        total_vessels = sum(p['point_count'] for p in self.ports)
        area_vals = areas if areas else [0]
        text = (
            f"DBSCAN SUMMARY\n\n"
            f"• Total Ports Detected: {total_ports}\n"
            f"• Total Stationary Vessels: {total_vessels:,}\n"
            f"• Area Range: {min(area_vals):.3f} – {max(area_vals):.3f} km²\n"
            f"• Average Area: {np.mean(area_vals):.3f} km²\n"
            f"\nPORT CATEGORIES:\n"
        )
        for cat in config.PORT_SIZE_CATEGORIES:
            count = sum(1 for p in self.ports if p['category'] == cat)
            specs = config.PORT_SIZE_CATEGORIES[cat]
            text += f"  • {cat} ({specs['min']:.2f}–{specs['max']:.2f} km²): {count} ports\n"
        text += "\nDBSCAN SCALE DISTRIBUTION:\n"
        for scale, cnt in scale_counts.items():
            text += f"  • {scale}: {cnt} ports\n"

        ax5.text(0, 1, text, va='top', family='monospace', fontsize=10)
        out5 = os.path.join(config.PLOTS_DIR, "text_summary.png")
        fig5.savefig(out5, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out5}")
        plt.close(fig5)

        # ────────────────────────────────────────────────────────────────
        # 6) Top 10 ports by area (horizontal bar chart)
        # ────────────────────────────────────────────────────────────────
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        # Sort ports descending by area, take top 10
        top10 = sorted(self.ports, key=lambda p: p['area_km2'], reverse=True)[:10]
        if top10:
            labels = [f"Port {i+1}" for i in range(len(top10))]
            areas_top = [p['area_km2'] for p in top10]
            y_pos = np.arange(len(top10))
            ax6.barh(y_pos, areas_top, color='skyblue', edgecolor='black')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(labels)
            ax6.invert_yaxis()
            ax6.set_xlabel('Area (km²)')
            ax6.set_title('Top 10 Ports by Area')
            for i, v in enumerate(areas_top):
                ax6.text(v + 0.1, i, f"{v:.3f}", va='center', fontsize=9)
            ax6.grid(True, alpha=0.3, axis='x')
        out6 = os.path.join(config.PLOTS_DIR, "top10_ports.png")
        fig6.savefig(out6, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out6}")
        plt.close(fig6)

    def make_interactive_map(self):
        """
        Build a Folium map showing each port’s center as a circle marker
        colored by category, plus a legend. Save as “multiscale_dbscan_map.html”.
        """
        center_lat = np.mean([p['center_lat'] for p in self.ports])
        center_lon = np.mean([p['center_lon'] for p in self.ports])

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        for p in self.ports:
            folium.CircleMarker(
                location=[p["center_lat"], p["center_lon"]],
                radius=5,
                color="black",
                fillColor=p["color"],
                fillOpacity=0.8,
                weight=1,
                popup=folium.Popup(
                    f"<b>Port category:</b> {p['category']}<br>"
                    f"<b>Area:</b> {p['area_km2']:.3f} km²<br>"
                    f"<b>Vessel density:</b> {p['vessel_density']:.1f} vessels/km²",
                    max_width=250
                )
            ).add_to(m)

        # Legend (HTML/CSS snippet)
        legend_html = """
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 180px; height: 140px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:12px; padding: 10px;">
            <b>Port Categories</b><br>
            <i style="background:red;opacity:0.7;">&nbsp;&nbsp;&nbsp;</i> Major Commercial<br>
            <i style="background:orange;opacity:0.7;">&nbsp;&nbsp;&nbsp;</i> Regional<br>
            <i style="background:blue;opacity:0.7;">&nbsp;&nbsp;&nbsp;</i> Local/Industrial<br>
            <i style="background:green;opacity:0.7;">&nbsp;&nbsp;&nbsp;</i> Small Harbor<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_html = os.path.join(config.MAPS_DIR, "multiscale_dbscan_map.html")
        m.save(out_html)
        logger.info(f"Saved interactive map to: {out_html}")