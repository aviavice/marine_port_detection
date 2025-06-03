# src/visualisation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import folium
import logging

from src import config
from src.utils import ensure_directories_exist

import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

class Visualiser:
    """
    Takes the final_ports list (from PortPostProcessor) and:
      (a) Creates several informative PNGs under config.PLOTS_DIR
      (b) Builds an interactive Folium map under config.MAPS_DIR
    """

    def __init__(self, final_ports):
        self.ports = final_ports
        ensure_directories_exist(config.PLOTS_DIR, config.MAPS_DIR)

    def plot_density_heatmap(self):
        """
        2D KDE of vessel density across port centers (weighted by density).
        """
        lats = np.array([p['center_lat'] for p in self.ports])
        lons = np.array([p['center_lon'] for p in self.ports])
        densities = np.array([p['vessel_density'] for p in self.ports])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(
            x=lons, y=lats,
            weights=densities,
            cmap="Reds",
            fill=True,
            thresh=0.05,
            levels=100,
            bw_adjust=1.0,
            ax=ax
        )
        ax.scatter(lons, lats, c='black', s=10, alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Vessel‐Density 2D KDE (weighted by density)")
        out = os.path.join(config.PLOTS_DIR, "density_heatmap.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out}")
        plt.close(fig)

    def plot_area_violin(self):
        """
        Violin plot of port areas by category on a log scale.
        """
        data = [(p['category'], p['area_km2']) for p in self.ports]
        if not data:
            return
        df = pd.DataFrame(data, columns=["Category", "Area_km2"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            x="Category",
            y="Area_km2",
            data=df,
            scale="width",
            inner="quartile",
            palette=[config.PORT_SIZE_CATEGORIES[c]['color']
                     for c in df['Category'].unique()],
            ax=ax
        )
        ax.set_yscale("log")
        ax.set_xlabel("Port Category")
        ax.set_ylabel("Area (km², log scale)")
        ax.set_title("Port Area Distribution by Category (Log Scale)")
        ax.grid(True, which="both", alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        out = os.path.join(config.PLOTS_DIR, "area_violin_log.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out}")
        plt.close(fig)

    def plot_correlation_matrix(self):
        """
        Correlation heatmap of area, vessel density, point count, and detected scale.
        """
        records = []
        scale_mapping = {
            "small_harbors": 0,
            "local_ports":    1,
            "regional_ports": 2,
            "major_ports":    3
        }
        for p in self.ports:
            records.append({
                "area": p["area_km2"],
                "density": p["vessel_density"],
                "points": p["point_count"],
                "scale": scale_mapping[p["detected_scale"]]
            })
        df = pd.DataFrame(records)
        if df.empty:
            return
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Correlation Matrix: Area, Density, Points, Scale")
        out = os.path.join(config.PLOTS_DIR, "correlation_matrix.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out}")
        plt.close(fig)

    def plot_category_pie(self):
        """
        Pie (donut) chart showing proportion of ports in each size category.
        """
        counts = {
            cat: sum(1 for p in self.ports if p['category'] == cat)
            for cat in config.PORT_SIZE_CATEGORIES
        }
        if not counts:
            return
        labels = [cat.replace('_',' ').title() for cat in counts.keys()]
        sizes = list(counts.values())
        colors = [config.PORT_SIZE_CATEGORIES[c]['color'] for c in counts.keys()]

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops=dict(size=10)
        )
        ax.set_title("Port‐Category Composition (by count)")
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        out = os.path.join(config.PLOTS_DIR, "category_composition.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {out}")
        plt.close(fig)

    def plot_summary(self):
        """
        The original six‐panel summary plots.
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
        category_data = {
            cat: [p['area_km2'] for p in self.ports if p['category'] == cat]
            for cat in config.PORT_SIZE_CATEGORIES
        }
        # Remove empty categories
        category_data = {k: v for k, v in category_data.items() if v}

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
            sum(1 for p in self.ports if p['detected_scale'] == scale_key)
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
            <i style="background:grey;opacity:0.5;">&nbsp;&nbsp;&nbsp;</i> Uncathegorised<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_html = os.path.join(config.MAPS_DIR, "multiscale_dbscan_map.html")
        m.save(out_html)
        logger.info(f"Saved interactive map to: {out_html}")

    def plot_all(self):
        """
        Call each plotting method in sequence.
        """
        # Original six summary plots
        self.plot_summary()

        # Advanced plots (excluding coastline)
        self.plot_density_heatmap()
        self.plot_area_violin()
        self.plot_correlation_matrix()
        self.plot_category_pie()