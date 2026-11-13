import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as cfg
import style as st

st.setup()
OUT = cfg.FIGURES_DIR
GREY = "#B3B3B3"


def box(ax, x, y, w, h, color, text="", tc="white", fs=8, weight="normal", r=0.03,
        ec="none", lw=1.0, ls="-", gradient=None, family="Open Sans", z=2, align="center"):
    """Rounded box in axes-data units. `gradient` = (left colour, right colour)."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=color if gradient is None else "none", ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    if gradient is not None:
        cmap = LinearSegmentedColormap.from_list("g", gradient)
        im = ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x, x + w, y, y + h), cmap=cmap,
                       aspect="auto", zorder=z, interpolation="bicubic")
        im.set_clip_path(p)
    if text:
        tx = x + w / 2 if align == "center" else x + 0.02
        ax.text(tx, y + h / 2, text, ha=align, va="center", color=tc, fontsize=fs,
                fontweight=weight, family=family, zorder=z + 1, linespacing=1.25)
    return p


def arrow(ax, pts, color=GREY, lw=1.8, head=True):
    """Polyline arrow through a list of (x, y) points."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round", zorder=1)
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=9,
                        color=color, lw=lw, zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def workflow():
    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.60))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1.667)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    FS = 8

    # LUCAS band
    box(ax, 0.02, 0.83, 1.13, 0.13, "#7A4D00", r=0.06,
        text="LUCAS Soil 2018: survey points and topsoil analyses, joined by point ID\n"
             "18,462 sites  ·  28 countries  ·  254 NUTS2 regions  ·  0–20 cm  ·  SOC in g kg$^{-1}$",
        tc="white", fs=FS)

    # Earth Engine container
    box(ax, 0.02, 0.05, 0.60, 0.70, "none", ec="#9CA3AF", lw=1.3, ls=(0, (2, 2)), r=0.04, z=0)
    ax.text(0.115, 0.765, "Google Earth Engine, sampled at each site", ha="left", va="center",
            fontsize=FS, color="#6B7280", fontweight="bold")
    pills = [
        (0.585, "AlphaEarth embeddings\n64 dimensions · 10 m · 2018", None, ("#4A56E0", "#B8338C"), "white"),
        (0.425, "Sentinel-2 spectral indices\n11 indices · 10–20 m · 2018", "#00753C", None, "white"),
        (0.265, "SoilGrids 2.0\n11 properties incl. SOC · 250 m", "#B8500F", None, "white"),
        (0.105, "Trends.Earth SDG 15.3.1\n4 layers · 250 m · 2000–2023", None, ("#C62828", "#B8500F"), "white"),
    ]
    for y, t, c, grad, tc in pills:
        box(ax, 0.16, y, 0.42, 0.115, c if c else "none", text=t, tc=tc, fs=FS, r=0.055, gradient=grad)
        arrow(ax, [(0.09, y + 0.10), (0.09, y + 0.0575), (0.16, y + 0.0575)])
    ax.plot([0.09, 0.09], [0.83, 0.20], color=GREY, lw=1.8, zorder=1)
    ax.text(0.063, 0.42, "one row per site", rotation=90, ha="center", va="center", fontsize=FS, color="#9CA3AF")

    # models and validation
    xm, wm = 0.69, 0.45
    box(ax, xm, 0.50, wm, 0.18, "#737373", r=0.05,
        text="Models\nridge regression (main)\nLightGBM (check and benchmark)", fs=FS)
    box(ax, xm, 0.26, wm, 0.17, "#126E87", r=0.05,
        text="Validation\nrandom 5-fold  ·  NUTS2 regions\nleave one country out", fs=FS)
    arrow(ax, [(0.58, 0.6425), (0.645, 0.6425), (0.645, 0.59), (xm, 0.59)])
    arrow(ax, [(0.58, 0.4825), (0.645, 0.4825), (0.645, 0.59)], head=False)
    arrow(ax, [(0.58, 0.3225), (0.645, 0.3225), (0.645, 0.59)], head=False)
    arrow(ax, [(xm + wm / 2, 0.50), (xm + wm / 2, 0.43)])

    # outputs
    xo, wo = 1.22, 0.43
    outs = [(0.56, "Embeddings vs spectral indices\nmeasured SOC, N and pH"),
            (0.39, "Measured vs modeled target\nmeasured SOC and SoilGrids SOC"),
            (0.105, "SOC stock change in the\nSDG 15.3.1 outputs")]
    for y, t in outs:
        box(ax, xo, y, wo, 0.115, "#003F22", text=t, fs=FS, r=0.055)
    arrow(ax, [(xm + wm, 0.345), (1.18, 0.345), (1.18, 0.6175), (xo, 0.6175)])
    arrow(ax, [(1.18, 0.4475), (xo, 0.4475)])
    arrow(ax, [(0.58, 0.1625), (xo, 0.1625)])
    ax.text(0.915, 0.178, "descriptive analysis, no model", ha="center", va="bottom", fontsize=FS, color="#6B7280")

    # inset map of the sites coloured by country
    pts = pd.read_csv(cfg.RESULTS_DIR / "points.csv.gz")
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.TH_LONG, pts.TH_LAT), crs=4326).to_crs(3035)
    iax = fig.add_axes([0.715, 0.70, 0.27, 0.29])
    codes = sorted(pts.NUTS_0.unique())
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(codes)}
    iax.scatter(g.geometry.x, g.geometry.y, s=0.15, c=[colors[c] for c in pts.NUTS_0], linewidths=0,
                rasterized=True, alpha=0.85)
    iax.set_aspect("equal")
    iax.set_axis_off()
    iax.text(-0.02, 0.05, "sites by\ncountry", transform=iax.transAxes, ha="right", fontsize=FS,
             color="#6B7280")
    st.save(fig, "Figure_1_workflow", OUT)


def sdg_framework():
    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.42))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 0.84)
    ax.set_axis_off()
    items = [
        (0.05, "#007038", "white", "Land cover", "Transitions between\nland cover classes",
         "Default: ESA CCI or national maps"),
        (0.69, "#004AAD", "white", "Land productivity", "Trend, state and performance of\nvegetation productivity (NPP\nproxied by NDVI)",
         "Default: MODIS / AVHRR NDVI"),
        (1.33, "#A0470F", "white", "Carbon stocks", "Change in SOC stock (0–30 cm),\na proxy for carbon stocks\nabove and below ground",
         "Default: SoilGrids, IPCC factors"),
    ]
    for x, fc, tc, title, body, src in items:
        box(ax, x, 0.46, 0.62, 0.35, fc, ec="#111111", lw=1.0, r=0.05)
        ax.text(x + 0.31, 0.765, title, ha="center", va="center", color=tc, fontsize=10,
                family="Montserrat", fontweight="bold")
        ax.text(x + 0.31, 0.63, body, ha="center", va="center", color="white", fontsize=8.5, linespacing=1.2)
        ax.text(x + 0.31, 0.49, src, ha="center", va="center", color="#F0F0F0", fontsize=8)
        arrow(ax, [(x + 0.31, 0.46), (x + 0.31, 0.345)])
    box(ax, 0.05, 0.20, 1.90, 0.14, "white", ec="#FF5757", lw=1.8, r=0.07,
        text="", z=2)
    ax.text(1.0, 0.295, "One out, all out", ha="center", va="center", fontsize=10,
            family="Montserrat", fontweight="bold")
    ax.text(1.0, 0.24, "A land unit is classified as degraded if any sub-indicator shows degradation.",
            ha="center", va="center", fontsize=8.5)
    arrow(ax, [(1.0, 0.20), (1.0, 0.16), (0.70, 0.16), (0.70, 0.115)])
    arrow(ax, [(1.0, 0.16), (1.30, 0.16), (1.30, 0.115)])
    box(ax, 0.50, 0.03, 0.40, 0.085, "#3F7A2D", text="Not degraded", fs=9.5, weight="bold", r=0.04,
        family="Montserrat")
    box(ax, 1.10, 0.03, 0.40, 0.085, "#C62828", text="Degraded", fs=9.5, weight="bold", r=0.04,
        family="Montserrat")
    st.save(fig, "Figure_S3_sdg_framework", OUT)


if __name__ == "__main__":
    workflow()
    sdg_framework()
