import json
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory, offset_copy

import config as cfg
import style as st
from style import C

st.setup()
RES = cfg.RESULTS_DIR
OUT = cfg.FIGURES_DIR
LAEA = "EPSG:3035"
EXTENT = (2.55e6, 6.55e6, 1.35e6, 5.45e6)   # x0, x1, y0, y1 in LAEA metres

scores = pd.read_csv(RES / "model_scores.csv")
folds = pd.read_csv(RES / "fold_scores.csv")
points = pd.read_csv(RES / "points.csv.gz")
oof = pd.read_csv(RES / "oof_predictions.csv.gz")


def score(model, target="logOC", scheme="random", col="r2_fold_mean"):
    return float(pick(scores, col, model=model, target=target, scheme=scheme))


def fold_values(model, target="logOC", scheme="random"):
    m = (folds.model == model) & (folds.target == target) & (folds.scheme == scheme)
    return folds.loc[m, "r2"].to_numpy()


def pick(df, col, **conds):
    """Value of `col` in the single row that matches all conditions."""
    m = np.ones(len(df), bool)
    for k, v in conds.items():
        m &= (df[k] == v).to_numpy()
    return df.loc[m, col].iloc[0]


def countries():
    g = gpd.read_file(cfg.DATA_DIR / "europe_countries.geojson").to_crs(LAEA)
    lucas = set(cfg.COUNTRY_NAMES.values()) | {"Czechia"}
    g["in_study"] = g["name"].isin(lucas)
    return g


def points_gdf():
    g = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.TH_LONG, points.TH_LAT),
                         crs="EPSG:4326").to_crs(LAEA)
    return g


def base_map(ax, g):
    g[~g["in_study"]].plot(ax=ax, color=C["outside"], edgecolor="white", linewidth=0.4)
    g[g["in_study"]].plot(ax=ax, color=C["land"], edgecolor="#CFCAC0", linewidth=0.3)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")   # newer geopandas labels the axes with the CRS units
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def oc_ticks(ax_cb, vals=(2, 5, 10, 20, 50, 100, 200, 500)):
    ax_cb.set_ticks(vals)
    ax_cb.set_ticklabels([str(v) for v in vals])
    ax_cb.minorticks_off()


# Figure 2
def fig2():
    g = countries()
    p = points_gdf()
    p["pred"] = np.expm1(oof["Embeddings|logOC|nuts2"])
    p["abs_err"] = (p["pred"] - p["OC"]).abs()

    fig, axes = plt.subplots(2, 2, figsize=(st.FULL, st.FULL * 1.08),
                             gridspec_kw=dict(wspace=0.04, hspace=0.20))
    norm = LogNorm(2, 500)
    gkg = "g kg$^{-1}$"
    panels = [
        ("A", "Measured SOC (LUCAS 2018)", "OC", norm, st.OC_CMAP,
         f"median {points.OC.median():.1f} {gkg}"),
        ("B", "SoilGrids 2.0 SOC", "SG_SOC_gkg", norm, st.OC_CMAP,
         f"median {points.SG_SOC_gkg.median():.1f} {gkg}"),
        ("C", "Predicted SOC (embeddings)", "pred", norm, st.OC_CMAP,
         f"median {p.pred.median():.1f} {gkg}, NUTS2 regions withheld"),
        ("D", "Absolute error of predicted SOC", "abs_err", LogNorm(1, 300), st.ERR_CMAP,
         f"|C − A|, median {p.abs_err.median():.1f} {gkg}"),
    ]
    for ax, (letter, title, col, nrm, cmap, note) in zip(axes.flat, panels):
        base_map(ax, g)
        d = p.sort_values(col)
        sc = ax.scatter(d.geometry.x, d.geometry.y, c=d[col].clip(nrm.vmin, nrm.vmax),
                        cmap=cmap, norm=nrm, s=1.1, linewidths=0, rasterized=True)
        st.panel(ax, letter, title, x=0.03, y=1.065)
        ax.text(0.06, 1.012, note, transform=ax.transAxes, fontsize=8, color=C["muted"], va="bottom")
        if letter in ("B", "D"):
            cax = ax.inset_axes([1.0, 0.12, 0.028, 0.74])
            cb = fig.colorbar(sc, cax=cax, extend="max")
            cb.outline.set_linewidth(0.4)
            if letter == "B":
                oc_ticks(cb)
                cb.set_label("SOC (g kg$^{-1}$), A to C")
            else:
                oc_ticks(cb, (1, 3, 10, 30, 100, 300))
                cb.set_label("Absolute SOC error (g kg$^{-1}$)")
    st.save(fig, "Figure_2_maps", OUT)


# Figure 3
def fig3():
    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.60))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], wspace=0.72, hspace=0.62)

    # A: one row per predictor set, grouped
    ax = fig.add_subplot(gs[:, 0])
    groups = [
        ("Location and land cover", [
            ("Coordinates, linear (2)", "Position, linear", C["base"]),
            ("10 nearest sites (2)", "Position, nearest neighbours", C["base_dark"]),
            ("Land cover class (8)", "Land cover class", C["base"])]),
        ("Satellite predictors", [
            ("NDVI + BSI (2)", "NDVI + BSI", C["ndvi"]),
            ("Spectral indices (11)", "Spectral indices", C["spec"]),
            ("Embeddings (64)", "Embeddings", C["emb"]),
            ("Embeddings + indices (75)", "Embeddings + spectral", C["emb"])]),
        ("With SoilGrids properties", [
            ("SoilGrids (11)", "SoilGrids properties", C["sg"]),
            ("SoilGrids + embeddings (75)", "SoilGrids + embeddings", C["sg"])]),
    ]
    ticks, labels, heads = [], [], []
    yy = 0
    for gi, (gname, rows) in enumerate(groups):
        if gi:
            yy -= 0.45
            ax.axhline(yy + 0.25, color=C["grid"], lw=0.8)
        ticks.append(yy)
        labels.append(gname)
        heads.append(yy)
        yy -= 1
        for label, model, color in rows:
            r, n = score(model, scheme="random"), score(model, scheme="nuts2")
            ax.plot([n, r], [yy, yy], color=color, lw=1.6, alpha=0.45, zorder=2)
            ax.scatter(r, yy, s=30, color="white", edgecolor=color, linewidth=1.3, zorder=4)
            ax.scatter(n, yy, s=34, color=color, zorder=5)
            ax.text(1.05, yy, f"{r:.2f}", transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=8, color=C["muted"])
            ax.text(1.25, yy, f"{n:.2f}", transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=8, color=C["ink"], fontweight="bold")
            ticks.append(yy)
            labels.append(label)
            yy -= 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for t, v in zip(ax.get_yticklabels(), ticks):
        if v in heads:
            t.set_fontweight("bold")
            t.set_color(C["muted"])
            t.set_fontsize(8)
    top = heads[0]
    ax.text(1.05, top, "Random", transform=ax.get_yaxis_transform(), ha="center", va="center",
            fontsize=8, color=C["muted"], fontweight="bold")
    ax.text(1.25, top, "NUTS2", transform=ax.get_yaxis_transform(), ha="center", va="center",
            fontsize=8, color=C["ink"], fontweight="bold")
    ax.set_ylim(yy + 0.4, top + 0.6)
    ax.set_xlim(0, 0.65)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("R² on log(SOC + 1)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=C["ink"],
                      markeredgewidth=1.2, markersize=5.5, label="random folds"),
               Line2D([], [], marker="o", ls="none", color=C["ink"], markersize=5.5,
                      label="NUTS2 regions withheld")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              handletextpad=0.2, columnspacing=1.2)
    st.panel(ax, "A", "Measured SOC, ridge regression", x=0.0, y=1.03)

    # B and C: embeddings against spectral indices, regions withheld
    pair = [("Spectral indices", C["spec"]), ("Embeddings", C["emb"])]

    def pair_bars(ax, labels, getter, ylim, yticks):
        x = np.arange(len(labels))
        for i, (m, col) in enumerate(pair):
            xs = x + (i - 0.5) * 0.36
            vals = [getter(k, m) for k in labels]
            ax.bar(xs, vals, 0.33, color=col, zorder=2, label=m)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + ylim * 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels.values())
        ax.set_ylim(0, ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel("R², NUTS2 withheld")
        st.ygrid(ax)

    ax = fig.add_subplot(gs[0, 1])
    targets = {"logOC": "SOC", "logN": "Total N", "pH_H2O": "pH"}
    pair_bars(ax, targets, lambda k, m: score(m, k, "nuts2"), 0.7, [0, 0.2, 0.4, 0.6])
    st.panel(ax, "B", "Three soil properties", x=0.0, y=1.05)

    ax = fig.add_subplot(gs[1, 1])
    sub = pd.read_csv(RES / "subset_scores.csv").query("scheme == 'nuts2'")
    subsets = {"Cropland": "Cropland", "Cropland and bareland": "Cropland +\nbareland",
               "Low vegetation (annual NDVI < 0.4)": "Annual\nNDVI < 0.4"}
    pair_bars(ax, subsets, lambda k, m: pick(sub, "r2_fold_mean", subset=k, model=m), 0.3, [0, 0.1, 0.2, 0.3])
    st.panel(ax, "C", "Refitted within subsets", x=0.0, y=1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.34), ncol=2, handlelength=0.9)
    st.save(fig, "Figure_3_prediction_skill", OUT)


# Figure 4
def log_axis(ax, which="both", vals=(2, 5, 10, 20, 50, 100, 500)):
    t = np.log1p(vals)
    if which in ("x", "both"):
        ax.set_xticks(t)
        ax.set_xticklabels([str(v) for v in vals])
    if which in ("y", "both"):
        ax.set_yticks(t)
        ax.set_yticklabels([str(v) for v in vals])


def density_panel(ax, x, y, cmap, lim=(0.9, 6.7), vals=(2, 5, 20, 100, 500)):
    """Hexbin of two log(x + 1) variables with the 1:1 line."""
    hb = ax.hexbin(x, y, gridsize=30, extent=lim + lim, bins="log", mincnt=1,
                   cmap=cmap, linewidths=0, edgecolors="none")
    ax.plot(lim, lim, ls="--", color=C["muted"], lw=0.9)
    ax.text(lim[1] - 0.55, lim[1] - 0.25, "1:1", fontsize=8, color=C["muted"], rotation=45,
            ha="center", va="center")
    log_axis(ax, vals=vals)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    return hb


def fig4():
    lc = pd.read_csv(RES / "land_cover_errors.csv").query("scheme == 'nuts2'")
    y = np.log1p(points["OC"].to_numpy())
    pe = oof["Embeddings|logOC|nuts2"].to_numpy()

    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.42))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.62)
    gs = [outer[0]] + list(outer[1].subgridspec(1, 2, wspace=0.10))

    ax = fig.add_subplot(gs[0])
    hb = density_panel(ax, y, pe, st.HEX_CMAP)
    ax.set_xlabel("Measured SOC (g kg$^{-1}$)")
    ax.set_ylabel("Predicted SOC (g kg$^{-1}$)")
    ax.text(0.96, 0.05, f"Pooled R² = {score('Embeddings', scheme='nuts2', col='r2_pooled'):.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    cax = ax.inset_axes([0.06, 0.87, 0.36, 0.03])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal", ticks=[1, 10, 100])
    cb.ax.set_xticklabels(["1", "10", "100"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.outline.set_linewidth(0.3)
    cb.ax.set_title("sites per cell", fontsize=8, color=C["muted"], pad=2, loc="left")
    st.panel(ax, "A", "Embeddings, NUTS2 folds", x=0.0, y=1.03)

    order = ["Cropland", "Bareland", "Grassland", "Shrubland", "Woodland"]
    yy = np.arange(len(order))[::-1]
    for k, (metric, label, letter, title, xmax) in enumerate([
            ("medae_gkg", "Error (g kg$^{-1}$)", "B", "Median absolute error", 34),
            ("rmse_gkg", "Error (g kg$^{-1}$)", "C", "Root mean square error", 150)]):
        ax = fig.add_subplot(gs[k + 1])
        for i, (m, col) in enumerate([("Spectral indices", C["spec"]), ("Embeddings", C["emb"])]):
            vals = [pick(lc, metric, land_cover=o, model=m) for o in order]
            ys = yy + (0.5 - i) * 0.36
            ax.barh(ys, vals, 0.34, color=col, zorder=2, label=m)
            for yi, v in zip(ys, vals):
                ax.text(v + xmax * 0.012, yi, f"{v:.0f}", va="center", fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel(label)
        st.xgrid(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(yy)
        if k == 0:
            ns = [int(pick(lc, "n", land_cover=o)) for o in order]
            ax.set_yticklabels([f"{o} ({n:,})" for o, n in zip(order, ns)])
            ax.legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=2, handlelength=0.9)
            st.panel(ax, letter, title, x=-0.66, y=1.03)
        else:
            ax.set_yticklabels([])
            st.panel(ax, letter, title, x=0.06, y=1.03)
    st.save(fig, "Figure_4_errors", OUT)


# Figure 5
def fig5():
    t = pd.read_csv(RES / "country_transfer.csv").sort_values("r2_embeddings")
    s = json.load(open(RES / "country_transfer_summary.json"))
    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.62))
    # the map's legends sit under it, so the map can reach the right edge
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.0], wspace=0.07, right=0.985)

    ax = fig.add_subplot(gs[0])
    y = np.arange(len(t))
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot([r.r2_spectral, r.r2_embeddings], [yi, yi], color="#D5D9DF", lw=1.3, zorder=1)
    ax.scatter(t.r2_spectral, y, marker="D", s=18, color=C["spec"], zorder=3, label="Spectral indices")
    ax.scatter(t.r2_embeddings, y, marker="o", s=28, color=C["emb"], zorder=4, label="Embeddings")
    ax.axvline(0, color=C["ink"], lw=0.7, zorder=2)
    # spectral mean first: each legend row is then one predictor set and its mean
    ax.axvline(s["mean_r2_spectral"], color=C["spec"], lw=0.9, ls=":", zorder=2,
               label=f"mean, spectral ({s['mean_r2_spectral']:.2f})")
    ax.axvline(s["mean_r2_embeddings"], color=C["emb"], lw=0.9, ls="--", zorder=2,
               label=f"mean, embeddings ({s['mean_r2_embeddings']:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(t.country)
    ax.set_xlim(-0.62, 0.6)
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["−0.6", "−0.4", "−0.2", "0", "0.2", "0.4", "0.6"])
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel("R² on log(SOC + 1) in the withheld country")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2, handletextpad=0.3,
              handlelength=1.6, columnspacing=1.0)
    st.panel(ax, "A", "Leave one country out", x=0.0, y=1.01)
    ax_a = ax

    ax = fig.add_subplot(gs[1])
    g = countries()
    g = g.merge(t[["country", "r2_embeddings"]], left_on="name", right_on="country", how="left")
    g[~g["in_study"]].plot(ax=ax, color=C["outside"], edgecolor="white", linewidth=0.4)
    small = g[g["in_study"] & g["r2_embeddings"].isna()]
    small.plot(ax=ax, color="white", edgecolor="#7B8490", linewidth=0.5, hatch="//////")
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.55)
    g[g["r2_embeddings"].notna()].plot(ax=ax, column="r2_embeddings", cmap=st.DIV_CMAP, norm=norm,
                                       edgecolor="white", linewidth=0.5)
    ax.set_xlim(EXTENT[0], EXTENT[1] - 0.25e6)
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_axis_off()
    # x on the map, y on panel A: the colorbar sits on A's axis line and the legend on A's legend rows
    row = blended_transform_factory(ax.transAxes, ax_a.transAxes)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=st.DIV_CMAP)
    cax = ax.inset_axes([0.08, 0.0, 0.84, 0.024], transform=row)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-0.2, 0, 0.2, 0.4])
    cb.ax.set_xticklabels(["−0.2", "0", "0.2", "0.4"])
    cb.set_label("R² in the withheld country (embeddings)", fontsize=8)
    cb.outline.set_linewidth(0.3)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#7B8490", hatch="//////",
                             label="withheld, not scored (< 100 sites)"),
                       Patch(facecolor=C["outside"], edgecolor="none", label="not sampled")],
              loc="upper left", bbox_to_anchor=(0.08, -0.115), handlelength=1.4, handleheight=1.0,
              # shift left by the legend padding (0.5 + 0.4 font sizes) so the patches start under the colorbar
              bbox_transform=offset_copy(row, fig, x=-0.9 * 8, units="points"))
    # the map keeps its aspect and is shorter than A: put its title at A's height
    st.panel(ax, "B", "Embedding transfer by country", x=0.07, y=1.01, y_ax=ax_a)
    st.save(fig, "Figure_5_country_transfer", OUT)


# Figure 6
def fig6():
    d = json.load(open(RES / "soilgrids_direct.json"))
    lc = pd.read_csv(RES / "land_cover_errors.csv").query("scheme == 'random'")
    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 0.9], wspace=0.62)
    tcol = {"measured": C["field"], "soilgrids": C["sg"]}

    def two_targets(ax, labels, pairs, xlabel):
        yy = np.arange(len(labels))[::-1]
        for i, key in enumerate(["measured", "soilgrids"]):
            ys = yy + (0.5 - i) * 0.36
            vals = [p[i] for p in pairs]
            ax.barh(ys, vals, 0.34, color=tcol[key], zorder=2)
            for yi, v in zip(ys, vals):
                ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8)
        ax.set_yticks(yy)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        st.xgrid(ax)

    # A: same predictors, two targets
    ax = fig.add_subplot(gs[0])
    cases = [("Spectral, random", "Spectral indices", "random"), ("Spectral, NUTS2", "Spectral indices", "nuts2"),
             ("Embeddings, random", "Embeddings", "random"), ("Embeddings, NUTS2", "Embeddings", "nuts2")]
    two_targets(ax, [c[0] for c in cases],
                [(score(m, "logOC", sch), score(m, "logSG_SOC", sch)) for _, m, sch in cases],
                "R², mean of five test folds")
    st.panel(ax, "A", "Two targets", x=0.0, y=1.04)
    ax_a = ax

    # B: SoilGrids against the measurements
    ax = fig.add_subplot(gs[1])
    lu = np.log1p(points["OC"].to_numpy())
    sg = np.log1p(points["SG_SOC_gkg"].to_numpy())
    hb = density_panel(ax, lu, sg, st.GREY_CMAP)
    cax = ax.inset_axes([0.47, -0.47, 0.48, 0.035])   # below the panel, clear of the data
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal", ticks=[1, 10, 100])
    cb.ax.set_xticklabels(["1", "10", "100"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.outline.set_linewidth(0.3)
    ax.text(0.43, -0.453, "sites per cell", transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color=C["muted"])
    ax.set_xlabel("Measured SOC (g kg$^{-1}$)")
    ax.set_ylabel("SoilGrids SOC (g kg$^{-1}$)")
    ax.text(0.96, 0.05, f"R² = {d['r2_log']:.2f}\nr² = {d['r_squared_log']:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax_b = ax

    # C: within land cover classes
    ax = fig.add_subplot(gs[2])
    order = ["Cropland", "Bareland", "Grassland", "Shrubland", "Woodland"]
    two_targets(ax, order, [(pick(lc, "r2", land_cover=o, model="Embeddings"),
                             pick(lc, "r2", land_cover=o, model="Embeddings to SoilGrids")) for o in order],
                "R² within class")
    st.panel(ax, "C", "By land cover", x=0.0, y=1.04)

    # panel B is square and shorter: align its title with A and C
    fig.canvas.draw()
    top = ax_a.get_position().y1 + 0.04 * ax_a.get_position().height
    bb = ax_b.get_position()
    fig.text(bb.x0 - 0.004, top, "B", ha="right", va="bottom", fontsize=11, fontweight="bold", family="Montserrat")
    fig.text(bb.x0 + 0.008, top + 0.002, "SoilGrids vs LUCAS", ha="left", va="bottom", fontsize=9,
             family="Montserrat", fontweight="bold")

    fig.legend(handles=[Patch(color=tcol["measured"], label="target: measured SOC (LUCAS)"),
                        Patch(color=tcol["soilgrids"], label="target: SoilGrids SOC")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14), handlelength=1.0,
               columnspacing=2.0)
    st.save(fig, "Figure_6_reference_target", OUT)


# Figure 7
STATUS_LABELS = {1: "Degradation, persistent", 2: "Degradation, recent", 3: "Degradation, baseline",
                 4: "Stable", 5: "Improvement, baseline", 6: "Improvement, recent",
                 7: "Improvement, persistent"}
STATUS_COLORS = {1: "#8A3A1C", 2: "#B5542A", 3: "#DE9C73", 4: "#8C939D",
                 5: "#9EBBDD", 6: "#4F86BD", 7: "#23507F"}


def fig7():
    s = json.load(open(RES / "sdg_attribution.json"))
    te = pd.read_csv(cfg.DATA_DIR / "TrendsEarth_SDG1531_LUCAS_points.csv")
    ch = points.merge(te[["POINTID", "SOC_PCT_CHANGE_2015_2023"]], on="POINTID")
    ch = ch.dropna(subset=["SOC_PCT_CHANGE_2015_2023", "STATUS_2023"])["SOC_PCT_CHANGE_2015_2023"]
    st_oc = points.dropna(subset=["STATUS_2023"])

    fig = plt.figure(figsize=(st.FULL, st.FULL * 0.62))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.08], height_ratios=[1, 1.2], wspace=0.75, hspace=0.62)

    # A: which component triggers degradation
    ax = fig.add_subplot(gs[0, 0])
    n = s["degraded_reporting"]
    items = [("Land productivity", s["flag_productivity"], C["spec"]),
             ("Land cover", s["flag_land_cover"], C["emb"]),
             ("SOC stock change", s["flag_soil_carbon"], C["sg"])]
    y = np.arange(3)[::-1]
    for yi, (lab, v, col) in zip(y, items):
        ax.barh(yi, 100 * v / n, color=col, height=0.6, zorder=2)
        ax.text(100 * v / n + 2, yi, f"{100 * v / n:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Share of the {n:,} degraded sites (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    st.panel(ax, "A", "Component that flags the degraded sites", x=0.0, y=1.05)

    # B: observation periods
    ax = fig.add_subplot(gs[0, 1])
    spans = [("Baseline", 2000, 2015, C["stable"]), ("Land productivity", 2008, 2023, C["spec"]),
             ("Land cover", 2015, 2023, C["emb"]), ("SOC stock change", 2015, 2023, C["sg"])]
    y = np.arange(len(spans))[::-1]
    for yi, (lab, a0, b0, col) in zip(y, spans):
        ax.barh(yi, b0 - a0, left=a0, color=col, height=0.56, zorder=2)
    ax.axvline(2018, color=C["lucas"], lw=1.0, ls="--", zorder=3)
    ax.text(2018.4, 3.45, "LUCAS 2018", ha="left", va="center", fontsize=8, color=C["lucas"])
    ax.set_yticks(y)
    ax.set_yticklabels([sp[0] for sp in spans])
    ax.set_xlim(1999, 2024.5)
    ax.set_xticks([2000, 2008, 2015, 2023])
    ax.set_ylim(-0.7, len(spans) - 0.4)
    ax.set_xlabel("Year")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    st.panel(ax, "B", "Period covered by each component", x=0.0, y=1.05)

    # C: modeled SOC stock change at the sites
    ax = fig.add_subplot(gs[1, 0])
    zero = (ch == 0).sum()
    nz = ch[ch != 0]
    bins = np.arange(-77.5, 32.6, 5)
    counts, _, _ = ax.hist(nz, bins=bins, color=C["sg"], edgecolor="white", linewidth=0.5, zorder=2)
    ax.plot([-10, -10], [0, counts.max() * 1.02], color=C["ink"], lw=0.8, ls="--")
    ax.text(-11.5, counts.max() * 1.02, "−10%\nthreshold", ha="right", va="top", fontsize=8)
    ax.text(0.02, 1.0, f"{zero:,} sites ({100 * zero / len(ch):.1f}%)\nwith no change are not shown",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=C["muted"])
    ax.set_xlabel("Modeled SOC stock change, 2015–2023 (%)")
    ax.set_ylabel("Sites")
    ax.set_ylim(0, counts.max() * 1.42)
    st.ygrid(ax)
    st.panel(ax, "C", f"SOC stock change at {len(nz):,} sites", x=0.0, y=1.05)

    # D: measured SOC by status class, median and interquartile range,
    # with the number of sites per class and the median of all sites
    ax = fig.add_subplot(gs[1, 1])
    codes = list(range(1, 8))
    y = np.arange(len(codes))[::-1]
    all_med = st_oc["OC"].median()
    ax.plot([all_med, all_med], [-0.6, len(codes) - 0.62], color=C["muted"], lw=0.8, ls=":", zorder=1)
    ax.text(all_med, len(codes) - 0.55, f"all sites {all_med:.1f}", ha="center", va="bottom",
            fontsize=7, color=C["muted"])
    ax.text(78, len(codes) - 0.55, "Sites", ha="right", va="bottom", fontsize=7, color=C["muted"])
    for yi, c in zip(y, codes):
        v = st_oc.loc[st_oc.STATUS_2023 == c, "OC"]
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        ax.plot([q1, q3], [yi, yi], color=STATUS_COLORS[c], lw=2.2, alpha=0.45, solid_capstyle="round")
        ax.scatter(med, yi, s=30, color=STATUS_COLORS[c], zorder=3)
        ax.text(q3 + 1.2, yi, f"{med:.1f}", va="center", fontsize=8)
        ax.text(78, yi, f"n = {len(v):,}", ha="right", va="center", fontsize=7, color=C["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels([STATUS_LABELS[c] for c in codes])
    ax.set_xlim(0, 78)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.spines["bottom"].set_bounds(0, 50)
    ax.set_ylim(-0.6, len(codes) + 0.1)
    ax.set_xlabel("Measured SOC in 2018 (g kg$^{-1}$)", x=25 / 78)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    st.panel(ax, "D", "Measured SOC by 2023 status class", x=0.0, y=1.05)
    st.save(fig, "Figure_7_sdg", OUT)


FIGS = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7}


# Supplementary
def figS2():
    lc = pd.read_csv(RES / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(st.HALF * 1.25, st.HALF * 0.95))
    for m, col, mk in [("Spectral indices", C["spec"], "D"), ("Embeddings", C["emb"], "o")]:
        g = lc[lc.model == m].groupby("fraction").agg(n=("n_train", "mean"), r2=("r2", "mean"),
                                                     lo=("r2", "min"), hi=("r2", "max")).reset_index()
        ax.fill_between(g.n, g.lo, g.hi, color=col, alpha=0.18, lw=0)
        ax.plot(g.n, g.r2, marker=mk, color=col, lw=1.4, markersize=4, label=m)
        for _, r in g.iloc[[0, 3, -1]].iterrows():
            ax.annotate(f"{r.r2:.2f}", (r.n, r.r2), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, color=col)
    ax.set_xscale("log")
    ax.set_xticks([400, 1000, 4000, 15000])
    ax.set_xticklabels(["400", "1,000", "4,000", "15,000"])
    ax.minorticks_off()
    ax.set_ylim(0.2, 0.55)
    ax.set_xlabel("Training sites (sampled within each country, log scale)")
    ax.set_ylabel("R² on all remaining sites")
    ax.legend(loc="lower right")
    st.ygrid(ax)
    st.save(fig, "Figure_S4_learning_curve", OUT)


def figS3():
    corr = pd.read_csv(RES / "embedding_correlations.csv", index_col=0)
    top = corr["logOC"].abs().sort_values(ascending=False).index[:12]
    m = corr.loc[top].sort_values("logOC")
    m = m.rename(columns={"logOC": "log(SOC + 1)", "SG clay": "Clay (SG)", "SG pH": "pH (SG)",
                          "SG bulk density": "Bulk density (SG)"})
    fig, ax = plt.subplots(figsize=(st.FULL * 0.72, st.FULL * 0.52))
    im = ax.imshow(m.values, cmap=st.DIV_CMAP, vmin=-0.85, vmax=0.85, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else C["ink"])
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index)
    ax.axvline(0.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson correlation")
    cb.outline.set_linewidth(0.3)
    st.save(fig, "Figure_S2_embedding_correlations", OUT)


def figS4():
    counts = points.groupby(["NUTS_0", "LC0_Desc"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total")
    classes = ["Cropland", "Grassland", "Woodland", "Shrubland", "Bareland"]
    other = [c for c in counts.columns if c not in classes + ["total"]]
    counts["Other"] = counts[other].sum(axis=1)
    cols = {"Cropland": "#D9B44A", "Grassland": "#8DBF6A", "Woodland": "#2F6F4E",
            "Shrubland": "#A38A5C", "Bareland": "#C9B8A6", "Other": "#9AA3AD"}
    fig, ax = plt.subplots(figsize=(st.FULL * 0.62, st.FULL * 0.72))
    left = np.zeros(len(counts))
    y = np.arange(len(counts))
    for c in classes + ["Other"]:
        ax.barh(y, counts[c], left=left, color=cols[c], height=0.72, label=c, zorder=2)
        left += counts[c].to_numpy()
    for yi, t in zip(y, counts.total):
        ax.text(max(t + 30, 130), yi, f"{t:,}", va="center", fontsize=8, color=C["muted"])
    ax.axvline(100, color=C["ink"], lw=0.8, ls="--")
    ax.annotate("100 sites: minimum to score\na withheld country", xy=(100, -0.65), xytext=(700, 0.3),
                fontsize=8, va="center", arrowprops=dict(arrowstyle="-", color=C["ink"], lw=0.6))
    ax.set_yticks(y)
    ax.set_yticklabels([cfg.COUNTRY_NAMES[c] for c in counts.index])
    ax.set_xlabel("LUCAS 2018 sites used in the analysis")
    ax.set_xlim(0, 4300)
    ax.set_ylim(-0.95, len(counts) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    st.xgrid(ax)
    ax.legend(loc="lower right", ncol=1, handlelength=1, title="LUCAS land cover", title_fontsize=8)
    st.save(fig, "Figure_S1_sites_by_country", OUT)


def figS5():
    g = countries()
    p = points_gdf().dropna(subset=["STATUS_2023"])
    fig, ax = plt.subplots(figsize=(st.FULL * 0.62, st.FULL * 0.62))
    base_map(ax, g)
    groups = [("Degradation (1 to 3)", [1, 2, 3], C["degr"]), ("Stable (4)", [4], C["stable"]),
              ("Improvement (5 to 7)", [5, 6, 7], C["impr"])]
    for lab, codes, col in groups:
        d = p[p.STATUS_2023.isin(codes)]
        ax.scatter(d.geometry.x, d.geometry.y, s=1.0, color=col, linewidths=0, rasterized=True,
                   label=f"{lab}: {len(d):,}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, markerscale=6,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0.0)
    st.save(fig, "Figure_S5_trendsearth_status", OUT)


FIGS.update({"figS2": figS2, "figS3": figS3, "figS4": figS4, "figS5": figS5})


if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        print("making", name)
        FIGS[name]()
