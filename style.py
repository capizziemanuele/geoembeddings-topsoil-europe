"""Figure style shared by all plots: fonts, colours and small helpers."""

import io

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory
from PIL import Image

MM = 1 / 25.4
FULL = 180 * MM      # two-column width in Frontiers
HALF = 85 * MM       # one-column width

# One colour per data source, used the same way in every figure
C = {
    "emb": "#2F6F4E",        # AlphaEarth embeddings
    "emb_light": "#8DBF9F",
    "spec": "#2D5F9A",       # Sentinel-2 spectral indices
    "spec_light": "#9EBBDD",
    "ndvi": "#8AB6D6",       # NDVI + BSI
    "sg": "#C77C2F",         # SoilGrids (modelled)
    "sg_light": "#E8C197",
    "lucas": "#8C5A0B",      # LUCAS field data
    "base": "#9AA3AD",       # simple baselines
    "base_dark": "#5E6873",
    "lgbm": "#1F4E3A",
    "degr": "#B5542A",
    "stable": "#A3A9B0",
    "impr": "#2D6F9A",
    "ink": "#2B2B2B",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "land": "#F4F2EE",
    "border": "#FFFFFF",
    "outside": "#E9E7E2",
    "field": "#454B55",      # measured OC as a target
}

# Organic carbon: pale sand to deep green, like the thesis maps
OC_CMAP = LinearSegmentedColormap.from_list(
    "oc", ["#F3E6B8", "#D9C27A", "#A9B35E", "#5E9150", "#2E6B45", "#123D2A"])
ERR_CMAP = LinearSegmentedColormap.from_list(
    "err", ["#FBF3E4", "#F2C58A", "#E07B39", "#A63A2B", "#4A1530"])
HEX_CMAP = LinearSegmentedColormap.from_list(
    "hex", ["#E9F1EB", "#B7D3BF", "#72A585", "#2F6F4E", "#143A28"])
GREY_CMAP = LinearSegmentedColormap.from_list(
    "grey", ["#EEEFF1", "#C4C8CE", "#8C939D", "#4F5763", "#1F242C"])
# Diverging: the neutral colour sits exactly in the middle of the scale
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", [(0.0, "#8C510A"), (0.25, "#D8B365"), (0.5, "#F5F1E8"),
            (0.67, "#7FBFB0"), (0.85, "#1B6B5F"), (1.0, "#0B3F38")])


def setup():
    mpl.rcParams.update({
        "font.family": ["Open Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#4B5563",
        "axes.labelcolor": C["ink"],
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "text.color": C["ink"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Open Sans",
        "mathtext.it": "Open Sans:italic",
        "mathtext.bf": "Open Sans:bold",
    })


def panel(ax, letter, title=None, x=-0.02, y=1.02, y_ax=None):
    """Bold panel letter (Montserrat) with an optional short title.

    `y_ax` measures y on another axes, to line the title up with a taller neighbour.
    """
    tr = blended_transform_factory(ax.transAxes, y_ax.transAxes) if y_ax else ax.transAxes
    ax.text(x, y, letter, transform=tr, ha="right", va="bottom",
            fontsize=11, fontweight="bold", family="Montserrat")
    if title:
        ax.text(x + 0.03, y + 0.004, title, transform=tr, ha="left",
                va="bottom", fontsize=9, family="Montserrat", fontweight="semibold")


def ygrid(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, folder):
    """Save PDF (vector), PNG (300 dpi) and TIFF (600 dpi, LZW) for upload.

    Main figures (Figure_1 to Figure_7) are padded with white margins to exactly
    180 mm, the Frontiers two-column width, so the journal never has to enlarge
    them. Supplementary figures keep their own width.
    """
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    width = 180 if not name.startswith("Figure_S") else None
    for ext, dpi in (("png", 300), ("tif", 600)):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        img = Image.open(buf).convert("RGB")
        if width:
            img = fit_width(img, width, dpi)
        kw = {"compression": "tiff_lzw"} if ext == "tif" else {}
        img.save(folder / f"{name}.{ext}", dpi=(dpi, dpi), **kw)
    plt.close(fig)


def fit_width(img, width_mm, dpi):
    """Pad (or, if slightly wider, shrink) an image to an exact width in mm."""
    target = round(width_mm / 25.4 * dpi)
    w, h = img.size
    if w > target:
        return img.resize((target, round(h * target / w)), Image.LANCZOS)
    canvas = Image.new("RGB", (target, h), "white")
    canvas.paste(img, ((target - w) // 2, 0))
    return canvas
