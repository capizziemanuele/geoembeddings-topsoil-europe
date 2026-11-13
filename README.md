# geoembeddings-topsoil-europe

Code for the paper "AlphaEarth embeddings for topsoil organic carbon prediction
across Europe: evaluation against field and modeled references" (Capizzi,
Schillaci and Giuliani, Frontiers in Soil Science, submitted to Frontiers in Soil Science).

We compare Google AlphaEarth embeddings with Sentinel-2 spectral indices for
predicting topsoil organic carbon at 18,462 LUCAS 2018 sites, and check how the
scores change when the target is SoilGrids instead of the field measurement.

## Files

- `gee/` - Earth Engine scripts that sample the predictors at the LUCAS points
- `config.py` - paths, seed, feature lists and model settings
- `soc.py` - data loading, models, folds and error metrics
- `run_analysis.py` - all analyses, writes the tables in `results/`
- `sensitivity.py` - extra checks on the ridge tuning, and soil data per country
- `make_figures.py`, `make_diagrams.py`, `style.py` - figures of the paper
- `results/` - the output tables used in the paper. `oof_predictions.csv.gz`
  holds the out-of-fold predictions without site identifiers; its rows follow
  the order of `results/points.csv.gz` (rebuilt by `run_analysis.py`)

## Data

The input tables are not included because of the LUCAS terms of use. See
`data/README.md` for how to get them.

## Run

```bash
pip install -r requirements.txt
python run_analysis.py
python sensitivity.py
python make_figures.py
python make_diagrams.py
```

`run_analysis.py` takes about 15 minutes. `requirements.txt` gives minimum
versions only. The run behind the paper used Python 3.11.15, pandas 3.0.2,
numpy 2.4.4, scikit-learn 1.8.0 and lightgbm 4.7.0 (recorded in
`results/manifest.json`); `requirements-pinned.txt` repeats these. Versions of
matplotlib and geopandas were not recorded.

`make_figures.py` and `make_diagrams.py` need the site-level table
`results/points.csv.gz`, which holds LUCAS coordinates and measured OC and is
therefore not distributed (see `.gitignore`). `run_analysis.py` rebuilds it from
the three input tables, so run it first.

## License

Code: MIT. The data keep their own licenses.

Contact: Emanuele Capizzi, <capizzi.emanuele@gmail.com>
