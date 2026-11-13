"""Settings shared by all scripts."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

SEED = 0          # used for every shuffle, folds are reproducible
N_FOLDS = 5
ALPHAS = np.logspace(-3, 4, 30)   # candidate ridge penalties

# Leave-one-country-out: countries with fewer than 30 points are dropped
# before fitting. Scores are reported only for countries with at least 100 points.
LOCO_MIN_TRAIN = 30
LOCO_MIN_REPORT = 100

# Feature sets
EMBEDDINGS = [f"A{i:02d}" for i in range(64)]
SPECTRAL = ["NDVI", "EVI", "SAVI", "NIRv", "NDRE", "NDWI", "NDMI",
            "NBR2", "BSI", "SOIL_CI", "SOIL_SWIRR"]
SOILGRIDS = ["SG_SOC", "SG_N", "SG_PH", "SG_CLAY", "SG_SAND", "SG_SILT",
             "SG_CEC", "SG_BDOD", "SG_CFVO", "SG_OCD", "SG_OCS"]

# Main land cover classes in LUCAS
LAND_COVER = ["Cropland", "Grassland", "Woodland", "Shrubland", "Bareland"]

# LightGBM settings
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.08, num_leaves=31,
                   min_child_samples=20, colsample_bytree=0.6,
                   random_state=SEED, verbose=-1, n_jobs=2)

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "UK": "United Kingdom",
}
