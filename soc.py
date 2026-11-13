import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg


# Data

def load_data():
    """Join LUCAS 2018, the Earth Engine export and Trends.Earth by POINTID.

    Steps:
      1. keep LUCAS samples taken at 0-20 cm           -> 18,744
      2. keep points present in the Earth Engine export -> 18,492
      3. drop OC values that are not numbers            -> 18,462
         (31 are '< LOD' and 2 are '<0.0' in the raw file; 30 of them
         are among the matched points)
    """
    lucas = pd.read_csv(cfg.DATA_DIR / "LUCAS-SOIL-2018.csv", low_memory=False)
    gee = pd.read_csv(cfg.DATA_DIR / "AEF_LUCAS_all_2018.csv", low_memory=False)
    trends = pd.read_csv(cfg.DATA_DIR / "TrendsEarth_SDG1531_LUCAS_points.csv")

    lucas = lucas[lucas["Depth"] == "0-20 cm"]
    keep = ["POINTID", "OC", "N", "pH_H2O", "NUTS_0", "NUTS_2",
            "TH_LAT", "TH_LONG", "Elev", "LC0_Desc"]
    predictors = cfg.EMBEDDINGS + cfg.SPECTRAL + cfg.SOILGRIDS
    df = lucas[keep].merge(gee[["POINTID"] + predictors], on="POINTID", how="inner")

    sdg = ["STATUS_2023", "LPD_2008_2023", "LC_DEGRADATION_2015_2023", "SOC_PCT_CHANGE_2015_2023"]
    df = df.merge(trends[["POINTID"] + sdg], on="POINTID", how="left")

    for col in ["OC", "N", "pH_H2O", "Elev"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["OC"]).reset_index(drop=True).copy()

    # Targets. OC is in g/kg and strongly skewed, so models use log(OC + 1).
    df["logOC"] = np.log1p(df["OC"])
    df["logN"] = np.log1p(df["N"])

    # SoilGrids stores SOC in dg/kg: divide by 10 to get g/kg, like LUCAS.
    df["SG_SOC_gkg"] = df["SG_SOC"] / 10
    df["logSG_SOC"] = np.log1p(df["SG_SOC_gkg"])

    df["country"] = df["NUTS_0"].map(cfg.COUNTRY_NAMES)
    return df


# Models

def ridge():
    """Standardise, then ridge regression with the penalty tuned by inner CV."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=cfg.ALPHAS, cv=5))


def lgbm():
    return LGBMRegressor(**cfg.LGBM_PARAMS)


def folds(df, scheme):
    """Return a list of (train, test) index pairs.

    scheme = 'random' -> 5 shuffled folds
    scheme = 'nuts2'  -> 5 folds, all points of a NUTS2 region in the same fold
    """
    if scheme == "random":
        cv = KFold(cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        return list(cv.split(df))
    if scheme == "nuts2":
        cv = GroupKFold(cfg.N_FOLDS)
        return list(cv.split(df, groups=df["NUTS_2"].astype(str)))
    raise ValueError(scheme)


def cross_validate(df, features, target, splits, model=ridge):
    """Fit one model per fold. Returns out-of-fold predictions and fold R2."""
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    pred = np.full(len(df), np.nan)
    fold_r2 = []
    for train, test in splits:
        m = model().fit(X[train], y[train])
        pred[test] = m.predict(X[test])
        fold_r2.append(r2_score(y[test], pred[test]))
    return pred, np.array(fold_r2)


def errors_gkg(obs_gkg, pred_log):
    """RMSE, MAE and median absolute error in g/kg.

    Predictions are back-transformed with exp(x) - 1, without bias correction.
    """
    err = np.expm1(pred_log) - obs_gkg
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
    }
