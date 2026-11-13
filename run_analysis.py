import json
import platform

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import BallTree, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
import soc

OUT = cfg.RESULTS_DIR
OUT.mkdir(exist_ok=True)
EMB, SPEC, SG = cfg.EMBEDDINGS, cfg.SPECTRAL, cfg.SOILGRIDS

df = soc.load_data()
splits = {"random": soc.folds(df, "random"), "nuts2": soc.folds(df, "nuts2")}


# 1. data
oc = df["OC"]
summary = {
    "n_points": len(df),
    "n_countries": int(df["NUTS_0"].nunique()),
    "n_nuts2": int(df["NUTS_2"].nunique()),
    "oc_mean": oc.mean(), "oc_median": oc.median(), "oc_sd": oc.std(),
    "oc_min": oc.min(), "oc_max": oc.max(),
    "oc_p05": oc.quantile(0.05), "oc_p95": oc.quantile(0.95),
    "sg_soc_mean": df["SG_SOC_gkg"].mean(), "sg_soc_median": df["SG_SOC_gkg"].median(),
    "sg_soc_max": df["SG_SOC_gkg"].max(),
}

# distance to the nearest other LUCAS point and correlation of their log OC
xy = np.radians(df[["TH_LAT", "TH_LONG"]].to_numpy())
dist, idx = BallTree(xy, metric="haversine").query(xy, k=2)
summary["nn_distance_km_median"] = float(np.median(dist[:, 1]) * 6371)
summary["nn_logoc_correlation"] = float(
    np.corrcoef(df["logOC"], df["logOC"].to_numpy()[idx[:, 1]])[0, 1])
df.groupby(["NUTS_0", "country"]).size().rename("points").reset_index() \
  .to_csv(OUT / "country_counts.csv", index=False)
df["LC0_Desc"].value_counts().rename("points").to_csv(OUT / "land_cover_counts.csv")
print("data:", summary["n_points"], "points")


# 2. cross-validated models
def knn():
    return make_pipeline(StandardScaler(),
                         KNeighborsRegressor(n_neighbors=10, weights="distance"))


def linear():
    return make_pipeline(StandardScaler(), LinearRegression())


lc_dummies = pd.get_dummies(df["LC0_Desc"], prefix="lc").astype(float)
df = pd.concat([df, lc_dummies], axis=1)
LC_COLS = list(lc_dummies.columns)
POS = ["TH_LAT", "TH_LONG"]

runs = [
    # (name, model, features, target)
    ("Position, linear", linear, POS, "logOC"),
    ("Position, nearest neighbours", knn, POS, "logOC"),
    ("Land cover class", soc.ridge, LC_COLS, "logOC"),
    ("NDVI + BSI", soc.ridge, ["NDVI", "BSI"], "logOC"),
    ("Spectral indices", soc.ridge, SPEC, "logOC"),
    ("Embeddings", soc.ridge, EMB, "logOC"),
    ("Embeddings + spectral", soc.ridge, EMB + SPEC, "logOC"),
    ("SoilGrids properties", soc.ridge, SG, "logOC"),
    ("SoilGrids + embeddings", soc.ridge, SG + EMB, "logOC"),
    ("NDVI + BSI", soc.ridge, ["NDVI", "BSI"], "logN"),
    ("Spectral indices", soc.ridge, SPEC, "logN"),
    ("Embeddings", soc.ridge, EMB, "logN"),
    ("NDVI + BSI", soc.ridge, ["NDVI", "BSI"], "pH_H2O"),
    ("Spectral indices", soc.ridge, SPEC, "pH_H2O"),
    ("Embeddings", soc.ridge, EMB, "pH_H2O"),
    ("NDVI + BSI", soc.ridge, ["NDVI", "BSI"], "logSG_SOC"),
    ("Spectral indices", soc.ridge, SPEC, "logSG_SOC"),
    ("Embeddings", soc.ridge, EMB, "logSG_SOC"),
    ("LightGBM spectral indices", soc.lgbm, SPEC, "logOC"),
    ("LightGBM embeddings", soc.lgbm, EMB, "logOC"),
    ("LightGBM all 86 predictors", soc.lgbm, EMB + SPEC + SG, "logOC"),
]

rows, fold_rows = [], []
# no POINTID here: the file is published, and the rows follow the order of
# results/points.csv.gz, so the figures can still align the two without an id
preds = pd.DataFrame(index=df.index)
for name, model, feats, target in runs:
    # N and pH have a few missing values: those rows are left out for that target
    keep = df[target].notna().to_numpy()
    d = df[keep].reset_index(drop=True)
    for scheme in splits:
        sp = splits[scheme] if keep.all() else soc.folds(d, scheme)
        pred_d, fr = soc.cross_validate(d, feats, target, sp, model)
        pred = np.full(len(df), np.nan)
        pred[keep] = pred_d
        row = {"model": name, "target": target, "scheme": scheme,
               "n_features": len(feats), "n": len(d),
               "r2_fold_mean": fr.mean(), "r2_fold_sd": fr.std(ddof=1),
               "r2_fold_min": fr.min(), "r2_fold_max": fr.max(),
               "r2_pooled": r2_score(d[target], pred_d)}
        if target == "logOC":
            row.update({f"{k}_gkg": v for k, v in soc.errors_gkg(df["OC"], pred).items()})
        rows.append(row)
        fold_rows += [{"model": name, "target": target, "scheme": scheme,
                       "fold": i + 1, "r2": r} for i, r in enumerate(fr)]
        preds[f"{name}|{target}|{scheme}"] = pred
        print(f"{name:30s} {target:10s} {scheme:7s} R2 = {fr.mean():.3f}")

scores = pd.DataFrame(rows)
scores.to_csv(OUT / "model_scores.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "fold_scores.csv", index=False)
preds.to_csv(OUT / "oof_predictions.csv.gz", index=False)


# 3. uncertainty of the paired differences
# Resample whole NUTS2 regions with replacement and recompute the R2 difference.
def region_bootstrap(y, pred_a, pred_b, regions, n_boot=1000, seed=cfg.SEED):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(regions, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(codes))]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(codes), len(codes))
        i = np.concatenate([members[k] for k in pick])
        diffs[b] = r2_score(y[i], pred_a[i]) - r2_score(y[i], pred_b[i])
    return np.percentile(diffs, [2.5, 97.5])


regions = df["NUTS_2"].astype(str).to_numpy()
y = df["logOC"].to_numpy()
fold_table = pd.DataFrame(fold_rows)
unc = []
for scheme in splits:
    a = preds[f"Embeddings|logOC|{scheme}"].to_numpy()
    b = preds[f"Spectral indices|logOC|{scheme}"].to_numpy()
    lo, hi = region_bootstrap(y, a, b, regions)
    same = (fold_table.target == "logOC") & (fold_table.scheme == scheme)
    folds_a = fold_table[same & (fold_table.model == "Embeddings")]["r2"].to_numpy()
    folds_b = fold_table[same & (fold_table.model == "Spectral indices")]["r2"].to_numpy()
    unc.append({"comparison": "Embeddings minus spectral, measured OC", "scheme": scheme,
                "diff_fold_mean": (folds_a - folds_b).mean(),
                "diff_fold_min": (folds_a - folds_b).min(),
                "diff_fold_max": (folds_a - folds_b).max(),
                "diff_pooled": r2_score(y, a) - r2_score(y, b),
                "ci95_low": lo, "ci95_high": hi})
print(pd.DataFrame(unc))
pd.DataFrame(unc).to_csv(OUT / "difference_uncertainty.csv", index=False)


# 4. SoilGrids against LUCAS
sg, lu = df["SG_SOC_gkg"].to_numpy(), df["OC"].to_numpy()
direct = {
    "r2_log": r2_score(np.log1p(lu), np.log1p(sg)),
    "r_squared_log": np.corrcoef(np.log1p(lu), np.log1p(sg))[0, 1] ** 2,
    "r2_gkg": r2_score(lu, sg),
    "rmse_gkg": float(np.sqrt(np.mean((sg - lu) ** 2))),
    "mae_gkg": float(np.mean(np.abs(sg - lu))),
    "medae_gkg": float(np.median(np.abs(sg - lu))),
    "bias_gkg": float(np.mean(sg - lu)),
    "spearman": float(pd.Series(sg).corr(pd.Series(lu), method="spearman")),
}
with open(OUT / "soilgrids_direct.json", "w") as f:
    json.dump(direct, f, indent=2)
print("SoilGrids vs LUCAS:", {k: round(v, 3) for k, v in direct.items()})


# 5. errors of the model by land cover
lc_rows = []
for lc in cfg.LAND_COVER + ["Artificial land", "Wetlands", "Water"]:
    m = (df["LC0_Desc"] == lc).to_numpy()
    for scheme in splits:
        for label, key, target in [
                ("Embeddings", "Embeddings|logOC", "logOC"),
                ("Spectral indices", "Spectral indices|logOC", "logOC"),
                ("Embeddings to SoilGrids", "Embeddings|logSG_SOC", "logSG_SOC")]:
            p = preds[f"{key}|{scheme}"].to_numpy()[m]
            obs = df.loc[m, target].to_numpy()
            r = {"land_cover": lc, "scheme": scheme, "model": label, "n": int(m.sum()),
                 "r2": r2_score(obs, p) if m.sum() > 2 else np.nan,
                 "sd_log_target": obs.std(ddof=1) if m.sum() > 2 else np.nan}
            if target == "logOC":
                r.update({f"{k}_gkg": v for k, v in soc.errors_gkg(df.loc[m, "OC"], p).items()})
                r["median_oc_gkg"] = df.loc[m, "OC"].median()
            lc_rows.append(r)
pd.DataFrame(lc_rows).to_csv(OUT / "land_cover_errors.csv", index=False)


# 6. Cropland and low-vegetation subsets (models refitted in each subset)
subsets = {
    "Cropland": df["LC0_Desc"] == "Cropland",
    "Cropland and bareland": df["LC0_Desc"].isin(["Cropland", "Bareland"]),
    "Low vegetation (annual NDVI < 0.4)": df["NDVI"] < 0.4,
}
sub_rows = []
for sname, mask in subsets.items():
    d = df[mask].reset_index(drop=True)
    for scheme in ["random", "nuts2"]:
        sp = soc.folds(d, scheme)
        for label, feats in [("Spectral indices", SPEC), ("Embeddings", EMB)]:
            pred, fr = soc.cross_validate(d, feats, "logOC", sp)
            sub_rows.append({"subset": sname, "n": len(d), "median_oc_gkg": d["OC"].median(),
                             "scheme": scheme, "model": label, "r2_fold_mean": fr.mean(),
                             "r2_fold_sd": fr.std(ddof=1),
                             **{f"{k}_gkg": v for k, v in soc.errors_gkg(d["OC"], pred).items()}})
subset_scores = pd.DataFrame(sub_rows)
subset_scores.to_csv(OUT / "subset_scores.csv", index=False)
print(subset_scores[["subset", "n", "scheme", "model", "r2_fold_mean"]])


# 7. leave one country out
counts = df["NUTS_0"].value_counts()
d = df[df["NUTS_0"].isin(counts[counts >= cfg.LOCO_MIN_TRAIN].index)].reset_index(drop=True)
logo = list(LeaveOneGroupOut().split(d, groups=d["NUTS_0"]))
d["pred_emb"], _ = soc.cross_validate(d, EMB, "logOC", logo)
d["pred_spec"], _ = soc.cross_validate(d, SPEC, "logOC", logo)
loco = []
for code, s in d.groupby("NUTS_0"):
    if len(s) < cfg.LOCO_MIN_REPORT:
        continue
    loco.append({"code": code, "country": cfg.COUNTRY_NAMES[code], "n": len(s),
                 "median_oc_gkg": s["OC"].median(),
                 "r2_embeddings": r2_score(s["logOC"], s["pred_emb"]),
                 "r2_spectral": r2_score(s["logOC"], s["pred_spec"]),
                 "bias_log_embeddings": (s["pred_emb"] - s["logOC"]).mean(),
                 "r_embeddings": np.corrcoef(s["logOC"], s["pred_emb"])[0, 1],
                 **{f"{k}_gkg": v for k, v in soc.errors_gkg(s["OC"], s["pred_emb"]).items()}})
loco = pd.DataFrame(loco).sort_values("r2_embeddings", ascending=False)
loco.to_csv(OUT / "country_transfer.csv", index=False)
rep = d[d["NUTS_0"].isin(loco["code"])]
loco_summary = {
    "n_fit_points": len(d), "n_fit_countries": int(d["NUTS_0"].nunique()),
    "n_reported_countries": len(loco), "n_reported_points": len(rep),
    "mean_r2_embeddings": loco["r2_embeddings"].mean(),
    "mean_r2_spectral": loco["r2_spectral"].mean(),
    "median_r2_embeddings": loco["r2_embeddings"].median(),
    "positive_embeddings": int((loco["r2_embeddings"] > 0).sum()),
    "negative_spectral": int((loco["r2_spectral"] < 0).sum()),
    "embeddings_better": int((loco["r2_embeddings"] > loco["r2_spectral"]).sum()),
    "pooled_r2_embeddings": r2_score(rep["logOC"], rep["pred_emb"]),
    "pooled_r2_spectral": r2_score(rep["logOC"], rep["pred_spec"]),
}
with open(OUT / "country_transfer_summary.json", "w") as f:
    json.dump(loco_summary, f, indent=2)
print(loco_summary)


# 8. learning curve
# Train on a growing share of points (sampled within each country), test on
# all remaining points. Repeated five times with different seeds.
lc_curve = []
for frac in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80]:
    for rep_i in range(5):
        train = df.groupby("NUTS_0", group_keys=False).sample(frac=frac, random_state=rep_i).index
        test = df.index.difference(train)
        for label, feats in [("Spectral indices", SPEC), ("Embeddings", EMB)]:
            m = soc.ridge().fit(df.loc[train, feats], df.loc[train, "logOC"])
            lc_curve.append({"fraction": frac, "repeat": rep_i, "n_train": len(train),
                             "model": label,
                             "r2": r2_score(df.loc[test, "logOC"], m.predict(df.loc[test, feats]))})
pd.DataFrame(lc_curve).to_csv(OUT / "learning_curve.csv", index=False)


# 9. Trends.Earth SDG 15.3.1 outputs
STATUS, LPD, LCD, SOCC = ("STATUS_2023", "LPD_2008_2023",
                          "LC_DEGRADATION_2015_2023", "SOC_PCT_CHANGE_2015_2023")
t = df.dropna(subset=[STATUS, LPD, LCD, SOCC]).copy()
flag_lpd = t[LPD].isin([1, 2])            # declining or moderate decline
flag_lc = t[LCD] == -1                    # land cover degradation
flag_soc = t[SOCC] <= -10                 # carbon loss of 10 % or more
derived = flag_lpd | flag_lc | flag_soc   # one out, all out
published = t[STATUS].isin([1, 2])        # degraded in the reporting period
n_flags = flag_lpd.astype(int) + flag_lc.astype(int) + flag_soc.astype(int)
sdg = {
    "n_points": len(t),
    "n_missing": int(len(df) - len(t)),
    "soc_change_zero": int((t[SOCC] == 0).sum()),
    "soc_change_zero_pct": 100 * (t[SOCC] == 0).mean(),
    "soc_change_min": t[SOCC].min(), "soc_change_max": t[SOCC].max(),
    "degraded_reporting": int(published.sum()),
    "degraded_any_incl_baseline": int(t[STATUS].isin([1, 2, 3]).sum()),
    "reconstruction_identical": bool((derived == published).all()),
    "flag_productivity": int((flag_lpd & published).sum()),
    "flag_land_cover": int((flag_lc & published).sum()),
    "flag_soil_carbon": int((flag_soc & published).sum()),
    "soil_carbon_only": int((flag_soc & ~flag_lpd & ~flag_lc).sum()),
    "productivity_only": int((flag_lpd & ~flag_soc & ~flag_lc).sum()),
    "two_or_more_flags": int((n_flags >= 2).sum()),
}
status_table = t.groupby(STATUS).agg(n=("OC", "size"), median_oc=("OC", "median"),
                                     mean_oc=("OC", "mean")).reset_index()
status_table.to_csv(OUT / "sdg_status_table.csv", index=False)
sdg["median_oc_degraded"] = float(t.loc[published, "OC"].median())
sdg["median_oc_not_degraded"] = float(t.loc[~published, "OC"].median())
sdg["median_oc_stable"] = float(t.loc[t[STATUS] == 4, "OC"].median())
with open(OUT / "sdg_attribution.json", "w") as f:
    json.dump(sdg, f, indent=2)
print(sdg)

# measured OC and land cover of each status class
detail = t.groupby(STATUS)["OC"].quantile([0.25, 0.5, 0.75]).unstack()
detail.columns = ["q25_oc", "median_oc", "q75_oc"]
detail.insert(0, "n", t.groupby(STATUS).size())
cover = pd.crosstab(t[STATUS], t["LC0_Desc"], normalize="index")[cfg.LAND_COVER] * 100
detail.join(cover.add_suffix("_pct")).reset_index() \
      .to_csv(OUT / "sdg_status_detail.csv", index=False)


# degraded (classes 1-2) against all other sites: difference of the median OC,
# with an interval from resampling whole NUTS2 regions, as in section 3
def median_difference_ci(values, flag, groups, n_boot=1000, seed=cfg.SEED):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(codes))]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([members[k] for k in rng.integers(0, len(codes), len(codes))])
        diffs[b] = np.median(values[i][flag[i]]) - np.median(values[i][~flag[i]])
    return np.percentile(diffs, [2.5, 97.5])


deg = published.to_numpy()
oc_t = t["OC"].to_numpy()
lo, hi = median_difference_ci(oc_t, deg, t["NUTS_2"].astype(str).to_numpy())
contrast = {
    "median_oc_degraded": float(np.median(oc_t[deg])),
    "median_oc_other": float(np.median(oc_t[~deg])),
    "difference": float(np.median(oc_t[deg]) - np.median(oc_t[~deg])),
    "difference_ci95_regions": [float(lo), float(hi)],
    "by_land_cover": {},
}
for lc in cfg.LAND_COVER:
    s = t[t["LC0_Desc"] == lc]
    d = s[STATUS].isin([1, 2])
    contrast["by_land_cover"][lc] = {"n_degraded": int(d.sum()), "n_other": int((~d).sum()),
                                     "median_degraded": float(s.loc[d, "OC"].median()),
                                     "median_other": float(s.loc[~d, "OC"].median())}
by_country = []
for name, s in t.groupby("country"):
    d = s[STATUS].isin([1, 2])
    by_country.append({"country": name, "n": len(s), "degraded_pct": 100 * d.mean(),
                       "median_oc": s["OC"].median(), "n_degraded": int(d.sum()),
                       "median_oc_degraded": s.loc[d, "OC"].median(),
                       "median_oc_other": s.loc[~d, "OC"].median()})
by_country = pd.DataFrame(by_country)
by_country.to_csv(OUT / "sdg_by_country.csv", index=False)
both = by_country[(by_country.n_degraded >= 20) & (by_country.n - by_country.n_degraded >= 20)]
contrast["countries_compared"] = len(both)
contrast["countries_higher_when_degraded"] = int((both.median_oc_degraded > both.median_oc_other).sum())
with open(OUT / "sdg_soc_contrast.json", "w") as f:
    json.dump(contrast, f, indent=2)
print(contrast)


# 10. what the embedding dimensions relate to
env = {"logOC": df["logOC"], "Latitude": df["TH_LAT"], "Elevation": df["Elev"],
       "NDVI": df["NDVI"], "NDMI": df["NDMI"], "BSI": df["BSI"],
       "SG clay": df["SG_CLAY"], "SG pH": df["SG_PH"], "SG bulk density": df["SG_BDOD"]}
env = pd.DataFrame(env)
corr = pd.DataFrame({dim: env.corrwith(df[dim]) for dim in EMB}).T
corr.to_csv(OUT / "embedding_correlations.csv")
single = [{"dim": dim, "r2_fold_mean": soc.cross_validate(df, [dim], "logOC", splits["random"])[1].mean()}
          for dim in EMB]
pd.DataFrame(single).to_csv(OUT / "single_dimension_r2.csv", index=False)


# 11. save summary
summary["loco"] = loco_summary
summary["sdg"] = sdg
summary["soilgrids_direct"] = direct
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
df[["POINTID", "TH_LAT", "TH_LONG", "NUTS_0", "NUTS_2", "LC0_Desc", "OC",
    "SG_SOC_gkg", STATUS]].to_csv(OUT / "points.csv.gz", index=False)

# package versions of this run
manifest = {"date": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "sites": len(df), "seed": cfg.SEED, "folds": cfg.N_FOLDS}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("done")
