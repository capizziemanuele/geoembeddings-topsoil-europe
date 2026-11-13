import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
import soc

OUT = cfg.RESULTS_DIR
df = soc.load_data()
splits = {"random": soc.folds(df, "random"), "nuts2": soc.folds(df, "nuts2")}
GRID = {"ridge__alpha": cfg.ALPHAS}


def nested(groups=None):
    """Ridge with scaling refitted inside every inner fold."""
    def make():
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        cv = GroupKFold(5) if groups is not None else 5
        return GridSearchCV(pipe, GRID, cv=cv, scoring="r2")
    return make


def run(name, features_name, features, target, scheme, model, groups=None):
    X, y = df[features].to_numpy(), df[target].to_numpy()
    r2, alphas = [], []
    for train, test in splits[scheme]:
        m = model()
        if groups is None:
            m.fit(X[train], y[train])
        else:
            m.fit(X[train], y[train], groups=groups[train])
        r2.append(m.score(X[test], y[test]))
        best = getattr(m, "best_params_", None)
        alphas.append(best["ridge__alpha"] if best else m[-1].alpha_)
    return {"model": name, "features": features_name, "scheme": scheme,
            "r2_fold_mean": float(np.mean(r2)), "r2_fold_sd": float(np.std(r2, ddof=1)),
            "median_alpha": float(np.median(alphas))}


rows = []
region = df["NUTS_2"].astype(str).to_numpy()
for features_name, features in [("Embeddings", cfg.EMBEDDINGS), ("Spectral indices", cfg.SPECTRAL)]:
    for scheme in ("random", "nuts2"):
        rows.append(run("Scaling on the outer training set (used in the paper)",
                        features_name, features, "logOC", scheme, soc.ridge))
        rows.append(run("Nested tuning, scaling refitted in each inner fold",
                        features_name, features, "logOC", scheme, nested()))
        rows.append(run("Nested tuning with NUTS2 regions kept together inside",
                        features_name, features, "logOC", scheme, nested(groups=region), groups=region))

sens = pd.DataFrame(rows)
sens.to_csv(OUT / "sensitivity.csv", index=False)
print(sens.to_string(index=False))

# soil texture and carbon range per country
country = df.groupby("country").agg(
    n=("OC", "size"), median_oc=("OC", "median"),
    q25_oc=("OC", lambda s: s.quantile(0.25)), q75_oc=("OC", lambda s: s.quantile(0.75)),
    sand=("SG_SAND", "mean"), clay=("SG_CLAY", "mean"), ph=("pH_H2O", "mean"))
country["iqr_oc"] = country["q75_oc"] - country["q25_oc"]
country = country[country["n"] >= cfg.LOCO_MIN_REPORT].sort_values("sand", ascending=False)
country.round(1).to_csv(OUT / "country_soil.csv")
print(country.round(1).head(8).to_string())
