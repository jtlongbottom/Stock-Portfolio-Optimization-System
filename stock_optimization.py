import os
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "stock_data.csv"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

try:
    df = pl.read_csv(str(INPUT), infer_schema_length=5000, try_parse_dates=True)
except Exception:
    df = pl.read_csv(str(INPUT), infer_schema_length=5000)

date_col = next((c for c in df.columns if c.lower() in ("date","datetime","timestamp")), df.columns[0])
dt = df[date_col].dtype
if dt == pl.Datetime:
    pass
elif dt == pl.Date:
    df = df.with_columns(pl.col(date_col).cast(pl.Datetime))
elif dt == pl.Utf8:
    df = df.with_columns(pl.col(date_col).str.strptime(pl.Datetime, strict=False, exact=False))
else:
    df = df.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))

df = df.drop_nulls([date_col]).sort(date_col)

num_cols = [c for c in df.columns if c != date_col and getattr(df[c].dtype, "is_numeric", lambda: False)()]

filled = df.select(pl.col(date_col), *[pl.col(c).forward_fill().backward_fill().alias(c) for c in num_cols])
rets = filled.select(pl.col(date_col), *[(pl.col(c).pct_change()).alias(c) for c in num_cols])

stats = []
for c in num_cols:
    r = rets.select(pl.col(c)).to_numpy().ravel()
    r = r[~np.isnan(r)]
    if r.size == 0:
        continue
    mu = r.mean()
    sd = r.std(ddof=0)
    ar = mu * 252
    av = sd * (252**0.5)
    sh = (ar / av) if av > 0 else np.nan
    stats.append([c, int(r.size), ar, av, sh, np.min(r) if r.size else np.nan, np.max(r) if r.size else np.nan])
kpis = pl.DataFrame(stats, schema=["asset","n_returns","annualized_return","annualized_vol","sharpe_ratio_approx","min_daily_ret","max_daily_ret"])
kpis.write_csv(OUT / "stock_kpis.csv")

MAX_CORR_ASSETS = 100
if len(num_cols) > MAX_CORR_ASSETS:
    variances = []
    for c in num_cols:
        r = rets.select(pl.col(c)).to_numpy().ravel()
        r = r[~np.isnan(r)]
        if r.size > 1:
            variances.append((c, float(np.var(r))))
    variances.sort(key=lambda t: t[1], reverse=True)
    top = [a for a,_ in variances[:MAX_CORR_ASSETS]]
else:
    top = num_cols

if len(top) >= 2:
    mats = []
    for c in top:
        arr = rets.select(pl.col(c)).to_numpy().ravel()
        mats.append(arr)
    R = np.column_stack(mats)
    R[np.isnan(R)] = 0.0
    C = np.corrcoef(R, rowvar=False)
    pd.DataFrame(C, index=top, columns=top).to_csv(OUT / "corr_matrix_subset.csv")

filled.write_parquet(OUT / "stock_data_cleaned.parquet")
rets.write_parquet(OUT / "stock_returns.parquet")
filled.head(1000).write_csv(OUT / "stock_data_cleaned_head.csv")
rets.head(1000).write_csv(OUT / "stock_returns_head.csv")

plot_series = top[:10] if len(top) else num_cols[:10]
if plot_series:
    pdf = filled.select(pl.col(date_col), *[pl.col(c) for c in plot_series]).to_pandas()
    fig = plt.figure()
    for c in plot_series:
        s = pdf[c].dropna()
        base = s.iloc[0] if len(s) else np.nan
        y = pdf[c] / base * 100.0 if pd.notna(base) and base != 0 else np.nan
        plt.plot(pdf[date_col], y, label=c)
    plt.title("Normalized Price Index (base=100)")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.legend(loc="best", ncol=2, fontsize=8)
    fig.savefig(OUT / "normalized_prices.png", bbox_inches="tight")
    plt.close(fig)

    rcat = rets.select(*[pl.col(c) for c in plot_series]).to_pandas().values.ravel()
    rcat = rcat[~np.isnan(rcat)]
    if rcat.size:
        fig = plt.figure()
        plt.hist(rcat, bins=50)
        plt.title("Distribution of Daily Returns (Sampled Assets)")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        fig.savefig(OUT / "returns_hist.png", bbox_inches="tight")
        plt.close(fig)

print(f"EDA saved to: {OUT}")

#%%
import polars as pl
from pathlib import Path

OUT = Path("outputs/eda")


df = pl.read_parquet(OUT / "stock_data_cleaned.parquet")
rets = pl.read_parquet(OUT / "stock_returns.parquet")
kpis = pl.read_csv(OUT / "stock_kpis.csv")

print("=== Cleaned dataset ===")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
print("")

print("=== Returns dataset ===")
print(f"Rows: {rets.shape[0]:,} | Columns: {rets.shape[1]:,}")
print("")

print("=== KPI summary (top 10 Sharpe ratios) ===")
display_df = (
    kpis.sort("sharpe_ratio_approx", descending=True)
        .head(10)
)
print(display_df)


display_df_pandas = display_df.to_pandas()
display_df_pandas
#%%
import polars as pl

df = pl.read_parquet("outputs/eda/stock_data_cleaned.parquet")
print(df.head(5))


print(df.schema)

print(df.select(df.columns[:10]).head(5))
#%%
import polars as pl
import numpy as np

OUT = pl.Config.set_tbl_rows(5)
ROOT = "outputs/eda"

# reload the cleaned prices
df = pl.read_parquet(f"{ROOT}/stock_data_cleaned.parquet")
date_col = df.columns[0]
num_cols = df.columns[1:]


varying = []
for c in num_cols:
    s = df[c].drop_nulls().to_numpy()
    if s.size > 1 and not np.allclose(s, s[0]):
        varying.append(c)
print(f"Varying columns: {len(varying)} of {len(num_cols)}")


rets = df.select(
    pl.col(date_col),
    *[(pl.col(c).pct_change()).alias(c) for c in varying]
)
rets.write_parquet(f"{ROOT}/stock_returns_varonly.parquet")
rets.head(5).write_csv(f"{ROOT}/stock_returns_varonly_head.csv")
print("[ok] Saved filtered returns file with non-constant assets.")

#%%
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("outputs/eda")
rets = pl.read_parquet(ROOT / "stock_returns_varonly.parquet")

asset_cols = rets.columns[1:]
R = rets.select(asset_cols).to_numpy()

n  = np.sum(~np.isnan(R), axis=0)
mu = np.nanmean(R, axis=0)
sd = np.nanstd(R, axis=0, ddof=0)
ar = mu * 252.0
av = sd * np.sqrt(252.0)
sh = np.divide(ar, av, out=np.full_like(ar, np.nan), where=av > 0)
mn = np.nanmin(R, axis=0)
mx = np.nanmax(R, axis=0)

kpis = pd.DataFrame({
    "asset": asset_cols,
    "n_returns": n.astype(int),
    "annualized_return": ar,
    "annualized_vol": av,
    "sharpe_ratio_approx": sh,
    "min_daily_ret": mn,
    "max_daily_ret": mx,
})

out = ROOT / "stock_kpis_varonly.csv"
kpis.to_csv(out, index=False)
print(f"[ok] KPI table saved → {out} | assets: {len(kpis)}")


MIN_OBS = 100
kf = kpis[(kpis["n_returns"] >= MIN_OBS) & (kpis["annualized_vol"] > 0)]
kf.sort_values("sharpe_ratio_approx", ascending=False).head(50).to_csv(ROOT / "summary_top50_sharpe.csv", index=False)
kf.sort_values("sharpe_ratio_approx", ascending=True ).head(50).to_csv(ROOT / "summary_bottom50_sharpe.csv", index=False)
kf.sort_values("annualized_vol",      ascending=False).head(50).to_csv(ROOT / "summary_top50_vol.csv", index=False)
kf.sort_values("annualized_vol",      ascending=True ).head(50).to_csv(ROOT / "summary_bottom50_vol.csv", index=False)
kf.sort_values("sharpe_ratio_approx", ascending=False).head(1000)[
    ["asset","n_returns","annualized_return","annualized_vol","sharpe_ratio_approx"]
].to_csv(ROOT / "universe_top1000_by_sharpe.csv", index=False)

print("[ok] Summaries written.")


#%%


import math, json, warnings
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
EDA  = ROOT / "outputs" / "eda"
OUT  = ROOT / "outputs" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

RET_PARQ = EDA / "stock_returns_varonly.parquet"
UNI_CSV  = EDA / "universe_top1000_by_sharpe.csv"

if not RET_PARQ.exists():
    raise FileNotFoundError(f"Missing {RET_PARQ}.")
if not UNI_CSV.exists():
    raise FileNotFoundError(f"Missing {UNI_CSV}. Run the EDA summary cell to create it.")

rets = pl.read_parquet(RET_PARQ)
date_col = rets.columns[0]
universe = set(pl.read_csv(UNI_CSV)["asset"].to_list())
asset_cols = [c for c in rets.columns[1:] if c in universe]

# features
def build_features_for(asset: str, df_ret: pl.DataFrame) -> pd.DataFrame:
    d = df_ret.select([date_col, asset]).rename({asset: "ret"}).to_pandas()
    d.sort_values(date_col, inplace=True)
    for k in (1,2,3,5,10,20):
        d[f"lag_{k}"] = d["ret"].shift(k)
    d["roll_mean_5"]  = d["ret"].rolling(5).mean()
    d["roll_mean_20"] = d["ret"].rolling(20).mean()
    d["roll_std_10"]  = d["ret"].rolling(10).std(ddof=0)
    d["roll_std_20"]  = d["ret"].rolling(20).std(ddof=0)
    d["roll_mom_10"]  = d["ret"].rolling(10).sum()
    d["roll_mom_20"]  = d["ret"].rolling(20).sum()
    d["y_next"] = d["ret"].shift(-1)
    d.dropna(inplace=True)
    return d

N_SPLITS = 5
SEED = 42
MIN_ROWS = 150

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
all_metrics = {}
pred_rows = []

for i, asset in enumerate(asset_cols, 1):
    dfA = build_features_for(asset, rets)
    if len(dfA) < MIN_ROWS:
        continue
    feats = [c for c in dfA.columns if c not in (date_col, "ret", "y_next")]
    X = dfA[feats].values
    y = dfA["y_next"].values
    dates = dfA[date_col].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",  ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=3, random_state=SEED, max_iter=20000))
    ])

    oof = np.full_like(y, np.nan, dtype=float)
    fold_metrics = []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        model.fit(X[tr], y[tr])
        p = model.predict(X[va])
        oof[va] = p
        mae = mean_absolute_error(y[va], p)
        rmse = math.sqrt(mean_squared_error(y[va], p))
        r2   = r2_score(y[va], p)
        diracc = np.mean((y[va] >= 0) == (p >= 0))
        fold_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2, "DirAcc": float(diracc)})

    mask = ~np.isnan(oof)
    mae_oof  = mean_absolute_error(y[mask], oof[mask])
    rmse_oof = math.sqrt(mean_squared_error(y[mask], oof[mask]))
    r2_oof   = r2_score(y[mask], oof[mask])
    diracc_o = float(np.mean((y[mask] >= 0) == (oof[mask] >= 0)))

    # refit on all data
    model.fit(X, y)
    latest_pred = float(model.predict(X[-1:].copy())[0])

    df_pred = pd.DataFrame({"date": dates, "asset": asset, "y_true": y, "y_pred": oof})
    pred_rows.append(df_pred)

    all_metrics[asset] = {
        "n_rows": int(len(dfA)),
        "n_features": int(len(feats)),
        "oof": {"MAE": mae_oof, "RMSE": rmse_oof, "R2": r2_oof, "DirAcc": diracc_o},
        "folds": fold_metrics,
        "latest_date": str(pd.to_datetime(dates[-1]).date()),
        "latest_pred_next_return": latest_pred
    }

    if i % 100 == 0:
        print(f"[{i}/{len(asset_cols)}] {asset}: OOF RMSE={rmse_oof:.5f}, DirAcc={diracc_o:.3f}")

pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)
pred_df.to_csv(OUT / "predictions_oof.csv", index=False)
with open(OUT / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\nSaved:")
print(" -", OUT / "predictions_oof.csv")
print(" -", OUT / "metrics.json")
#%%
import math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

import os
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(os.getcwd())

EDA  = ROOT / "outputs" / "eda"
OUT  = ROOT / "outputs" / "ml_multimodel"
OUT.mkdir(parents=True, exist_ok=True)

rets = pd.read_parquet(EDA / "stock_returns_varonly.parquet")
universe = pd.read_csv(EDA / "universe_top1000_by_sharpe.csv")["asset"].tolist()
assets = [c for c in rets.columns if c in universe]
date_col = rets.columns[0]
rets = rets[[date_col] + assets]
rets[date_col] = pd.to_datetime(rets[date_col])

def build_features(df, col):
    d = df[[date_col, col]].rename(columns={col:"ret"}).copy()
    d = d.sort_values(date_col)
    for k in (1,2,3,5,10,20):
        d[f"lag_{k}"] = d["ret"].shift(k)
    d["roll_mean_5"]  = d["ret"].rolling(5).mean()
    d["roll_mean_20"] = d["ret"].rolling(20).mean()
    d["roll_std_10"]  = d["ret"].rolling(10).std(ddof=0)
    d["roll_std_20"]  = d["ret"].rolling(20).std(ddof=0)
    d["roll_mom_10"]  = d["ret"].rolling(10).sum()
    d["roll_mom_20"]  = d["ret"].rolling(20).sum()
    d["y_next"] = d["ret"].shift(-1)
    return d.dropna()

tscv = TimeSeriesSplit(n_splits=5)
SEED = 42
MIN_ROWS = 150

models = {
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=3, random_state=SEED, max_iter=20000))
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, objective="reg:squarederror"
    ),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=1.0, epsilon=0.001))
    ])
}

all_metrics = {m: {} for m in models}
pred_records = []

for i, asset in enumerate(assets, 1):
    dfA = build_features(rets, asset)
    if len(dfA) < MIN_ROWS:
        continue
    feats = [c for c in dfA.columns if c not in (date_col,"ret","y_next")]
    X, y = dfA[feats].values, dfA["y_next"].values
    dates = dfA[date_col].values

    for mname, model in models.items():
        oof = np.full_like(y, np.nan, dtype=float)
        fold_metrics = []
        for fold, (tr, va) in enumerate(tscv.split(X), 1):
            model.fit(X[tr], y[tr])
            p = model.predict(X[va])
            oof[va] = p
            mae = mean_absolute_error(y[va], p)
            rmse = math.sqrt(mean_squared_error(y[va], p))
            r2   = r2_score(y[va], p)
            diracc = np.mean((y[va]>=0)==(p>=0))
            fold_metrics.append({"fold":fold,"MAE":mae,"RMSE":rmse,"R2":r2,"DirAcc":float(diracc)})

        mask = ~np.isnan(oof)
        if not np.any(mask):
            continue
        mae_oof  = mean_absolute_error(y[mask], oof[mask])
        rmse_oof = math.sqrt(mean_squared_error(y[mask], oof[mask]))
        r2_oof   = r2_score(y[mask], oof[mask])
        diracc_o = float(np.mean((y[mask]>=0)==(oof[mask]>=0)))

        model.fit(X, y)
        latest_pred = float(model.predict(X[-1:].copy())[0])

        all_metrics[mname][asset] = {
            "n_rows": int(len(dfA)),
            "n_features": int(len(feats)),
            "oof": {"MAE":mae_oof,"RMSE":rmse_oof,"R2":r2_oof,"DirAcc":diracc_o},
            "folds": fold_metrics,
            "latest_date": str(pd.to_datetime(dates[-1]).date()),
            "latest_pred_next_return": latest_pred
        }

        for d, yt, yp in zip(dates, y, oof):
            pred_records.append([d, asset, mname, yt, yp])

    if i % 50 == 0:
        print(f"[{i}/{len(assets)}] processed")

pred_df = pd.DataFrame(pred_records, columns=["date","asset","model","y_true","y_pred"])
pred_df.to_csv(OUT / "predictions_all_models.csv", index=False)

for mname, mdict in all_metrics.items():
    with open(OUT / f"metrics_{mname}.json", "w") as f:
        json.dump(mdict, f, indent=2)

print("\nSaved outputs to:", OUT)
#%%
# summarize_ml_results.py

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(os.getcwd())

OUT_MM = ROOT / "outputs" / "ml_multimodel"
OUT_SUM = ROOT / "outputs" / "ml_summary"
OUT_SUM.mkdir(parents=True, exist_ok=True)


rows = []
for p in OUT_MM.glob("metrics_*.json"):
    model = p.stem.replace("metrics_", "")
    with open(p, "r") as f:
        d = json.load(f)
    for asset, v in d.items():
        o = v.get("oof", {})
        rows.append({
            "model": model,
            "asset": asset,
            "n_rows": v.get("n_rows"),
            "n_features": v.get("n_features"),
            "MAE": o.get("MAE"),
            "RMSE": o.get("RMSE"),
            "R2": o.get("R2"),
            "DirAcc": o.get("DirAcc"),
            "latest_date": v.get("latest_date"),
            "latest_pred_next_return": v.get("latest_pred_next_return"),
        })

if not rows:
    raise SystemExit(f"No metrics_*.json files found in {OUT_MM}")

per_asset = pd.DataFrame(rows)
per_asset.to_csv(OUT_SUM / "per_asset_metrics_all_models.csv", index=False)

#Model-level summary tables
def _agg_tbl(df):
    return pd.DataFrame({
        "Mean": df.mean(numeric_only=True),
        "Median": df.median(numeric_only=True),
        "Std": df.std(numeric_only=True)
    })

summary = (per_asset
           .groupby("model")[["MAE","RMSE","R2","DirAcc"]]
           .mean()
           .reset_index())
summary.to_csv(OUT_SUM / "model_comparison_summary.csv", index=False)

summary_full = per_asset.groupby("model").apply(
    lambda g: _agg_tbl(g[["MAE","RMSE","R2","DirAcc"]])
).reset_index().rename(columns={"level_1":"stat"})
summary_full.to_csv(OUT_SUM / "model_comparison_summary_full.csv", index=False)

print("[ok] Wrote summaries:",
      OUT_SUM / "model_comparison_summary.csv",
      OUT_SUM / "model_comparison_summary_full.csv", sep="\n - ")

#Plots: errors & directional acc
plt.figure()
summary.set_index("model")[["MAE","RMSE"]].plot(kind="bar")
plt.title("Error comparison across models")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig(OUT_SUM / "error_comparison.png")
plt.close()

plt.figure()
summary.set_index("model")[["DirAcc"]].plot(kind="bar")
plt.title("Directional accuracy across models")
plt.ylabel("Accuracy")
plt.ylim(0.45, 0.65)  # adjust if needed
plt.tight_layout()
plt.savefig(OUT_SUM / "diracc_comparison.png")
plt.close()


plt.figure()
per_asset.boxplot(column="DirAcc", by="model", grid=False)
plt.title("Directional accuracy distribution by model")
plt.suptitle("")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(OUT_SUM / "diracc_distribution_boxplot.png")
plt.close()

print("[ok] Wrote plots:",
      OUT_SUM / "error_comparison.png",
      OUT_SUM / "diracc_comparison.png",
      OUT_SUM / "diracc_distribution_boxplot.png", sep="\n - ")

# Top assets tables (by DirAcc and by R2)
topk = 20
tops = []
for m, g in per_asset.groupby("model"):
    top_dir = g.sort_values("DirAcc", ascending=False).head(topk).assign(rank=range(1, topk+1))
    top_r2  = g.sort_values("R2", ascending=False).head(topk).assign(rank=range(1, topk+1))
    top_dir.to_csv(OUT_SUM / f"top{topk}_assets_by_diracc_{m}.csv", index=False)
    top_r2.to_csv(OUT_SUM / f"top{topk}_assets_by_r2_{m}.csv", index=False)
    tops.append((m, len(g), top_dir.iloc[0]["asset"], top_dir.iloc[0]["DirAcc"], top_r2.iloc[0]["asset"], top_r2.iloc[0]["R2"]))

pd.DataFrame(tops, columns=["model","n_assets","best_by_diracc","best_diracc",
                            "best_by_r2","best_r2"]).to_csv(OUT_SUM / "top_assets_overview.csv", index=False)

print("[ok] Wrote top-asset tables to", OUT_SUM)


pred_path = OUT_MM / "predictions_all_models.csv"
if pred_path.exists():
    pred = pd.read_csv(pred_path)
    cov = (pred.assign(valid=~pred["y_pred"].isna())
                .groupby(["model","asset"])["valid"].mean()
                .reset_index(name="coverage_rate"))
    cov.to_csv(OUT_SUM / "coverage_by_model_asset.csv", index=False)
    cov_summary = cov.groupby("model")["coverage_rate"].agg(["mean","median","std"])
    cov_summary.to_csv(OUT_SUM / "coverage_summary_by_model.csv")
    print("[ok] Wrote coverage summaries.")
else:
    print("[info] predictions_all_models.csv not found; skipped coverage analysis.")

print("\n[DONE] ML summaries saved in:", OUT_SUM)
#%%portfolio optimization

import os, sys, json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import objective_functions
except Exception as e:
    EfficientFrontier = None

ROOT = Path(os.getcwd())
EDA_DIR = ROOT / "outputs" / "eda"
MM_DIR  = ROOT / "outputs" / "ml_multimodel"
SUM_DIR = ROOT / "outputs" / "ml_summary"
OUT_DIR = ROOT / "outputs" / "optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RET_PARQ = EDA_DIR / "stock_returns_varonly.parquet"

def load_mu_from_metrics(model_name: str) -> pd.Series:
    p1 = MM_DIR / f"metrics_{model_name}.json"
    if p1.exists():
        with open(p1, "r") as f:
            d = json.load(f)
        recs = []
        for asset, v in d.items():
            mu_d = v.get("latest_pred_next_return", np.nan)
            recs.append((asset, mu_d))
        s = pd.Series({a: mu for a, mu in recs}, dtype=float)
        s = s.dropna()
        return s
    p2 = SUM_DIR / "per_asset_metrics_all_models.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        df = df[df["model"].str.lower() == model_name.lower()].copy()
        if not df.empty:
            s = df.set_index("asset")["latest_pred_next_return"].astype(float)
            s = s.dropna()
            return s
    p3 = ROOT / "outputs" / "ml" / "metrics.json"
    if p3.exists():
        with open(p3, "r") as f:
            d = json.load(f)
        recs = []
        for asset, v in d.items():
            mu_d = v.get("latest_pred_next_return", np.nan)
            recs.append((asset, mu_d))
        s = pd.Series({a: mu for a, mu in recs}, dtype=float).dropna()
        return s
    raise FileNotFoundError("Could not find ML metrics for expected returns.")

def compute_covariance_annualized(rets: pd.DataFrame) -> pd.DataFrame:
    R = rets.drop(columns=[rets.columns[0]]).copy()
    cov_daily = R.cov(min_periods=50)
    cov_ann = cov_daily * 252.0
    return cov_ann

def align_mu_sigma(mu_daily: pd.Series, cov_ann: pd.DataFrame):
    common = sorted(list(set(mu_daily.index).intersection(cov_ann.columns)))
    if len(common) < 2:
        raise ValueError(f"Too few intersecting assets between mu and Sigma: {len(common)}")
    mu_ann = (mu_daily.loc[common] * 252.0).astype(float)
    S = cov_ann.loc[common, common].astype(float)
    return mu_ann, S

def optimize_portfolios(mu_ann: pd.Series, Sigma: pd.DataFrame, risk_free: float = 0.02):
    if EfficientFrontier is None:
        raise ImportError("PyPortfolioOpt not available. Install with: pip install PyPortfolioOpt")
    results = {}
    ef = EfficientFrontier(mu_ann, Sigma, weight_bounds=(0.0, 1.0))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    w_aggr = ef.max_sharpe(risk_free_rate=risk_free)
    perf_a = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free)
    results["aggressive"] = {"weights": w_aggr, "perf": perf_a}
    ef = EfficientFrontier(mu_ann, Sigma, weight_bounds=(0.0, 1.0))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    w_cons = ef.min_volatility()
    perf_c = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free)
    results["conservative"] = {"weights": w_cons, "perf": perf_c}
    assets = list(mu_ann.index)
    wa = pd.Series(w_aggr).reindex(assets).fillna(0.0).values
    wc = pd.Series(w_cons).reindex(assets).fillna(0.0).values
    wm = (0.5 * wa + 0.5 * wc)
    wm = wm / wm.sum()
    target_ret = float((mu_ann.values @ wm))
    ef = EfficientFrontier(mu_ann, Sigma, weight_bounds=(0.0, 1.0))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef.efficient_return(target_ret=target_ret)
    w_mod = ef.clean_weights()
    perf_m = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free)
    results["moderate"] = {"weights": w_mod, "perf": perf_m}
    return results

def save_results(results: dict, out_dir: Path):
    rows = []
    for name, d in results.items():
        w = pd.Series(d["weights"], name="weight")
        w.index.name = "asset"
        w = w[w > 0].sort_values(ascending=False)
        w.to_csv(out_dir / f"weights_{name}.csv")
        er, vol, sharpe = d["perf"]
        rows.append({"portfolio": name, "exp_return": er, "volatility": vol, "sharpe": sharpe})
    pd.DataFrame(rows).to_csv(out_dir / "portfolio_kpis.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

def main():
    ap = argparse.ArgumentParser(description="Optimize portfolios from ML outputs + EDA covariance")
    ap.add_argument("--model", type=str, default="ElasticNet")
    ap.add_argument("--risk_free", type=float, default=0.02)
    args = ap.parse_args()
    if not RET_PARQ.exists():
        raise FileNotFoundError(f"Missing returns parquet: {RET_PARQ}. Run the EDA step first.")
    rets = pd.read_parquet(RET_PARQ)
    Sigma = compute_covariance_annualized(rets)
    mu_daily = load_mu_from_metrics(args.model)
    mu_ann, S = align_mu_sigma(mu_daily, Sigma)
    results = optimize_portfolios(mu_ann, S, risk_free=args.risk_free)
    save_results(results, OUT_DIR)
    print("[OK] Saved:")
    for fn in ["weights_aggressive.csv","weights_conservative.csv","weights_moderate.csv","portfolio_kpis.csv","summary.json"]:
        print(" -", OUT_DIR / fn)

if __name__ == "__main__":
    main()

#%%
import os, json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(".").resolve()
EDA_DIR = ROOT / "outputs" / "eda"
MM_DIR  = ROOT / "outputs" / "ml_multimodel"
OUT_DIR = ROOT / "outputs" / "optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RET_PARQ = EDA_DIR / "stock_returns_varonly.parquet"
METRICS  = MM_DIR / "metrics_ElasticNet.json"  # change model name

print("[paths]")
print("ROOT:", ROOT)
print("RET_PARQ exists?", RET_PARQ.exists())
print("METRICS exists? ", METRICS.exists())


rets = pd.read_parquet(RET_PARQ)
date_col = rets.columns[0]
R = rets.drop(columns=[date_col])
cov_ann = R.cov(min_periods=50) * 252.0


with open(METRICS, "r") as f:
    d = json.load(f)
mu_daily = pd.Series({a: v.get("latest_pred_next_return", np.nan) for a, v in d.items()}, dtype=float).dropna()

print("\n[shapes]")
print("R (returns matrix) shape:", R.shape)
print("cov_ann shape:", cov_ann.shape)
print("mu_daily size:", mu_daily.shape)

common = sorted(set(mu_daily.index).intersection(cov_ann.columns))
print("\n[alignment]")
print("common assets:", len(common))

if len(common) < 2:
    missing_from_cov = [a for a in mu_daily.index if a not in cov_ann.columns][:20]
    missing_from_mu  = [a for a in cov_ann.columns if a not in mu_daily.index][:20]
    print("Assets in mu but not in covariance (sample):", missing_from_cov)
    print("Assets in covariance but not in mu (sample):", missing_from_mu)
    raise SystemExit("Too few intersecting assets. Fix names/universe and rerun.")

mu_ann = (mu_daily.loc[common] * 252.0).astype(float)
Sigma  = cov_ann.loc[common, common].astype(float)

# Optimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

risk_free = 0.02

def max_sharpe(mu, S, rf):
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))

    w = ef.max_sharpe(risk_free_rate=rf)
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
    return w, perf

def min_vol(mu, S, rf):
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    w = ef.min_volatility()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
    return w, perf

w_aggr, perf_a = max_sharpe(mu_ann, Sigma, risk_free)
w_cons,  perf_c = min_vol(mu_ann, Sigma, risk_free)

assets = list(mu_ann.index)
wa = pd.Series(w_aggr).reindex(assets).fillna(0.0).values
wc = pd.Series(w_cons ).reindex(assets).fillna(0.0).values
wm = (0.5 * wa + 0.5 * wc)
wm = wm / wm.sum()

target_return = float(mu_ann.values @ wm)

ef = EfficientFrontier(mu_ann, Sigma, weight_bounds=(0.0, 1.0))
ef.add_objective(objective_functions.L2_reg, gamma=0.001)
ef.efficient_return(target_return=target_return)
w_mod = ef.clean_weights()
perf_m = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free)

from pathlib import Path
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _save(weights, name):
    s = pd.Series(weights, name="weight")
    s.index.name = "asset"
    s = s[s > 0].sort_values(ascending=False)
    s.to_csv(OUT_DIR / f"weights_{name}.csv")

_save(w_aggr, "aggressive")
_save(w_cons, "conservative")
_save(w_mod,  "moderate")

kpis = pd.DataFrame([
    {"portfolio":"aggressive",   "exp_return": perf_a[0], "volatility": perf_a[1], "sharpe": perf_a[2]},
    {"portfolio":"conservative", "exp_return": perf_c[0], "volatility": perf_c[1], "sharpe": perf_c[2]},
    {"portfolio":"moderate",     "exp_return": perf_m[0], "volatility": perf_m[1], "sharpe": perf_m[2]},
])
kpis.to_csv(OUT_DIR / "portfolio_kpis.csv", index=False)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(
        {
            "aggressive":   {"weights": w_aggr, "perf": {"exp_return": perf_a[0], "vol": perf_a[1], "sharpe": perf_a[2]}},
            "conservative": {"weights": w_cons, "perf": {"exp_return": perf_c[0], "vol": perf_c[1], "sharpe": perf_c[2]}},
            "moderate":     {"weights": w_mod,  "perf": {"exp_return": perf_m[0], "vol": perf_m[1], "sharpe": perf_m[2]}},
        },
        f, indent=2
    )
print("[OK] wrote outputs to", OUT_DIR)
