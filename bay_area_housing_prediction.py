"""
Bay Area Housing Price Prediction
==================================
Trains a Gradient Boosting model (best from comparison study) and exports
housing_data.geojson for bay_area_housing_dashboard.html.

Model: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
Expected performance: R2=0.9274, MAPE=11.33%

Usage:
    cd Final_version_work
    python bay_area_housing_prediction.py
    python -m http.server 8080
    # open http://localhost:8080/bay_area_housing_dashboard.html
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_FILE = "bay_area_properties_kid_friendly_score_cool_score.csv"
OUT_GEOJSON = "housing_data.geojson"

print("=" * 70)
print("BAY AREA HOUSING PRICE PREDICTION")
print("Model: Gradient Boosting (best from comparison study)")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
print("\n[1/5] Loading data...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df):,} properties, {len(df.columns)} columns")
print(f"  Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")

# Save display/geo columns BEFORE feature engineering touches the frame
geo = df[
    [
        "latitude", "longitude", "city", "zip_code", "property_type",
        "bedrooms", "bathrooms", "sqft", "sale_price",
        "coolness_index", "kid_friendly_score",
    ]
].copy()

# -----------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (exact match to notebook cell 1)
# -----------------------------------------------------------------------
print("\n[2/5] Feature engineering...")

df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month
df["sale_quarter"] = df["sale_date"].dt.quarter
df["property_age"] = 2026 - df["year_built"]
df["is_new_construction"] = (df["property_age"] <= 5).astype(int)
df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1)
df["sqft_per_bedroom"] = df["sqft"] / (df["bedrooms"] + 1)
df["amenity_score"] = df["pool"] + df["fireplace"] + df["parking_spaces"]
df["has_hoa"] = (df["hoa_fee"] > 0).astype(int)
df["is_single_family"] = (df["property_type"] == "Single Family").astype(int)

# Re-compute city/zip averages and merge (creates _x/_y suffixes like notebook)
df_zip_avg = (
    df.groupby("zip_code")["sale_price"]
    .mean()
    .reset_index()
    .rename(columns={"sale_price": "avg_zip_code_sale_price"})
)
df_city_avg = (
    df.groupby("city")["sale_price"]
    .mean()
    .reset_index()
    .rename(columns={"sale_price": "avg_city_sale_price"})
)
df = df.merge(df_zip_avg, on="zip_code").merge(df_city_avg, on="city")

print(f"  Total columns after engineering: {len(df.columns)}")

# -----------------------------------------------------------------------
# 3. PREPARE ML DATA  (exact match to notebook)
# -----------------------------------------------------------------------
print("\n[3/5] Preparing ML data...")

y_price = df["sale_price"].copy()

# Replicate notebook DROP_COLS exactly, including the tab-key bug so that
# walk_score and transit_score remain as features (matching 77-feature set)
DROP_COLS = [
    "property_id",
    "sale_date",
    "sale_price",
    "latitude",
    "longitude",
    "year_built",
    "zip_code",
    "is_single_family",
    "price_per_sqft",
    "price_per_sqft_log",          # not in CSV, silently skipped
    "n_parks",
    "Attractions",
    "Restaurants",
    "Things_to_do_score",
    "resturant_score",
    "walk_score\ttransit_score",   # intentional tab: matches notebook bug, not dropped
    "bike_score",
]

df_ml = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

cat_cols = [c for c in ["property_type", "county", "city"] if c in df_ml.columns]
df_encoded = pd.get_dummies(df_ml, columns=cat_cols, drop_first=False)

X = df_encoded.select_dtypes(include=["number", "bool"])

print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
print(f"  Dropped: {[c for c in DROP_COLS if c in df.columns]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# -----------------------------------------------------------------------
# 4. TRAIN GRADIENT BOOSTING  (best model from comparison study)
# -----------------------------------------------------------------------
print("\n[4/5] Training Gradient Boosting...")
print("  n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42")
print("  (This may take 3-6 minutes)")

gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)
gb.fit(X_train, y_train)

y_pred_test = gb.predict(X_test)
r2 = float(r2_score(y_test, y_pred_test))
mae = float(mean_absolute_error(y_test, y_pred_test))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)

print(f"\n  === Test Performance ===")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  ${mae:,.0f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  MAPE: {mape:.2f}%")

# Predict for ALL properties (for the dashboard)
all_pred = gb.predict(X)

# -----------------------------------------------------------------------
# 5. EXPORT GEOJSON
# -----------------------------------------------------------------------
print("\n[5/5] Exporting GeoJSON...")

median_price = int(np.median(geo["sale_price"]))
n_cities = int(geo["city"].nunique())

features = []
for i in range(len(geo)):
    row = geo.iloc[i]
    pp = int(round(float(all_pred[i])))
    features.append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    round(float(row.longitude), 5),
                    round(float(row.latitude), 5),
                ],
            },
            "properties": {
                "ct": str(row.city),
                "zp": str(row.zip_code),
                "tp": str(row.property_type),
                "bd": int(row.bedrooms),
                "ba": float(row.bathrooms),
                "sf": int(row.sqft),
                "pr": int(row.sale_price),
                "pp": pp,
                "ci": round(float(row.coolness_index), 1),
                "ks": round(float(row.kid_friendly_score), 1),
            },
        }
    )

geojson = {
    "type": "FeatureCollection",
    "meta": {
        "model": "Gradient Boosting",
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "n_properties": len(features),
        "n_cities": n_cities,
        "median_price": median_price,
    },
    "features": features,
}

with open(OUT_GEOJSON, "w") as f:
    json.dump(geojson, f, separators=(",", ":"))

size_mb = os.path.getsize(OUT_GEOJSON) / 1_000_000
print(f"  Exported {len(features):,} features -> {OUT_GEOJSON} ({size_mb:.1f} MB)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("\nNext steps:")
print("  cd Final_version_work")
print("  python3 -m http.server 8080")
print("  Open: http://localhost:8080/bay_area_housing_dashboard.html")
