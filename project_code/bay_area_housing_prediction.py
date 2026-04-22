"""
Bay Area Housing Price Prediction
==================================
Trains a Gradient Boosting model (best from comparison study) and exports
housing_data_final.geojson for housing_dashboard_final.html.

Model: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
Expected performance: R2=0.9274, MAPE=11.33%

Usage:
    cd into the folder containing housing_dashboard_final.html
    python -m http.server 8080
    # open http://localhost:8080/housing_dashboard_final.html
"""

import json
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "bay_area_properties_kid_friendly_score_cool_score.csv")
OUT_GEOJSON = os.path.join(SCRIPT_DIR, "housing_data_final.geojson")
DASHBOARD_HTML = "housing_dashboard_final.html"

# ---------------------------------------------------------------------------
# Coordinate fixup (ported from Final_version_work/generate_geojson.py)
# ---------------------------------------------------------------------------
# The raw CSV has two classes of bad coordinates that make the map look
# noisy: points that fall in San Francisco Bay (open water) and points
# that land in the wrong city's territory. We ray-cast against a bay-
# water polygon, check distance to the claimed city centroid, and when
# either check fails we jitter the point back to the city centroid.
_CITY_COORDS = {
    "Alameda":             (37.7652, -122.2416, 0.010),
    "Albany":              (37.8869, -122.2978, 0.005),
    "Atherton":            (37.4613, -122.1975, 0.006),
    "Belmont":             (37.5202, -122.2758, 0.006),
    "Berkeley":            (37.8716, -122.2727, 0.012),
    "Burlingame":          (37.5841, -122.3440, 0.007),
    "Campbell":            (37.2872, -121.9500, 0.008),
    "Castro Valley":       (37.6941, -122.0864, 0.010),
    "Colma":               (37.6769, -122.4547, 0.004),
    "Cupertino":           (37.3230, -122.0322, 0.010),
    "Daly City":           (37.6879, -122.4702, 0.008),
    "Dublin":              (37.7022, -121.9358, 0.010),
    "Emeryville":          (37.8313, -122.2852, 0.005),
    "Foster City":         (37.5585, -122.2711, 0.006),
    "Fremont":             (37.5485, -121.9886, 0.018),
    "Gilroy":              (37.0058, -121.5683, 0.012),
    "Half Moon Bay":       (37.4636, -122.4286, 0.008),
    "Hayward":             (37.6688, -122.0808, 0.014),
    "Livermore":           (37.6819, -121.7680, 0.014),
    "Los Altos":           (37.3852, -122.1141, 0.008),
    "Los Altos Hills":     (37.3795, -122.1377, 0.008),
    "Los Gatos":           (37.2266, -121.9746, 0.010),
    "Menlo Park":          (37.4530, -122.1817, 0.008),
    "Millbrae":            (37.5985, -122.3872, 0.005),
    "Milpitas":            (37.4323, -121.8996, 0.010),
    "Morgan Hill":         (37.1305, -121.6544, 0.012),
    "Mountain View":       (37.3861, -122.0839, 0.010),
    "Newark":              (37.5316, -122.0402, 0.008),
    "Oakland":             (37.8044, -122.2712, 0.018),
    "Pacifica":            (37.6138, -122.4869, 0.008),
    "Palo Alto":           (37.4419, -122.1430, 0.012),
    "Pleasanton":          (37.6624, -121.8747, 0.010),
    "Portola Valley":      (37.3841, -122.2350, 0.006),
    "Redwood City":        (37.4852, -122.2364, 0.010),
    "San Bruno":           (37.6305, -122.4111, 0.007),
    "San Carlos":          (37.5072, -122.2602, 0.006),
    "San Jose":            (37.3382, -121.8863, 0.022),
    "San Leandro":         (37.7249, -122.1561, 0.010),
    "San Mateo":           (37.5630, -122.3255, 0.010),
    "Santa Clara":         (37.3541, -121.9552, 0.012),
    "Saratoga":            (37.2639, -122.0230, 0.008),
    "South San Francisco": (37.6547, -122.4077, 0.008),
    "Sunnyvale":           (37.3688, -122.0363, 0.012),
    "Union City":          (37.5934, -122.0439, 0.008),
}

_BAY_WATER_POLY = [
    (37.860, -122.360), (37.880, -122.390), (37.910, -122.420),
    (37.940, -122.450), (37.970, -122.450), (38.060, -122.500),
    (38.060, -122.350), (37.970, -122.380), (37.940, -122.360),
    (37.960, -122.340), (37.895, -122.305), (37.835, -122.300),
    (37.795, -122.275), (37.770, -122.240), (37.745, -122.210),
    (37.700, -122.190), (37.640, -122.155), (37.580, -122.120),
    (37.530, -122.080), (37.490, -122.050), (37.440, -121.990),
    (37.425, -122.020), (37.440, -122.070), (37.470, -122.110),
    (37.510, -122.170), (37.530, -122.210), (37.560, -122.250),
    (37.590, -122.310), (37.620, -122.375), (37.660, -122.380),
    (37.700, -122.380), (37.770, -122.390), (37.810, -122.380),
    (37.830, -122.370), (37.860, -122.360),
]


def _point_in_poly(lat, lon, poly):
    n = len(poly)
    inside = False
    x, y = lon, lat
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _in_bay(lat, lon):
    return _point_in_poly(lat, lon, _BAY_WATER_POLY)


def _dist(lat1, lon1, lat2, lon2):
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * 0.82   # rough longitude correction at lat ~37.5
    return (dlat ** 2 + dlon ** 2) ** 0.5


def _safe_coords(lat, lon, city, rng, max_retries=60):
    """Return (lat, lon) guaranteed to be on land AND near the claimed city.

    If the raw coordinate fails either check, jitter around the city
    centroid (Gaussian with per-city radius) until we land on dry ground.
    Fallback to the exact centroid after max_retries. Returns the original
    coordinate when it passes both checks.
    """
    clat, clon, radius = _CITY_COORDS.get(city, (37.55, -122.05, 0.010))
    max_dist = max(0.045, radius * 2.5)

    in_water = _in_bay(lat, lon)
    too_far  = _dist(lat, lon, clat, clon) > max_dist

    if not in_water and not too_far:
        return lat, lon, False

    for _ in range(max_retries):
        jlat = clat + rng.gauss(0, radius)
        jlon = clon + rng.gauss(0, radius)
        if not _in_bay(jlat, jlon):
            return round(jlat, 5), round(jlon, 5), True
    return clat, clon, True

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
        "latitude", "longitude", "city", "county", "zip_code", "property_type",
        "bedrooms", "bathrooms", "sqft", "lot_size", "year_built", "sale_price",
        "coolness_index", "kid_friendly_score",
        "walk_score", "transit_score",
    ]
].copy()

# -----------------------------------------------------------------------
# 2. FEATURE ENGINEERING  
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

# Re-compute city/zip averages and merge 
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
# 3. PREPARE ML DATA 
# -----------------------------------------------------------------------
print("\n[3/5] Preparing ML data...")

y_price = df["sale_price"].copy()

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
    "price_per_sqft_log",          
    "n_parks",
    "Attractions",
    "Restaurants",
    "Things_to_do_score",
    "resturant_score",
    "walk_score\ttransit_score",  
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
print("  (This may take 1-2 minutes)")

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

def _safe_int(v, default=0):
    try:
        return int(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _safe_num(v, default=0):
    try:
        return float(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


rng = random.Random(42)
relocated = 0
skipped_coords = 0

features = []
for i in range(len(geo)):
    row = geo.iloc[i]

    try:
        lat = float(row.latitude)
        lon = float(row.longitude)
    except (TypeError, ValueError):
        skipped_coords += 1
        continue
    if lat == 0 or lon == 0 or pd.isna(lat) or pd.isna(lon):
        skipped_coords += 1
        continue

    city = str(row.city) if pd.notna(row.city) else ""
    new_lat, new_lon, was_relocated = _safe_coords(lat, lon, city, rng)
    if was_relocated:
        relocated += 1

    pp = int(round(float(all_pred[i])))
    features.append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    round(float(new_lon), 5),
                    round(float(new_lat), 5),
                ],
            },
            "properties": {
                "ct": city,
                "co": str(row.county) if pd.notna(row.county) else "",
                "zp": str(row.zip_code),
                "tp": str(row.property_type),
                "bd": _safe_int(row.bedrooms),
                "ba": _safe_num(row.bathrooms),
                "sf": _safe_int(row.sqft),
                "lo": _safe_int(row.lot_size),
                "yr": _safe_int(row.year_built),
                "pr": _safe_int(row.sale_price),
                "pp": pp,
                "ci": round(_safe_num(row.coolness_index), 1),
                "ks": round(_safe_num(row.kid_friendly_score), 1),
                "ws": _safe_int(row.walk_score),
                "ts": _safe_int(row.transit_score),
            },
        }
    )

print(f"  Coords relocated: {relocated:,}  (water + wrong-city combined)")
if skipped_coords:
    print(f"  Coords skipped:   {skipped_coords:,}  (null / 0 / non-numeric)")

geojson = {
    "type": "FeatureCollection",
    "meta": {
        "model": "Gradient Boosting",
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "n_properties": len(features),
        "n_cities": n_cities,
        "median_price": median_price,
        "coords_relocated": relocated,
        "coords_skipped":   skipped_coords,
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
print(f"  cd \"{SCRIPT_DIR}\"")
print("  python3 -m http.server 8080")
print(f"  Open: http://localhost:8080/{DASHBOARD_HTML}")
