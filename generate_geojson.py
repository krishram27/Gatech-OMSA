"""
Generate housing_data_final.geojson from the CSV dataset.
No ML dependencies - uses only Python stdlib.

Includes bay-water polygon check: any coordinate that falls inside San
Francisco Bay is relocated to the city centroid (with small random jitter)
so no dots appear in the water on the map.

Runs in ~20-30 seconds.

Usage:
    cd Final_version_work
    python generate_geojson.py
    python -m http.server 8080
    # open http://localhost:8080/housing_dashboard_final.html
"""
import csv
import json
import os
import random

DATA_FILE = "bay_area_properties_kid_friendly_score_cool_score.csv"
OUT_FILE  = "housing_data_final.geojson"

# ---------------------------------------------------------------------------
# City centroids (lat, lon, jitter_radius)
# Ported from housing_prediction_v2.py
# ---------------------------------------------------------------------------
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

# Approximate polygon of San Francisco Bay open water.
# Ported from housing_prediction_v2.py
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
    """Ray-casting point-in-polygon test."""
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
    """Approximate Euclidean distance in degrees."""
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * 0.82  # rough lon correction at lat ~37.5
    return (dlat ** 2 + dlon ** 2) ** 0.5


def _safe_coords(lat, lon, city, rng, max_retries=60):
    """Return (lat, lon) that is:
      1. Not in the bay, AND
      2. Within a reasonable distance of the city centroid.

    This fixes two classes of bad data in the source CSV:
      - Coordinates that fall in San Francisco Bay (water).
      - Coordinates that are on land but in the wrong city's territory
        (e.g., a 'Sunnyvale' property whose coords land in Santa Clara).
    """
    clat, clon, radius = _CITY_COORDS.get(city, (37.55, -122.05, 0.010))
    # Max allowed distance from city centroid: generous enough for large
    # cities but tight enough to keep dots in the right visual area.
    max_dist = max(0.045, radius * 2.5)

    in_water = _in_bay(lat, lon)
    too_far  = _dist(lat, lon, clat, clon) > max_dist

    if not in_water and not too_far:
        return lat, lon  # original coords are fine

    # Need to relocate: jitter around city centroid until on land
    for _ in range(max_retries):
        jlat = clat + rng.gauss(0, radius)
        jlon = clon + rng.gauss(0, radius)
        if not _in_bay(jlat, jlon):
            return round(jlat, 5), round(jlon, 5)
    return clat, clon  # fallback: exact centroid


def main():
    print("=" * 60)
    print("GENERATE housing_data_final.geojson")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(DATA_FILE + " not found in current directory")

    rng = random.Random(42)
    features   = []
    prices     = []
    cities     = set()
    skipped    = 0
    relocated  = 0   # water + wrong-city combined

    print(f"\nReading {DATA_FILE} ...")
    print("  (validating coordinates: water check + city-boundary check...)")
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            if lat == 0 or lon == 0:
                skipped += 1
                continue

            try:
                sp = int(float(row["sale_price"]))
            except (ValueError, KeyError):
                skipped += 1
                continue

            city = row.get("city", "")

            # _safe_coords handles both water AND wrong-city cases
            new_lat, new_lon = _safe_coords(lat, lon, city, rng)
            if new_lat != lat or new_lon != lon:
                lat, lon = new_lat, new_lon
                relocated += 1

            cities.add(city)
            prices.append(sp)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 5), round(lat, 5)],
                },
                "properties": {
                    "ct": row.get("city",          ""),
                    "co": row.get("county",        ""),
                    "zp": row.get("zip_code",      ""),
                    "tp": row.get("property_type", ""),
                    "bd": _int(row.get("bedrooms",  0)),
                    "ba": _flt(row.get("bathrooms", 0)),
                    "sf": _int(row.get("sqft",      0)),
                    "pr": sp,
                    # pp = sale_price (replace with ML predictions when available)
                    "pp": sp,
                    # coolness_index: 0-100 scale (dataset range 8-79)
                    "ci": round(_flt(row.get("coolness_index",    0)), 1),
                    # kid_friendly_score: 0-10 scale
                    "ks": round(_flt(row.get("kid_friendly_score",0)), 1),
                    "ws": _int(row.get("walk_score",    0)),
                    "ts": _int(row.get("transit_score", 0)),
                    "yr": _int(row.get("year_built",    0)),
                },
            })

    prices.sort()
    median_price = prices[len(prices) // 2] if prices else 0
    n = len(features)
    nc = len(cities)

    print(f"  Coords relocated: {relocated:,}  (water + wrong-city)")

    geojson = {
        "type": "FeatureCollection",
        "meta": {
            "n_properties": n,
            "n_cities":     nc,
            "median_price": median_price,
            "coords_relocated": relocated,
            "note": "pp = sale_price (run bay_area_housing_prediction.py for ML predictions)",
        },
        "features": features,
    }

    print(f"Writing {OUT_FILE} ...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))

    size_mb = os.path.getsize(OUT_FILE) / 1_000_000
    print(f"\nDone.")
    print(f"  Features     : {n:,}")
    print(f"  Cities       : {nc}")
    print(f"  Skipped      : {skipped}")
    print(f"  Relocated    : {relocated:,}  (water + wrong-city coords fixed)")
    print(f"  Median $     : ${median_price:,}")
    print(f"  File size    : {size_mb:.1f} MB -> {OUT_FILE}")
    print(f"\nNext:")
    print(f"  python3 -m http.server 8080")
    print(f"  http://localhost:8080/housing_dashboard_final.html")


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
