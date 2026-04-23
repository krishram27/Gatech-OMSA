==============================================================================
BAY AREA HOUSING PRICE PREDICTION
==============================================================================


1. DESCRIPTION
--------------

At a high level, this codebase trains a Gradient Boosting model (the best
performer from our comparison study) and exports a housing_data_final.geojson
that powers an interactive dashboard at housing_dashboard_final.html. The
dashboard lets a user explore ~50,000 Bay Area properties on two linked maps
(Explore + Affordability), filter by county, property type, bedrooms,
coolness score, and kids score, and stress test affordability against
mortgage rate, income, down payment, and housing budget. Each file in the
repository plays a specific role in bringing the solution to life.

  bay_area_housing_prediction.py
    The end-to-end training and export script. It loads
    bay_area_properties_kid_friendly_score_cool_score.csv, performs feature
    engineering (time features, property age, bath/bed ratio, amenity score,
    zip and city average prices, one-hot encoded property type / county /
    city), trains a GradientBoostingRegressor with n_estimators=200,
    learning_rate=0.05, max_depth=5, random_state=42, reports held-out
    R2/MAE/RMSE/MAPE, and then writes every property's predicted price plus
    display fields into housing_data_final.geojson. Coordinates are
    validated against a San Francisco Bay water polygon and the claimed
    city's centroid, and jittered back on land when the raw coordinate falls
    in the bay or in the wrong city. Expected test metrics: R2 = 0.9274,
    MAPE = 11.33%.

  housing_dashboard_final.html
    The end-user dashboard. A single self-contained HTML file that uses
    Mapbox GL JS for the two maps, D3 for the vertical coolness / kids
    score sliders and the affordability results table, and vanilla JS for
    filter pills and the affordability math. On load it fetches
    housing_data_final.geojson (with a cache-buster so you always get the
    current file), parses features into an in-memory array, and drives
    every view off the same filtered subset. Hover popups show the ML
    predicted price, walk/transit scores, lot size, coolness score, and
    kids score per property.

  bay_area_properties_kid_friendly_score_cool_score.csv
    The source dataset. 49,907 Bay Area property sales (2021-03-14 to
    2026-03-12) across 44 cities in 3 counties (Alameda, San Mateo, Santa
    Clara). 44 columns including sale_price, sqft, bedrooms, bathrooms,
    year_built, lot_size, latitude, longitude, hoa_fee, walk_score,
    transit_score, coolness_index, kid_friendly_score,
    avg_zip_code_sale_price, and avg_city_sale_price. Every
    housing_data_final.geojson feature is derived from one row in this
    file.


2. INSTALLATION
---------------

Use the following steps to install and set up our codebase:

  1. Unzip our submitted Zip file
  
  2. Navigate to the unzipped folder on your terminal

  3. Enter the CODE folder

         cd CODE

  4. Install the Python dependencies used by bay_area_housing_prediction.py

         pip install numpy pandas scikit-learn

     Python 3.9 or newer is recommended.


3. EXECUTION
------------

Use the following steps to launch our dashboard:

  1. Ensure you're in the Bay_Area_Housing directory

         cd ~/Bay_Area_Housing

  2. Run bay_area_housing_prediction.py to train the model and generate
     housing_data_final.geojson

         python bay_area_housing_prediction.py

     The script takes 1-2 minutes on a modern laptop. When it finishes,
     it prints the test metrics (R2, MAE, RMSE, MAPE) and the exact
     command to start the local web server.

  3. Copy and paste the output of the Python file into your terminal. It
     should look something like this:

         cd "/Users/you/Bay_Area_Housing"
         python3 -m http.server 8080

  4. Open http://localhost:8080/housing_dashboard_final.html in your
     browser and you're good to go.

     If the map looks empty or the dots look misplaced, hard-refresh the
     tab (Cmd+Shift+R on macOS, Ctrl+Shift+R on Windows/Linux) to clear
     any cached copy of housing_data_final.geojson from a prior run.