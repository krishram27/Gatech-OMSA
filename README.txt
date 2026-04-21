Bay Area Housing Price Prediction
==================================
1. Description
At a high level, this codebase trains a Gradient Boosting model (best from comparison study) and exports a
housing_data.geojson to create our dashboard with bay_area_housing_dashboard.html. Each file in the
repository plays a vital role in bringing our solution to life. 

**bay_area_housing_prediction.py** This Python file is used to 
**generate_geojson.py** This Python file is used to
**housing_dashboard_final.html** This HTML renders our entire dashboard using
**bay_area_properties_kid_friendly_score_cool_score.csv** This CSV file contains
**ML_model_predictions_compareV3.ipynb** This Jupyter file showcasese our process (insert Linlin's README content here)

2. Installation
Use the following steps to install and set up our codebase:
    1. Create a folder on you local machine titled 'Bay_Area_Housing'
        e.g. mkdir ~/Bay_Area_Housing
    2. Download all the files provided in our zip file
    3. Move all downloaded files to the Bay_Area_Housing folder
        e.g. mv bay_area_housing_prediction.py ~/Bay_Area_Housing

3. Execution
Use the following steps to launch our dashboard:
    1. Ensure you're in the Bay_Area_Housing directory
        e.g. cd ~/Bay_Area_Housing
    2. Start a http server:
        python -m http.server 8080
    3. Open http://localhost:8080/bay_area_housing_dashboard.html
