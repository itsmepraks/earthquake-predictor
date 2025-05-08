# Test Data Examples for Earthquake Risk Predictor

This document provides example input configurations for the Streamlit application to help showcase Low, Medium, and High risk predictions. These are primarily tailored for the **LightGBM (Tuned)** model but can serve as a starting point for testing other models.

**Note:** The earthquake parameters (`main_eq_magnitude`, `main_eq_depth`, `main_eq_epicenter_lat`, `main_eq_epicenter_lon`) are set to their fixed training values in these examples, as handled by the application logic. For the "LightGBM (Tuned)" model, if both `area_percentage` and `height_percentage` are expected features, the UI's "Building Height (ft)" slider controls `area_percentage`, and `height_percentage` will be defaulted to `0.0` by the application.

## Model: LightGBM (Tuned)

### Low Risk Example

This configuration represents a new, small, single-story building with strong construction materials on flat land.

```json
{
    "geo_level_1_id": 13,
    "geo_level_2_id": 702,
    "geo_level_3_id": 6261,
    "count_floors_pre_eq": 1,
    "age": 0,
    "area_percentage": 10,
    "height_percentage": 0.0, 
    "land_surface_condition": "n",
    "foundation_type": "r",
    "roof_type": "n",
    "ground_floor_type": "v",
    "other_floor_type": "q",
    "position": "s",
    "plan_configuration": "d",
    "has_superstructure_adobe_mud": 0,
    "has_superstructure_mud_mortar_stone": 0,
    "has_superstructure_stone_flag": 0,
    "has_superstructure_cement_mortar_stone": 0,
    "has_superstructure_mud_mortar_brick": 0,
    "has_superstructure_cement_mortar_brick": 0,
    "has_superstructure_timber": 0,
    "has_superstructure_bamboo": 0,
    "has_superstructure_rc_non_engineered": 0,
    "has_superstructure_rc_engineered": 1,
    "has_superstructure_other": 0,
    "legal_ownership_status": "v",
    "count_families": 1,
    "has_secondary_use": 0,
    "has_secondary_use_agriculture": 0,
    "has_secondary_use_hotel": 0,
    "has_secondary_use_rental": 0,
    "has_secondary_use_institution": 0,
    "has_secondary_use_school": 0,
    "has_secondary_use_industry": 0,
    "has_secondary_use_health_post": 0,
    "has_secondary_use_gov_office": 0,
    "has_secondary_use_use_police": 0,
    "has_secondary_use_other": 0,
    "main_eq_magnitude": 7.8,
    "main_eq_depth": 15.0,
    "main_eq_epicenter_lat": 28.23,
    "main_eq_epicenter_lon": 84.73
}
```

### Medium Risk Example

This configuration represents a moderately aged, multi-story building with average construction materials and some secondary use, on a moderate slope.

```json
{
    "geo_level_1_id": 13,
    "geo_level_2_id": 702,
    "geo_level_3_id": 6261,
    "count_floors_pre_eq": 3,
    "age": 30,
    "area_percentage": 50,
    "height_percentage": 0.0,
    "land_surface_condition": "o",
    "foundation_type": "u",
    "roof_type": "q",
    "ground_floor_type": "x",
    "other_floor_type": "j",
    "position": "j",
    "plan_configuration": "d",
    "has_superstructure_adobe_mud": 0,
    "has_superstructure_mud_mortar_stone": 0,
    "has_superstructure_stone_flag": 0,
    "has_superstructure_cement_mortar_stone": 0,
    "has_superstructure_mud_mortar_brick": 0,
    "has_superstructure_cement_mortar_brick": 1,
    "has_superstructure_timber": 0,
    "has_superstructure_bamboo": 0,
    "has_superstructure_rc_non_engineered": 0,
    "has_superstructure_rc_engineered": 0,
    "has_superstructure_other": 0,
    "legal_ownership_status": "v",
    "count_families": 2,
    "has_secondary_use": 1,
    "has_secondary_use_agriculture": 0,
    "has_secondary_use_hotel": 0,
    "has_secondary_use_rental": 1,
    "has_secondary_use_institution": 0,
    "has_secondary_use_school": 0,
    "has_secondary_use_industry": 0,
    "has_secondary_use_health_post": 0,
    "has_secondary_use_gov_office": 0,
    "has_secondary_use_use_police": 0,
    "has_secondary_use_other": 0,
    "main_eq_magnitude": 7.8,
    "main_eq_depth": 15.0,
    "main_eq_epicenter_lat": 28.23,
    "main_eq_epicenter_lon": 84.73
}
```

### High Risk Example

This configuration represents a very old, tall, and large area building with weak construction materials, on a steep slope, and in a potentially higher-risk geographic zone.

```json
{
    "geo_level_1_id": 26, 
    "geo_level_2_id": 802, 
    "geo_level_3_id": 10000,
    "count_floors_pre_eq": 7,
    "age": 100,
    "area_percentage": 100,
    "height_percentage": 0.0,
    "land_surface_condition": "t",
    "foundation_type": "h",
    "roof_type": "x",
    "ground_floor_type": "f",
    "other_floor_type": "x",
    "position": "o",
    "plan_configuration": "m",
    "has_superstructure_adobe_mud": 1,
    "has_superstructure_mud_mortar_stone": 1,
    "has_superstructure_stone_flag": 0,
    "has_superstructure_cement_mortar_stone": 0,
    "has_superstructure_mud_mortar_brick": 0,
    "has_superstructure_cement_mortar_brick": 0,
    "has_superstructure_timber": 0,
    "has_superstructure_bamboo": 0,
    "has_superstructure_rc_non_engineered": 0,
    "has_superstructure_rc_engineered": 0,
    "has_superstructure_other": 1,
    "legal_ownership_status": "r",
    "count_families": 5,
    "has_secondary_use": 1,
    "has_secondary_use_agriculture": 0,
    "has_secondary_use_hotel": 0,
    "has_secondary_use_rental": 0,
    "has_secondary_use_institution": 0,
    "has_secondary_use_school": 0,
    "has_secondary_use_industry": 0,
    "has_secondary_use_health_post": 0,
    "has_secondary_use_gov_office": 0,
    "has_secondary_use_use_police": 0,
    "has_secondary_use_other": 1,
    "main_eq_magnitude": 7.8,
    "main_eq_depth": 15.0,
    "main_eq_epicenter_lat": 28.23,
    "main_eq_epicenter_lon": 84.73
}
```

