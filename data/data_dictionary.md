# Data Dictionary

This document describes the datasets and features used in the Nepal Earthquake Risk Prediction project.

## Datasets

*   **Building Data (Primary):** `data/processed/buildings_features_earthquakes.csv`
    *   Source: Derived from DrivenData competition ("Richter's Predictor: Modeling Earthquake Damage") combined with main Gorkha earthquake event features. Original survey data from Nepal's National Planning Commission, Katmandu Living Labs, and Central Bureau of Statistics.
    *   Description: Contains structural, ownership, usage, and location features for buildings affected by the 2015 Gorkha earthquake, merged with basic features of the main shock.
*   **Shapefiles (Administrative Boundaries):** `data/npl_adm_nd_20240314_ab_shp/`
    *   Source: Likely Nepal government sources or international administrative boundary datasets (e.g., OCHA, HDX). Dated March 2024.
    *   Description: Contains shapefiles for various administrative levels in Nepal (ADM0 - Country, ADM1 - Districts, ADM2 - Municipalities/VDCs, ADM3 - Wards). Used for map visualizations.
*   **Raw DrivenData Files:** `data/nepal_land_data/`
    *   Source: DrivenData competition.
    *   Description: Original training values (`train_values.csv`), training labels (`train_labels.csv`), and test values (`test_values.csv`). Used as the basis for the primary building dataset.

## Processed Data: `buildings_features_earthquakes.csv`

This file contains the merged and preprocessed data used for model training and evaluation. It combines building structure information with features derived from the main 2015 Gorkha earthquake event.

| Column Name                           | Description                                                                 | Data Type        | Source/Notes                                     |
|---------------------------------------|-----------------------------------------------------------------------------|------------------|--------------------------------------------------|
| `building_id`                         | Unique identifier for each building                                         | Integer          | DrivenData Building Dataset                      |
| `geo_level_1_id`                      | Geographic region ID (Level 1 - Largest, typically District)                | Integer          | DrivenData Building Dataset                      |
| `geo_level_2_id`                      | Geographic region ID (Level 2 - Typically Municipality or VDC)              | Integer          | DrivenData Building Dataset                      |
| `geo_level_3_id`                      | Geographic region ID (Level 3 - Smallest, typically Ward)                   | Integer          | DrivenData Building Dataset                      |
| `count_floors_pre_eq`                 | Number of floors before the earthquake                                      | Integer          | DrivenData Building Dataset                      |
| `age`                                 | Age of the building (years)                                                 | Integer          | DrivenData Building Dataset                      |
| `area_percentage`                     | Normalized area of the building footprint                                   | Float            | DrivenData Building Dataset                      |
| `height_percentage`                   | Normalized height of the building                                           | Float            | DrivenData Building Dataset                      |
| `land_surface_condition`              | Condition of the land surface where the building is located                 | Categorical      | DrivenData Building Dataset (n: Flat, o: Moderate slope, t: Steep slope) |
| `foundation_type`                     | Type of foundation used for the building                                  | Categorical      | DrivenData Building Dataset (h: Adobe/Mud, i: Bamboo/Timber, r: RC - Reinforced Concrete, u: Brick/Cement Mortar, w: Stone/Cement Mortar) |
| `roof_type`                           | Type of roof used for the building                                          | Categorical      | DrivenData Building Dataset (n: RCC/RB/RBC, q: Bamboo/Timber-Light roof, x: Bamboo/Timber-Heavy roof) |
| `ground_floor_type`                 | Material used for the ground floor                                          | Categorical      | DrivenData Building Dataset (f: Mud/Adobe, m: Mud Mortar-Stone/Brick, v: Cement-Stone/Brick, x: Timber, z: Other) |
| `other_floor_type`                    | Material used for upper floors                                              | Categorical      | DrivenData Building Dataset (j: Timber, q: RCC/RB/RBC, s: Tiled/Stone/Slate, x: Mud/Adobe) |
| `position`                            | Position of the building relative to others                                 | Categorical      | DrivenData Building Dataset (j: Attached-1 side, o: Attached-2 sides, s: Not attached, t: Attached-3 sides) |
| `plan_configuration`                  | Shape of the building plan                                                  | Categorical      | DrivenData Building Dataset (a: A-shape, c: C-shape, d: Rectangular, f: F-shape, h: H-shape, l: L-shape, m: Multi-projected, n: N-shape, o: Others, q: Square, s: S-shape, t: T-shape, u: U-shape, z: Z-shape) |
| `has_superstructure_adobe_mud`        | Binary flag: Building superstructure includes Adobe/Mud                   | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_mud_mortar_stone` | Binary flag: Building superstructure includes Mud Mortar Stone            | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_stone_flag`       | Binary flag: Building superstructure includes Stone Flag                  | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_cement_mortar_stone`| Binary flag: Building superstructure includes Cement Mortar Stone         | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_mud_mortar_brick` | Binary flag: Building superstructure includes Mud Mortar Brick            | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_cement_mortar_brick`| Binary flag: Building superstructure includes Cement Mortar Brick         | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_timber`           | Binary flag: Building superstructure includes Timber                      | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_bamboo`           | Binary flag: Building superstructure includes Bamboo                      | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_rc_non_engineered`| Binary flag: Building superstructure includes RC (Non-Engineered)         | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_rc_engineered`    | Binary flag: Building superstructure includes RC (Engineered)             | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_superstructure_other`            | Binary flag: Building superstructure includes Other materials             | Binary (0/1)     | DrivenData Building Dataset                      |
| `legal_ownership_status`              | Legal ownership status of the land                                          | Categorical      | DrivenData Building Dataset (a: Attached, r: Rented, v: Private, w: Other/Unknown) |
| `count_families`                      | Number of families living in the building                                   | Integer          | DrivenData Building Dataset                      |
| `has_secondary_use`                   | Binary flag: Building has a secondary use                                   | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_agriculture`       | Binary flag: Secondary use is Agriculture                                   | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_hotel`             | Binary flag: Secondary use is Hotel                                         | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_rental`            | Binary flag: Secondary use is Rental                                        | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_institution`     | Binary flag: Secondary use is Institution                                   | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_school`            | Binary flag: Secondary use is School                                        | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_industry`          | Binary flag: Secondary use is Industry                                      | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_health_post`     | Binary flag: Secondary use is Health Post                                 | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_gov_office`        | Binary flag: Secondary use is Government Office                             | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_use_police`        | Binary flag: Secondary use is Police Station                                | Binary (0/1)     | DrivenData Building Dataset                      |
| `has_secondary_use_other`             | Binary flag: Secondary use is Other                                         | Binary (0/1)     | DrivenData Building Dataset                      |
| `damage_grade`                        | Damage grade assigned after the earthquake (Target Variable)              | Integer (1, 2, 3)| DrivenData Building Dataset (1: Low, 2: Medium, 3: High/Complete) |
| `main_eq_magnitude`                   | Magnitude of the main Gorkha earthquake (Mw 7.8)                            | Float            | Engineered Feature (USGS/NSC Data)             |
| `main_eq_depth`                       | Depth of the main Gorkha earthquake (km)                                  | Float            | Engineered Feature (USGS/NSC Data)             |
| `main_eq_epicenter_lat`               | Latitude of the main Gorkha earthquake epicenter                            | Float            | Engineered Feature (USGS/NSC Data)             |
| `main_eq_epicenter_lon`               | Longitude of the main Gorkha earthquake epicenter                           | Float            | Engineered Feature (USGS/NSC Data)             |

## Raw Data Sources

*(Placeholder for descriptions of raw datasets like USGS earthquake catalog, NSC aftershock data, SRTM elevation, original DrivenData files, etc.)* 