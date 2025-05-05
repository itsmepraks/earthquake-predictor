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

## Feature Descriptions (`buildings_features_earthquakes.csv`)

*(Based primarily on DrivenData competition description)*

**Note on Categorical Codes:** The DrivenData source states that single-letter codes for categorical variables were obfuscated and randomly assigned. While meanings for `foundation_type` have been inferred from external context, other codes (`land_surface_condition`, `roof_type`, `ground_floor_type`, `other_floor_type`, `position`, `plan_configuration`, `legal_ownership_status`) should be treated as opaque identifiers unless further verified.

### Identifiers
*   `building_id` (int): Unique identifier for each building.

### Location
*   `geo_level_1_id` (int): Geographic region ID, level 1 (largest region, 0-30).
*   `geo_level_2_id` (int): Geographic region ID, level 2 (medium region, 0-1427). *Potentially corresponds to ADM2 shapefile PCODE.*
*   `geo_level_3_id` (int): Geographic region ID, level 3 (most specific sub-region, 0-12567).

### Building Structure & Age
*   `count_floors_pre_eq` (int): Number of floors before the earthquake.
*   `age` (int): Age of the building in years.
*   `area_percentage` (int): Normalized area of the building footprint.
*   `height_percentage` (int): Normalized height of the building.
*   `land_surface_condition` (categorical): Surface condition of the land. (Codes: `n`, `o`, `t` - meaning unknown/obfuscated).
*   `foundation_type` (categorical): Type of foundation.
    *   `h`: Pile Foundation
    *   `i`: Isolated Foundation
    *   `r`: Raft/Mat Foundation
    *   `u`: Under-reamed Foundation
    *   `w`: Well Foundation
*   `roof_type` (categorical): Type of roof. (Codes: `n`, `q`, `x` - meaning unknown/obfuscated).
*   `ground_floor_type` (categorical): Type of ground floor. (Codes: `f`, `m`, `v`, `x`, `z` - meaning unknown/obfuscated).
*   `other_floor_type` (categorical): Type of construction used in upper floors (excluding roof). (Codes: `j`, `q`, `s`, `x` - meaning unknown/obfuscated).
*   `position` (categorical): Position of the building. (Codes: `j`, `o`, `s`, `t` - meaning unknown/obfuscated).
*   `plan_configuration` (categorical): Building plan configuration. (Codes: `a`, `c`, `d`, `f`, `m`, `n`, `o`, `q`, `s`, `u` - meaning unknown/obfuscated).

### Superstructure Materials (Binary Flags)
*   `has_superstructure_adobe_mud` (binary): Superstructure made of Adobe/Mud (1=Yes, 0=No).
*   `has_superstructure_mud_mortar_stone` (binary): Superstructure made of Mud Mortar - Stone.
*   `has_superstructure_stone_flag` (binary): Superstructure made of Stone.
*   `has_superstructure_cement_mortar_stone` (binary): Superstructure made of Cement Mortar - Stone.
*   `has_superstructure_mud_mortar_brick` (binary): Superstructure made of Mud Mortar - Brick.
*   `has_superstructure_cement_mortar_brick` (binary): Superstructure made of Cement Mortar - Brick.
*   `has_superstructure_timber` (binary): Superstructure made of Timber.
*   `has_superstructure_bamboo` (binary): Superstructure made of Bamboo.
*   `has_superstructure_rc_non_engineered` (binary): Superstructure made of non-engineered reinforced concrete.
*   `has_superstructure_rc_engineered` (binary): Superstructure made of engineered reinforced concrete.
*   `has_superstructure_other` (binary): Superstructure made of other materials.

### Ownership & Usage
*   `legal_ownership_status` (categorical): Legal ownership status. (Codes: `a`, `r`, `v`, `w` - meaning unknown/obfuscated).
*   `count_families` (int): Number of families living in the building.
*   `has_secondary_use` (binary): Building has secondary use (1=Yes, 0=No).
*   `has_secondary_use_agriculture` (binary): Secondary use: Agriculture.
*   `has_secondary_use_hotel` (binary): Secondary use: Hotel.
*   `has_secondary_use_rental` (binary): Secondary use: Rental.
*   `has_secondary_use_institution` (binary): Secondary use: Institution.
*   `has_secondary_use_school` (binary): Secondary use: School.
*   `has_secondary_use_industry` (binary): Secondary use: Industry.
*   `has_secondary_use_health_post` (binary): Secondary use: Health Post.
*   `has_secondary_use_gov_office` (binary): Secondary use: Government Office.
*   `has_secondary_use_use_police` (binary): Secondary use: Police Station.
*   `has_secondary_use_other` (binary): Other secondary use.

### Added Earthquake Features (Simplified)
*   `main_eq_magnitude` (float): Magnitude of the main 2015 Gorkha shock (applied uniformly).
*   `main_eq_depth` (float): Depth of the main 2015 Gorkha shock (applied uniformly).
*   `main_eq_epicenter_lat` (float): Latitude of the main 2015 Gorkha shock epicenter (applied uniformly).
*   `main_eq_epicenter_lon` (float): Longitude of the main 2015 Gorkha shock epicenter (applied uniformly).

### Target Variable
*   `damage_grade` (int): Damage grade assigned after the earthquake.
    *   `1`: Low damage
    *   `2`: Medium damage
    *   `3`: Complete destruction / Collapse 