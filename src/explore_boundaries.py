import pandas as pd
import geopandas as gpd
import sys
import numpy as np

# Define file paths
BUILDING_VALUES_FILE = 'data/nepal_land_data/train_values.csv'
ADM3_SHAPEFILE = 'data/npl_adm_nd_20240314_ab_shp/npl_admbnda_adm3_nd_20240314.shp'
LOOKUP_FILE = 'data/npl_adm_nd_20240314_ab_shp/npl_admbndt_adminUnitLookup.dbf'
SAMPLE_SIZE = 1000  # Load a small sample of the large building data file

def explore_ids():
    """Loads building data and admin boundaries to compare ID columns."""
    print("--- Loading Building Data Sample ---")
    try:
        # Load only a sample and specific columns initially to save memory
        cols_to_load = ['building_id', 'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
        building_df = pd.read_csv(BUILDING_VALUES_FILE, usecols=cols_to_load, nrows=SAMPLE_SIZE)
        print(f"Loaded {len(building_df)} rows from {BUILDING_VALUES_FILE}")
        print("Building Data Columns:", building_df.columns.tolist())
        print("Building Data Head:\n", building_df.head())
        print("\nGeo Level ID Value Counts (Sample):")
        print("- Geo Level 1:", building_df['geo_level_1_id'].nunique(), "unique values. Example:", building_df['geo_level_1_id'].unique()[:5])
        print("- Geo Level 2:", building_df['geo_level_2_id'].nunique(), "unique values. Example:", building_df['geo_level_2_id'].unique()[:5])
        print("- Geo Level 3:", building_df['geo_level_3_id'].nunique(), "unique values. Example:", building_df['geo_level_3_id'].unique()[:5])
        print("\nBuilding Data Info:")
        building_df.info()
    except FileNotFoundError:
        print(f"Error: Building data file not found at {BUILDING_VALUES_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading building data: {e}")
        sys.exit(1)
        
    print("\n--- Loading ADM3 Boundary Data ---")
    try:
        admin_gdf = gpd.read_file(ADM3_SHAPEFILE)
        print(f"Loaded {len(admin_gdf)} features from {ADM3_SHAPEFILE}")
        print("Admin Boundary Columns:", admin_gdf.columns.tolist())
        print("Admin Boundary Data Head (showing relevant columns):\n", admin_gdf[['ADM3_EN', 'ADM3_PCODE', 'ADM2_PCODE', 'ADM1_PCODE', 'ADM0_PCODE']].head())
        print("\nAdmin Boundary PCODE Value Counts:")
        print("- ADM3 PCODE:", admin_gdf['ADM3_PCODE'].nunique(), "unique values. Example:", admin_gdf['ADM3_PCODE'].unique()[:5])
        print("- ADM2 PCODE:", admin_gdf['ADM2_PCODE'].nunique(), "unique values. Example:", admin_gdf['ADM2_PCODE'].unique()[:5])
        print("- ADM1 PCODE:", admin_gdf['ADM1_PCODE'].nunique(), "unique values. Example:", admin_gdf['ADM1_PCODE'].unique()[:5])
        print("\nAdmin Boundary Info (excluding geometry):")
        admin_gdf.drop(columns='geometry').info()
        
    except FileNotFoundError:
        print(f"Error: Shapefile not found at {ADM3_SHAPEFILE}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        sys.exit(1)
        
    print("\n--- Loading Admin Unit Lookup Table ---")
    lookup_df = None
    try:
        # Try reading with geopandas (which uses fiona, might handle dbf)
        lookup_df = gpd.read_file(LOOKUP_FILE)
        print(f"Successfully loaded lookup table using geopandas: {LOOKUP_FILE}")
    except Exception as e_gpd:
        print(f"Failed to load lookup DBF with geopandas: {e_gpd}")
        # Try reading directly with pandas (might work for simple DBF)
        # Note: This might require additional libraries like `simpledbf` or `dbfread` if pandas fails
        try:
            lookup_df = pd.read_csv(LOOKUP_FILE) # Pandas doesn't directly support DBF
            # We would need a library like simpledbf: pip install simpledbf
            # from simpledbf import Dbf5
            # dbf = Dbf5(LOOKUP_FILE)
            # lookup_df = dbf.to_dataframe()
            print("Note: Reading DBF with pandas directly is not supported. Need specific library.")
            # For now, we'll just report failure if geopandas didn't work.
            print(f"Could not load {LOOKUP_FILE}. Manual inspection or different library might be needed.")
            sys.exit(1)
        except Exception as e_pd:
             print(f"Failed to load lookup DBF with pandas attempt: {e_pd}")
             print(f"Could not load {LOOKUP_FILE}. Manual inspection or different library might be needed.")
             sys.exit(1)
             
    if lookup_df is not None:
        print(f"Loaded {len(lookup_df)} rows from {LOOKUP_FILE}")
        print("Lookup Table Columns:", lookup_df.columns.tolist())
        print("Lookup Table Head:\n", lookup_df.head())
        print("\nLookup Table Info:")
        lookup_df.info()
        
        # --- Check if lookup table helps link IDs ---
        print("\n--- Checking Link via Lookup Table ---")
        # Check if lookup table contains both PCODEs and potential numeric IDs
        has_pcode = any('PCODE' in col.upper() for col in lookup_df.columns)
        # Look for columns that might match geo_level_id format (numeric)
        numeric_cols = lookup_df.select_dtypes(include=np.number).columns
        has_potential_geo_id = len(numeric_cols) > 0
        
        print(f"Lookup table contains PCODE-like columns: {has_pcode}")
        print(f"Lookup table contains numeric columns: {has_potential_geo_id}")
        if has_potential_geo_id:
             print(f"Potential numeric ID columns: {numeric_cols.tolist()}")
             
        # Example check: Does it have ADM3_PCODE and a numeric column?
        if 'ADM3_PCODE' in lookup_df.columns and has_potential_geo_id:
            print("Lookup table seems promising for linking ADM3_PCODE to a numeric ID.")
            # Further analysis would involve checking if the numeric column values
            # match the range/distribution of geo_level_3_id
        else:
            print("Lookup table might not contain the direct link we need based on column names/types.")

    print("\n--- Comparison --- ")
    # Check data types
    print(f"Building geo_level_3_id dtype: {building_df['geo_level_3_id'].dtype}")
    print(f"Admin ADM3_PCODE dtype: {admin_gdf['ADM3_PCODE'].dtype}")
    
    # If admin pcode is string and building id is int, we might need conversion for joining.
    # Let's check if the PCODEs look like numbers that could match the geo_level_ids
    print("\nComparing example values:")
    print(f"Building geo_level_3_id examples: {building_df['geo_level_3_id'].unique()[:10]}")
    print(f"Admin ADM3_PCODE examples: {admin_gdf['ADM3_PCODE'].unique()[:10]}")
    
    # A simple check if any IDs match directly (might not if types differ or pcodes have prefixes/suffixes)
    try:
        match_check = building_df['geo_level_3_id'].isin(admin_gdf['ADM3_PCODE'].astype(int) if admin_gdf['ADM3_PCODE'].dtype == 'object' else admin_gdf['ADM3_PCODE'])
        print(f"\nDirect match check (geo_level_3_id in ADM3_PCODE): {match_check.sum()} matches found in the sample of {SAMPLE_SIZE}.")
    except Exception as e:
        print(f"\nCould not perform direct match check (likely due to type mismatch or non-numeric PCODEs): {e}")

if __name__ == "__main__":
    explore_ids() 