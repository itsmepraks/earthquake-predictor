import os
import sys
import logging
import pandas as pd
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SHAPEFILE_DIR = os.path.join(PROJECT_ROOT, 'data', 'npl_adm_nd_20240314_ab_shp') # Corrected path

# Input file paths
BUILDING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'buildings_features_earthquakes.csv')
TERRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'terrain_features_by_adm3.csv')
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, 'npl_admbnda_adm3_nd_20240314.shp') # ADM3 Level

# Output file path
OUTPUT_MERGED_PATH = os.path.join(PROCESSED_DATA_DIR, 'buildings_features_earthquakes_terrain.csv')

def main():
    logging.info("Starting merge process for terrain features...")

    # 1. Load datasets
    logging.info(f"Loading building data from {BUILDING_DATA_PATH}...")
    try:
        building_df = pd.read_csv(BUILDING_DATA_PATH)
        logging.info(f"Building data loaded: {building_df.shape}")
        logging.info(f"Building data columns: {building_df.columns.tolist()}")
        logging.info(f"Unique geo_level_1_id: {building_df['geo_level_1_id'].nunique()}")
        logging.info(f"Unique geo_level_2_id: {building_df['geo_level_2_id'].nunique()}")
        logging.info(f"Unique geo_level_3_id: {building_df['geo_level_3_id'].nunique()}")

    except Exception as e:
        logging.error(f"Failed to load building data: {e}")
        sys.exit(1)

    logging.info(f"Loading terrain data from {TERRAIN_DATA_PATH}...")
    try:
        terrain_df = pd.read_csv(TERRAIN_DATA_PATH)
        logging.info(f"Terrain data loaded: {terrain_df.shape}")
        logging.info(f"Terrain data columns: {terrain_df.columns.tolist()}")
        # Assuming PCODE column identified correctly in previous script
        pcode_col = 'ADM3_PCODE' # Verify this matches the CSV header
        if pcode_col not in terrain_df.columns:
            logging.error(f"PCODE column '{pcode_col}' not found in terrain data. Columns available: {terrain_df.columns.tolist()}")
            # Attempt fallback if necessary, or exit
            if 'PCODE' in terrain_df.columns:
                 pcode_col = 'PCODE'
                 logging.warning(f"Using fallback PCODE column: {pcode_col}")
            else:
                 sys.exit(1)
        logging.info(f"Unique {pcode_col}: {terrain_df[pcode_col].nunique()}")

    except Exception as e:
        logging.error(f"Failed to load terrain data: {e}")
        sys.exit(1)

    logging.info(f"Loading shapefile attributes from {SHAPEFILE_PATH}...")
    try:
        admin_gdf = gpd.read_file(SHAPEFILE_PATH)
        logging.info(f"Shapefile loaded: {admin_gdf.shape}")
        # Keep only relevant columns (PCODEs, Names)
        keep_cols = [col for col in admin_gdf.columns if 'PCODE' in col or 'EN' in col or 'pcode' in col or '_en' in col]
        if 'geometry' in admin_gdf.columns and 'geometry' not in keep_cols:
             keep_cols.append('geometry') # Keep geometry if needed for spatial joins later

        admin_info_df = admin_gdf[keep_cols].copy()
        # Drop geometry if not needed for this merge step
        if 'geometry' in admin_info_df.columns:
            admin_info_df = admin_info_df.drop(columns=['geometry'])

        logging.info(f"Shapefile attribute columns: {admin_info_df.columns.tolist()}")
        logging.info(f"Unique ADM3_PCODE in shapefile: {admin_info_df['ADM3_PCODE'].nunique()}")
        logging.info(f"Unique ADM2_PCODE in shapefile: {admin_info_df['ADM2_PCODE'].nunique()}")
        logging.info(f"Unique ADM1_PCODE in shapefile: {admin_info_df['ADM1_PCODE'].nunique()}")

    except Exception as e:
        logging.error(f"Failed to load shapefile: {e}")
        sys.exit(1)

    # 2. Merge terrain data with admin names/hierarchy from shapefile
    # This gives us ADM1/ADM2/ADM3 names associated with the terrain stats
    logging.info("Merging terrain stats with shapefile attributes...")
    terrain_merged_df = pd.merge(terrain_df, admin_info_df, on=pcode_col, how='left')
    logging.info(f"Terrain data merged with admin info: {terrain_merged_df.shape}")
    # Check for merge issues (missing PCODEs)
    if terrain_merged_df['ADM1_PCODE'].isnull().any(): # Check a column we expect from shapefile
        logging.warning("Some PCODEs from terrain data did not match shapefile attributes.")


    # 3. Attempt to merge building data with terrain data
    logging.info("Attempting to merge building data with terrain data...")

    # Strategy 1: Direct join on geo_level_3_id == ADM3_PCODE?
    # Very unlikely to work based on previous findings, but check a few values.
    logging.info("Checking if geo_level_3_id matches ADM3_PCODE format...")
    sample_geo3 = building_df['geo_level_3_id'].unique()[:5]
    sample_pcode3 = terrain_merged_df[pcode_col].unique()[:5]
    logging.info(f"Sample geo_level_3_id: {sample_geo3}")
    logging.info(f"Sample ADM3_PCODE: {sample_pcode3}")
    # --> If formats differ vastly, direct join is impossible.

    # Strategy 2: Join using intermediate names or inferred hierarchy?
    # This requires understanding the geo_level_id system better.
    # Example: Can we map geo_level_1_id -> ADM1_EN/PCODE, geo_level_2_id -> ADM2_EN/PCODE?

    # !! Placeholder for Merge Logic !!
    # This is the core challenge. We need to establish the link.
    # For now, let's simulate a merge failure/skip.
    merged_data = None
    logging.warning("Merge strategy between building geo_level_ids and terrain ADM PCODEs is NOT YET IMPLEMENTED.")
    logging.warning("Requires investigation into the meaning/mapping of geo_level_ids.")

    # Example potential merge if geo_level_3_id somehow mapped to ADM3_PCODE
    # merged_data = pd.merge(building_df, terrain_merged_df, left_on='geo_level_3_id', right_on=pcode_col, how='left')

    # Example potential merge if geo_level_2_id mapped to ADM2_PCODE
    # Need to aggregate terrain_merged_df by ADM2_PCODE first
    # terrain_adm2 = terrain_merged_df.groupby('ADM2_PCODE').agg({ 'elev_mean': 'mean', 'slope_mean': 'mean'}).reset_index() # Example aggregation
    # merged_data = pd.merge(building_df, terrain_adm2, left_on='geo_level_2_id', right_on='ADM2_PCODE', how='left')

    if merged_data is not None:
        logging.info(f"Successfully merged data: {merged_data.shape}")

        # Select only necessary terrain columns to add (avoid duplicates)
        terrain_cols_to_add = [col for col in terrain_df.columns if col != pcode_col]
        final_df = merged_data # Simplified for now

        # 4. Save output
        logging.info(f"Saving merged data to {OUTPUT_MERGED_PATH}...")
        final_df.to_csv(OUTPUT_MERGED_PATH, index=False)
        logging.info("Merged data saved successfully.")
    else:
        logging.error("Merge failed or was skipped due to lack of linking strategy.")
        logging.info("Further investigation needed for geo_level_id to PCODE mapping.")

if __name__ == "__main__":
    main() 