import os
import sys
import logging
import pandas as pd
import geopandas as gpd
import rioxarray
from rasterstats import zonal_stats
import xarray as xr  # Keep xarray import for potential future use or type hints

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src directory to sys.path to allow for utils import
# Adjust the path depth as needed based on where the script is run from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Assuming utils might contain path helpers or constants eventually
# from utils import config # Example if you create a config utility

# Define file paths (adjust if your structure differs)
# Consider moving these to a config file/module later
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
# SRTM_IMG_PATH = os.path.join(RAW_DATA_DIR, 'srtm_raw', 'srtm_cgiar_nepal_boundary.img') # Old path
SRTM_IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'srtm_raw', 'srtm_cgiar_nepal_boundary.img') # Corrected path
# Use the most granular shapefile available for zonal stats
# SHAPEFILE_PATH = os.path.join(RAW_DATA_DIR, 'npl_adm_nd_20240314_ab_shp', 'npl_admbnda_adm3_nd_20240314.shp') # Old path
SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'npl_adm_nd_20240314_ab_shp', 'npl_admbnda_adm3_nd_20240314.shp') # Corrected path
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'terrain_features_by_adm3.csv')

def calculate_terrain_attributes(dem_path):
    """
    Calculates slope and aspect from a DEM using rioxarray and xrspatial.
    Note: Requires xrspatial to be installed. Add 'xrspatial' to requirements.txt if needed.

    Args:
        dem_path (str): Path to the DEM file.

    Returns:
        tuple: (slope_da, aspect_da) xarray.DataArrays for slope and aspect.
               Returns (None, None) if xrspatial is not installed or calculation fails.
    """
    try:
        import xrspatial.multisurface as ms
    except ImportError:
        logging.error("xrspatial is required for slope/aspect calculation. Please install it (`pip install xrspatial`) and add it to requirements.txt.")
        return None, None

    logging.info(f"Loading DEM from {dem_path}...")
    try:
        dem_da = rioxarray.open_rasterio(dem_path, masked=True).squeeze() # squeeze removes band dim if single band
        dem_da.name = "elevation"

        # Ensure CRS is set for calculations
        if dem_da.rio.crs is None:
            logging.warning("DEM CRS is not set. Assuming EPSG:4326 for calculations. Reproject if necessary.")
            # Attempt to set a common geographic CRS if missing, adjust if needed
            # dem_da = dem_da.rio.set_crs("EPSG:4326") # Uncomment and adjust if needed

        logging.info("Calculating slope...")
        slope_da = ms.slope(dem_da)
        slope_da.name = "slope"

        logging.info("Calculating aspect...")
        aspect_da = ms.aspect(dem_da)
        aspect_da.name = "aspect"

        return slope_da, aspect_da

    except Exception as e:
        logging.error(f"Error calculating terrain attributes: {e}")
        return None, None


def main():
    logging.info("Starting terrain feature processing...")

    # 1. Load Shapefile
    logging.info(f"Loading shapefile from {SHAPEFILE_PATH}...")
    try:
        admin_gdf = gpd.read_file(SHAPEFILE_PATH)
        # Ensure shapefile has a CRS, reproject if necessary to match raster later
        if admin_gdf.crs is None:
             logging.warning(f"Shapefile {SHAPEFILE_PATH} has no CRS defined. Assuming EPSG:4326.")
             admin_gdf = admin_gdf.set_crs("EPSG:4326") # Adjust if needed

        # Check for required columns (adjust PCODE column name if different)
        pcode_col = 'ADM3_PCODE' # Double check this column name in your shapefile
        if pcode_col not in admin_gdf.columns:
             alt_pcode_cols = ['PCODE', 'ADM_PCODE', 'geo_level_3_id'] # Common alternatives
             found = False
             for col in alt_pcode_cols:
                 if col in admin_gdf.columns:
                     pcode_col = col
                     logging.warning(f"Using '{pcode_col}' as the PCODE column.")
                     found = True
                     break
             if not found:
                logging.error(f"Could not find a suitable PCODE column (tried {pcode_col}, {alt_pcode_cols}) in {SHAPEFILE_PATH}. Aborting.")
                sys.exit(1)

        logging.info(f"Shapefile loaded successfully. Found {len(admin_gdf)} features. Using PCODE column: {pcode_col}")

    except Exception as e:
        logging.error(f"Failed to load shapefile {SHAPEFILE_PATH}: {e}")
        sys.exit(1)

    # 2. Calculate Slope and Aspect (Optional but Recommended)
    # Note: Slope/Aspect calculation might be memory intensive
    slope_da, aspect_da = calculate_terrain_attributes(SRTM_IMG_PATH)

    # 3. Perform Zonal Statistics for Elevation
    logging.info("Calculating zonal statistics for Elevation...")
    try:
        # Ensure vector and raster CRS match before zonal_stats
        # It's often better to reproject the vector to match the raster
        with rioxarray.open_rasterio(SRTM_IMG_PATH) as dem_src:
            target_crs = dem_src.rio.crs
            logging.info(f"Raster CRS: {target_crs}. Shapefile CRS: {admin_gdf.crs}")
            if admin_gdf.crs != target_crs:
                logging.info(f"Reprojecting shapefile to match raster CRS ({target_crs})...")
                admin_gdf = admin_gdf.to_crs(target_crs)
                logging.info("Shapefile reprojected.")

        # Use the file path directly with rasterstats
        elevation_stats = zonal_stats(admin_gdf,
                                      SRTM_IMG_PATH,
                                      stats=['mean', 'median', 'std', 'min', 'max'], # Add more stats if needed
                                      prefix='elev_',
                                      geojson_out=False, # We'll merge back to the GeoDataFrame
                                      nodata=-9999) # Check appropriate nodata value for SRTM
        elevation_df = pd.DataFrame(elevation_stats)
        logging.info("Elevation zonal statistics calculated.")
    except Exception as e:
        logging.error(f"Failed during elevation zonal statistics: {e}")
        # Continue without elevation if it fails, or exit depending on requirements
        elevation_df = pd.DataFrame() # Create empty df

    # 4. Perform Zonal Statistics for Slope (if calculated)
    slope_df = pd.DataFrame()
    if slope_da is not None:
        logging.info("Calculating zonal statistics for Slope...")
        try:
             # Workaround: rasterstats needs a file path or rasterio dataset handle.
             # Save temp slope raster or use alternative zonal stats method if needed.
             # For now, let's try writing slope_da to a temporary file.
             TEMP_SLOPE_PATH = os.path.join(PROCESSED_DATA_DIR, 'temp_slope.tif')
             slope_da.rio.to_raster(TEMP_SLOPE_PATH)

             slope_stats = zonal_stats(admin_gdf,
                                      TEMP_SLOPE_PATH,
                                      stats=['mean', 'median', 'std'],
                                      prefix='slope_',
                                      geojson_out=False,
                                      nodata=-9999) # Adjust nodata if needed
             slope_df = pd.DataFrame(slope_stats)
             os.remove(TEMP_SLOPE_PATH) # Clean up temp file
             logging.info("Slope zonal statistics calculated.")
        except Exception as e:
            logging.error(f"Failed during slope zonal statistics: {e}")
            if os.path.exists(TEMP_SLOPE_PATH):
                os.remove(TEMP_SLOPE_PATH)

    # 5. Perform Zonal Statistics for Aspect (if calculated)
    # Aspect is circular (0-360 degrees), simple mean isn't always meaningful.
    # Consider calculating stats like circular mean/std or proportion of N/S/E/W facing slopes if needed.
    # For now, we'll calculate the mean as a basic measure.
    aspect_df = pd.DataFrame()
    if aspect_da is not None:
        logging.info("Calculating zonal statistics for Aspect...")
        try:
             TEMP_ASPECT_PATH = os.path.join(PROCESSED_DATA_DIR, 'temp_aspect.tif')
             aspect_da.rio.to_raster(TEMP_ASPECT_PATH)
             aspect_stats = zonal_stats(admin_gdf,
                                      TEMP_ASPECT_PATH,
                                      stats=['mean', 'median', 'std'], # Mean aspect might be less useful
                                      prefix='aspect_',
                                      geojson_out=False,
                                      nodata=-9999) # Adjust nodata if needed
             aspect_df = pd.DataFrame(aspect_stats)
             os.remove(TEMP_ASPECT_PATH) # Clean up temp file
             logging.info("Aspect zonal statistics calculated.")
        except Exception as e:
            logging.error(f"Failed during aspect zonal statistics: {e}")
            if os.path.exists(TEMP_ASPECT_PATH):
                os.remove(TEMP_ASPECT_PATH)


    # 6. Merge results
    logging.info("Merging terrain features with administrative boundaries...")
    # Start with PCODEs from the original shapefile
    result_df = admin_gdf[[pcode_col]].copy()

    # Add stats DataFrames if they are not empty
    if not elevation_df.empty:
        result_df = pd.concat([result_df, elevation_df], axis=1)
    if not slope_df.empty:
        result_df = pd.concat([result_df, slope_df], axis=1)
    if not aspect_df.empty:
        result_df = pd.concat([result_df, aspect_df], axis=1)

    # Remove geometry column if it accidentally got added back during concat/merge
    if 'geometry' in result_df.columns:
        result_df = result_df.drop(columns=['geometry'])


    # 7. Save output
    logging.info(f"Saving aggregated terrain features to {OUTPUT_CSV_PATH}...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info("Terrain feature processing completed successfully.")


if __name__ == "__main__":
    # Add xrspatial to requirements if using slope/aspect
    logging.warning("Ensure 'xrspatial', 'rasterio', 'rioxarray', 'rasterstats', 'geopandas' are in requirements.txt and installed.")
    main() 