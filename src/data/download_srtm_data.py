import requests
import os
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
    print("Error: EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in the .env file.")
    exit(1)

# --- Configuration ---
# Bounding box for Nepal (adjust slightly outward to ensure full coverage)
MIN_LAT = 26.0
MAX_LAT = 31.0
MIN_LON = 80.0
MAX_LON = 89.0

# SRTM GL3 data source URL (3 arc-second resolution)
BASE_URL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/"

# Output directory for raw zipped tiles
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'srtm_raw')
# ---

def get_required_tiles(min_lat, max_lat, min_lon, max_lon):
    """Generates a list of SRTM tile names covering the bounding box."""
    tiles = []
    # Iterate through integer latitude and longitude values covering the box
    for lat in range(math.floor(min_lat), math.ceil(max_lat)):
        for lon in range(math.floor(min_lon), math.ceil(max_lon)):
            # Format latitude (N/S) and longitude (E/W)
            lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            # Construct filename (e.g., N27E085.SRTMGL3.hgt.zip)
            tile_name = f"{lat_str}{lon_str}.SRTMGL3.hgt.zip"
            tiles.append(tile_name)
    return tiles

def download_srtm_tile(tile_name, output_dir, username, password):
    """Downloads a single SRTM tile zip file."""
    tile_url = BASE_URL + tile_name
    output_path = os.path.join(output_dir, tile_name)

    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Tile {tile_name} already exists. Skipping.")
        return True

    print(f"Downloading {tile_name} from {tile_url}...")
    try:
        response = requests.get(tile_url, auth=(username, password), stream=True, timeout=60)
        response.raise_for_status() # Check for HTTP errors

        # Save the file chunk by chunk
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {tile_name}")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Tile {tile_name} not found (404). This might be expected for ocean tiles.")
            return False # Tile doesn't exist
        else:
            print(f"HTTP Error downloading {tile_name}: {e}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {tile_name}: {e}")
        return False
    except IOError as e:
        print(f"Error writing file {output_path}: {e}")
        return False

def download_all_srtm_data():
    """Downloads all required SRTM tiles for the defined Nepal region."""
    print("Starting SRTM data download...")
    required_tiles = get_required_tiles(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    print(f"Identified {len(required_tiles)} potential tiles to download.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving tiles to: {OUTPUT_DIR}")

    successful_downloads = 0
    failed_downloads = []

    for tile in required_tiles:
        if download_srtm_tile(tile, OUTPUT_DIR, EARTHDATA_USERNAME, EARTHDATA_PASSWORD):
            successful_downloads += 1
        else:
            # Only add to failed list if it wasn't a 404 (non-existent tile)
            # Check if the failed file exists to determine if it was a 404 or other error
             output_path_check = os.path.join(OUTPUT_DIR, tile)
             if os.path.exists(output_path_check): # If file exists, it was likely partial/corrupt
                 failed_downloads.append(tile)
             else: # If no file, check if it was expected 404 or genuine error
                 # Re-request head to confirm 404 without downloading again
                 try:
                     head_resp = requests.head(BASE_URL + tile, auth=(EARTHDATA_USERNAME, EARTHDATA_PASSWORD), timeout=10)
                     if head_resp.status_code != 404:
                         failed_downloads.append(tile)
                 except requests.exceptions.RequestException:
                      failed_downloads.append(tile) # Add if HEAD request also fails


    print("\nDownload Summary:")
    print(f"- Successfully processed/downloaded: {successful_downloads} tiles")
    print(f"- Failed/Incomplete downloads: {len(failed_downloads)} tiles")
    if failed_downloads:
        print(f"  Failed tiles: {', '.join(failed_downloads)}")
    print("SRTM download process complete.")

if __name__ == "__main__":
    download_all_srtm_data() 