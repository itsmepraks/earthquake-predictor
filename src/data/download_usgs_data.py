import requests
import datetime
import os

# Define API parameters
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
START_TIME = "1900-01-01T00:00:00"
MIN_LATITUDE = 26.3667
MAX_LATITUDE = 30.45
MIN_LONGITUDE = 80.0667
MAX_LONGITUDE = 88.2
MIN_MAGNITUDE = 4.0
FORMAT = "csv"

# Define output directory and file path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_FILENAME = f"usgs_himalayan_earthquakes_{START_TIME[:4]}_present_M{MIN_MAGNITUDE}.csv"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def download_usgs_data():
    """Downloads earthquake data from the USGS API based on defined parameters."""
    print(f"Constructing API query for earthquakes in Nepal region...")
    params = {
        'format': FORMAT,
        'starttime': START_TIME,
        # 'endtime': Use default (present time)
        'minlatitude': MIN_LATITUDE,
        'maxlatitude': MAX_LATITUDE,
        'minlongitude': MIN_LONGITUDE,
        'maxlongitude': MAX_LONGITUDE,
        'minmagnitude': MIN_MAGNITUDE,
        'orderby': 'time' # Order by time ascending
    }

    try:
        print(f"Sending request to {BASE_URL}...")
        response = requests.get(BASE_URL, params=params, timeout=60) # Increased timeout for potentially large dataset
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        print(f"Request successful. Status code: {response.status_code}")

        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Saving data to {OUTPUT_FILE_PATH}...")
        with open(OUTPUT_FILE_PATH, 'wb') as f:
            f.write(response.content)
        print("Data successfully downloaded and saved.")

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
    except IOError as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    download_usgs_data() 