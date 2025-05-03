import requests
import os

# Define URL for the raw CSV file on GitHub
RAW_CSV_URL = "https://raw.githubusercontent.com/amitness/earthquakes/master/earthquakes.csv"

# Define output directory and file path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_FILENAME = "nsc_nepal_earthquakes_1994_present.csv"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def download_nsc_data():
    """Downloads the cleaned NSC earthquake data CSV from the amitness/earthquakes GitHub repo."""
    print(f"Attempting to download NSC earthquake data from {RAW_CSV_URL}...")

    try:
        response = requests.get(RAW_CSV_URL, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        print(f"Request successful. Status code: {response.status_code}")

        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Saving data to {OUTPUT_FILE_PATH}...")
        with open(OUTPUT_FILE_PATH, 'wb') as f:
            f.write(response.content)
        print("NSC Data successfully downloaded and saved.")

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
    except IOError as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    download_nsc_data() 