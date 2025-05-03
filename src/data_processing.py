import pandas as pd
import os
import numpy as np

# Define data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
USGS_FILE = os.path.join(DATA_DIR, 'usgs_himalayan_earthquakes_1900_present_M4.0.csv')
NSC_FILE = os.path.join(DATA_DIR, 'nsc_nepal_earthquakes_1994_present.csv')
DRIVEN_DATA_DIR = os.path.join(DATA_DIR, 'nepal_land_data')
TRAIN_VALUES_FILE = os.path.join(DRIVEN_DATA_DIR, 'train_values.csv')
TRAIN_LABELS_FILE = os.path.join(DRIVEN_DATA_DIR, 'train_labels.csv')
TEST_VALUES_FILE = os.path.join(DRIVEN_DATA_DIR, 'test_values.csv')

def load_clean_usgs_data():
    """Loads and performs initial cleaning on the USGS dataset."""
    print("--- Loading and Cleaning USGS Data ---")
    try:
        usgs_df = pd.read_csv(USGS_FILE)
        print(f"Original USGS Shape: {usgs_df.shape}")

        # Convert time to datetime
        usgs_df['time'] = pd.to_datetime(usgs_df['time'], errors='coerce')
        # Drop rows where time conversion failed
        usgs_df.dropna(subset=['time'], inplace=True)

        # Select and rename columns
        usgs_df = usgs_df[['time', 'latitude', 'longitude', 'depth', 'mag', 'magType', 'id', 'place']].copy()
        usgs_df.rename(columns={'id': 'event_id'}, inplace=True)

        print(f"Cleaned USGS Shape: {usgs_df.shape}")
        print("Cleaned USGS Info:")
        usgs_df.info()
        print("\nCleaned USGS First 5 Rows:")
        print(usgs_df.head())
        return usgs_df

    except FileNotFoundError:
        print(f"Error: USGS file not found at {USGS_FILE}")
        return None
    except Exception as e:
        print(f"Error processing USGS data: {e}")
        return None

# --- Placeholder for NSC Cleaning Function ---
def load_clean_nsc_data():
    """Loads and performs initial cleaning on the NSC dataset."""
    print("\n--- Loading and Cleaning NSC Data ---")
    try:
        nsc_df = pd.read_csv(NSC_FILE)
        print(f"Original NSC Shape: {nsc_df.shape}")

        # --- Combine Date and Time ---
        # Fill missing Time with '00:00:00'
        nsc_df['Time'].fillna('00:00:00', inplace=True)
        # Combine A.D. date and Time
        # Assuming A.D. is YYYY-MM-DD. Errors will coerce to NaT
        nsc_df['datetime_str'] = nsc_df['A.D.'] + ' ' + nsc_df['Time']
        nsc_df['time'] = pd.to_datetime(nsc_df['datetime_str'], errors='coerce')

        # Drop rows where time conversion failed
        nsc_df.dropna(subset=['time'], inplace=True)

        # --- Make time column timezone-aware (assume UTC) ---
        try:
            # Localize to UTC. If already localized (e.g., due to future pandas changes),
            # this might raise an error, hence the try-except.
            nsc_df['time'] = nsc_df['time'].dt.tz_localize('UTC')
        except TypeError:
             # If already tz-aware, attempt conversion just in case it's not UTC
            try:
                nsc_df['time'] = nsc_df['time'].dt.tz_convert('UTC')
            except TypeError as te:
                print(f"Warning: Could not convert NSC time column to UTC: {te}")
                # Keep timezone-naive if conversion fails unexpectedly
                pass

        # --- Rename Columns ---
        nsc_df.rename(columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Magnitude': 'mag',
            'Location': 'place' # Use Location as place
        }, inplace=True)

        # --- Add Missing Standard Columns ---
        nsc_df['depth'] = np.nan # Depth data not available in NSC
        nsc_df['magType'] = 'ML' # Assume ML based on typical regional network magnitudes
        nsc_df['event_id'] = 'NSC_' + nsc_df.index.astype(str)

        # Fill missing place values
        nsc_df['place'].fillna('Unknown', inplace=True)

        # --- Select Final Standard Columns ---
        standard_cols = ['time', 'latitude', 'longitude', 'depth', 'mag', 'magType', 'event_id', 'place']
        nsc_clean_df = nsc_df[standard_cols].copy()

        print(f"Cleaned NSC Shape: {nsc_clean_df.shape}")
        print("Cleaned NSC Info:")
        nsc_clean_df.info()
        print("\nCleaned NSC First 5 Rows:")
        print(nsc_clean_df.head())
        return nsc_clean_df

    except FileNotFoundError:
        print(f"Error: NSC file not found at {NSC_FILE}")
        return None
    except Exception as e:
        print(f"Error loading NSC data: {e}")
        return None

# --- Placeholder for DrivenData Loading Functions ---
def load_driven_data():
    """Loads the DrivenData competition files."""
    print("\n--- Loading DrivenData (Building Damage) ---")
    dataframes = {}
    try:
        train_values_df = pd.read_csv(TRAIN_VALUES_FILE, index_col='building_id')
        train_labels_df = pd.read_csv(TRAIN_LABELS_FILE, index_col='building_id')
        test_values_df = pd.read_csv(TEST_VALUES_FILE, index_col='building_id')

        print(f"Train Values Shape: {train_values_df.shape}")
        # train_values_df.info(verbose=False)
        print(f"Train Labels Shape: {train_labels_df.shape}")
        # train_labels_df.info()
        print(f"Test Values Shape: {test_values_df.shape}")
        # test_values_df.info(verbose=False)

        dataframes['train_values'] = train_values_df
        dataframes['train_labels'] = train_labels_df
        dataframes['test_values'] = test_values_df
        return dataframes

    except FileNotFoundError as e:
        print(f"Error: DrivenData file not found. Please check paths.")
        return None
    except Exception as e:
        print(f"Error loading DrivenData: {e}")
        return None


if __name__ == "__main__":
    usgs_clean_df = load_clean_usgs_data()
    nsc_clean_df = load_clean_nsc_data()
    driven_data_dfs = load_driven_data()

    if usgs_clean_df is not None:
        print("\nUSGS Data Loaded Successfully.")

    if nsc_clean_df is not None:
        print("\nNSC Data Loaded Successfully (Cleaned).")

    if driven_data_dfs is not None:
        print("\nDrivenData Loaded Successfully.")

    # --- Combine earthquake data ---
    print("\n--- Combining Earthquake Data ---")
    earthquake_df = None
    if usgs_clean_df is not None and nsc_clean_df is not None:
        # Concatenate directly as both time columns should now be datetime64[ns, UTC]
        earthquake_df = pd.concat([usgs_clean_df, nsc_clean_df], ignore_index=True)

        # Verify the dtype after concatenation
        print("Verifying time column dtype after concat:", earthquake_df['time'].dtype)
        # Drop rows where time might still be NaT after concat (shouldn't happen ideally)
        earthquake_df.dropna(subset=['time'], inplace=True)

        print(f"Combined Earthquake Data Shape: {earthquake_df.shape}")
        print("Combined Earthquake Data Info:")
        earthquake_df.info()
        print("\nCombined Earthquake Data - Value Counts for 'magType':")
        print(earthquake_df['magType'].value_counts())
        print("\nCombined Earthquake Data - Checking for duplicates based on time/lat/lon:")
        duplicate_cols = ['time', 'latitude', 'longitude']
        duplicates = earthquake_df[earthquake_df.duplicated(subset=duplicate_cols, keep=False)]
        print(f"Found {duplicates.shape[0]} potential duplicates based on exact time/lat/lon.")
        if not duplicates.empty:
            print(duplicates.sort_values(by='time').head())

        # Sort by time
        earthquake_df.sort_values(by='time', inplace=True)
        print("\nCombined DataFrame Head (Sorted by Time):")
        print(earthquake_df.head())
        # --- Optional: Save combined data ---
        # combined_file = os.path.join(DATA_DIR, 'combined_earthquakes.csv')
        # print(f"\nSaving combined data to {combined_file}...")
        # earthquake_df.to_csv(combined_file, index=False)
        # print("Save complete.")

        # --- Analyze Combined Data ---
        print("\n--- Analyzing Combined Earthquake Data ---")
        # Missing Values Analysis
        print("Missing Value Percentage:")
        missing_percentage = (earthquake_df.isnull().sum() / len(earthquake_df)) * 100
        print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

        # --- Handle Missing Depth ---
        if 'depth' in earthquake_df.columns and earthquake_df['depth'].isnull().any():
            median_depth = earthquake_df['depth'].median()
            print(f"\nImputing {earthquake_df['depth'].isnull().sum()} missing depth values with median: {median_depth:.2f} km")
            earthquake_df['depth'].fillna(median_depth, inplace=True)
            # Verify imputation
            print(f"Missing depth values after imputation: {earthquake_df['depth'].isnull().sum()}")

        # Descriptive Statistics for Numerical Columns
        print("\nDescriptive Statistics (after depth imputation):")
        print(earthquake_df[['latitude', 'longitude', 'depth', 'mag']].describe())

    elif usgs_clean_df is not None:
        print("Only USGS data available.")
        earthquake_df = usgs_clean_df
    elif nsc_clean_df is not None:
        print("Only NSC data available.")
        earthquake_df = nsc_clean_df
    else:
        print("No earthquake data loaded.")

    # --- Final Summary ---
    print("\nData loading and initial standardization complete.") 