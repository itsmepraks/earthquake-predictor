import pandas as pd
import numpy as np

# Define file paths
USGS_FILE = 'data/usgs_himalayan_earthquakes_1900_present_M4.0.csv'
NSC_FILE = 'data/nsc_nepal_earthquakes_1994_present.csv'
OUTPUT_FILE = 'data/processed/earthquakes_categorized.csv'
BUILDING_VALUES_FILE = 'data/nepal_land_data/train_values.csv'
BUILDING_LABELS_FILE = 'data/nepal_land_data/train_labels.csv'
FINAL_OUTPUT_FILE = 'data/processed/buildings_features_earthquakes.csv'

def load_and_preprocess_earthquakes():
    """Loads, merges, and preprocesses USGS and NSC earthquake data."""
    # Load datasets
    try:
        usgs_df = pd.read_csv(USGS_FILE)
        nsc_df = pd.read_csv(NSC_FILE)
        print(f"Loaded USGS data: {usgs_df.shape}")
        print(f"Loaded NSC data: {nsc_df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None

    # --- Schema Standardization ---
    # USGS: time, latitude, longitude, depth, mag, place
    # NSC: B.S.,A.D.,Time,Latitude,Longitude,Magnitude,Recorded Centre,Location
    
    # Rename columns for consistency
    usgs_df.rename(columns={
        'time': 'datetime',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'depth': 'depth',
        'mag': 'magnitude',
        'place': 'region'
    }, inplace=True)
    
    nsc_df.rename(columns={
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Magnitude': 'magnitude',
        'Location': 'region'
    }, inplace=True)

    # Combine NSC date and time columns (handle potential missing times)
    if 'Time' in nsc_df.columns:
        nsc_df['Time'] = nsc_df['Time'].fillna('00:00:00').astype(str)
        nsc_df['Time'] = nsc_df['Time'].apply(lambda x: x if ':' in x and len(x.split(':')) == 3 else (x + ':00' if ':' in x and len(x.split(':')) == 2 else '00:00:00'))
        nsc_df['datetime_str'] = nsc_df['A.D.'].astype(str) + ' ' + nsc_df['Time']
        nsc_df['datetime'] = pd.to_datetime(nsc_df['datetime_str'], errors='coerce')
    elif 'A.D.' in nsc_df.columns:
        nsc_df['datetime'] = pd.to_datetime(nsc_df['A.D.'], errors='coerce')
    else:
        print("Warning: Could not find 'A.D.' or 'Time' columns in NSC data to create datetime.")
        nsc_df['datetime'] = pd.NaT

    # Convert USGS datetime string to datetime objects (handle potential errors)
    usgs_df['datetime'] = pd.to_datetime(usgs_df['datetime'], errors='coerce')
    # Ensure USGS datetime is UTC (it often is, but make explicit)
    if usgs_df['datetime'].dt.tz is None:
        # If tz-naive, assume UTC and localize
        usgs_df['datetime'] = usgs_df['datetime'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
    else:
        # If tz-aware, convert to UTC
        usgs_df['datetime'] = usgs_df['datetime'].dt.tz_convert('UTC')

    # --- Handle NSC Datetime Timezone ---
    if 'datetime' in nsc_df.columns and pd.api.types.is_datetime64_any_dtype(nsc_df['datetime']):
        # Assuming NSC times are Nepal Time (NPT, UTC+5:45)
        try:
            # Localize as Nepal time, then convert to UTC
            nsc_df['datetime'] = nsc_df['datetime'].dt.tz_localize('Asia/Kathmandu', ambiguous='NaT', nonexistent='NaT')
            nsc_df['datetime'] = nsc_df['datetime'].dt.tz_convert('UTC')
            print("Localized NSC datetime to NPT and converted to UTC.")
        except Exception as e:
            print(f"Warning: Could not localize/convert NSC timezone: {e}")
            # Fallback: make naive to allow concatenation, might lose precision
            # nsc_df['datetime'] = nsc_df['datetime'].dt.tz_localize(None)
            # Or handle rows with conversion errors separately
            pass # Keep as NaT if localization fails severely
    # --- End Timezone Handling ---

    # Calculate median depth from USGS data *before* merging
    median_usgs_depth = usgs_df['depth'].median()
    print(f"Median depth from USGS data: {median_usgs_depth:.2f} km")

    # Add depth column to NSC data using the median USGS depth
    nsc_df['depth'] = median_usgs_depth

    # Select and order columns
    common_cols = ['datetime', 'latitude', 'longitude', 'depth', 'magnitude', 'region']
    nsc_processed = nsc_df[common_cols].copy()
    nsc_processed['source'] = 'NSC'
    
    usgs_cols_to_keep = common_cols + ['magType', 'nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource']
    usgs_existing_cols = [col for col in usgs_cols_to_keep if col in usgs_df.columns]
    usgs_processed = usgs_df[usgs_existing_cols].copy()
    usgs_processed['source'] = 'USGS'

    # Combine dataframes
    combined_df = pd.concat([usgs_processed, nsc_processed], ignore_index=True, sort=False)

    # Drop rows with NaT datetimes (from parsing errors)
    initial_rows = len(combined_df)
    combined_df.dropna(subset=['datetime'], inplace=True)
    if initial_rows > len(combined_df):
        print(f"Dropped {initial_rows - len(combined_df)} rows with invalid datetime formats.")

    # Sort by datetime
    combined_df.sort_values(by='datetime', inplace=True)

    # Remove duplicates (keeping the first occurrence - potentially USGS if time matches exactly)
    initial_rows = len(combined_df)
    combined_df.drop_duplicates(subset=['datetime', 'latitude', 'longitude', 'magnitude'], keep='first', inplace=True)
    if initial_rows > len(combined_df):
        print(f"Dropped {initial_rows - len(combined_df)} duplicate earthquake records.")

    print(f"Combined data shape before depth handling (NSC now has imputed depth): {combined_df.shape}")

    # --- Handle Missing Values (Depth) ---
    # Check if any depths are still missing (primarily from USGS data) and fill with overall median
    if combined_df['depth'].isnull().any():
        overall_median_depth = combined_df['depth'].median()
        missing_depth_count = combined_df['depth'].isnull().sum()
        combined_df['depth'].fillna(overall_median_depth, inplace=True)
        print(f"Filled {missing_depth_count} missing USGS depth values with overall median: {overall_median_depth:.2f} km")
    else:
        print("No missing depth values found after initial processing.")

    print(f"Processed data shape: {combined_df.shape}")
    return combined_df

def categorize_magnitude(df):
    """Adds a magnitude category column to the DataFrame."""
    if 'magnitude' not in df.columns:
        print("Error: 'magnitude' column not found.")
        return df
        
    conditions = [
        (df['magnitude'] < 5.0),
        (df['magnitude'] >= 5.0) & (df['magnitude'] < 6.0),
        (df['magnitude'] >= 6.0) & (df['magnitude'] < 7.0),
        (df['magnitude'] >= 7.0) & (df['magnitude'] < 8.0),
        (df['magnitude'] >= 8.0)
    ]
    categories = ['Minor', 'Moderate', 'Strong', 'Major', 'Great']
    
    df['magnitude_category'] = np.select(conditions, categories, default='Unknown')
    print("Added 'magnitude_category' column.")
    return df

def find_main_gorkha_event(earthquake_df):
    """Identifies the main 2015 Gorkha earthquake event."""
    print("\n--- Identifying Main Gorkha Earthquake Event (April 2015) ---")
    # Filter for April-May 2015
    start_date = '2015-04-24'
    end_date = '2015-05-31'
    gorkha_period_df = earthquake_df[
        (earthquake_df['datetime'] >= pd.Timestamp(start_date, tz='UTC')) &
        (earthquake_df['datetime'] <= pd.Timestamp(end_date, tz='UTC'))
    ].copy()
    
    if gorkha_period_df.empty:
        print("Warning: No earthquakes found in the April-May 2015 period.")
        return None
        
    # Find the event with the largest magnitude in that period
    main_event = gorkha_period_df.loc[gorkha_period_df['magnitude'].idxmax()]
    
    print("Identified Main Event:")
    print(main_event[['datetime', 'latitude', 'longitude', 'depth', 'magnitude', 'source']])
    return main_event

def combine_building_earthquake_features(building_df, main_earthquake_event):
    """Adds main earthquake features to the building dataframe."""
    print("\n--- Adding Main Earthquake Features to Building Data ---")
    if main_earthquake_event is None:
        print("Skipping feature addition as main event was not found.")
        return building_df
        
    # Add features from the main event to every building row
    # Prefixing with 'main_eq_' to avoid potential column name clashes
    building_df['main_eq_magnitude'] = main_earthquake_event['magnitude']
    building_df['main_eq_depth'] = main_earthquake_event['depth']
    building_df['main_eq_epicenter_lat'] = main_earthquake_event['latitude']
    building_df['main_eq_epicenter_lon'] = main_earthquake_event['longitude']
    
    print("Added main earthquake features (magnitude, depth, epicenter lat/lon).")
    return building_df

if __name__ == "__main__":
    print("Starting earthquake data preprocessing and feature engineering...")
    earthquake_data = load_and_preprocess_earthquakes()
    
    if earthquake_data is not None:
        earthquake_data = categorize_magnitude(earthquake_data)
        # Save the processed earthquake data (as done before)
        try:
            earthquake_data.to_csv(OUTPUT_FILE, index=False)
            print(f"Successfully saved categorized earthquake data to {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving earthquake data to {OUTPUT_FILE}: {e}")

        # Find the main Gorkha event
        main_event = find_main_gorkha_event(earthquake_data)
        
        # Load Building Data
        print("\n--- Loading Building Data ---")
        try:
            # Load the full building values and labels datasets
            building_values_df = pd.read_csv(BUILDING_VALUES_FILE, index_col='building_id')
            building_labels_df = pd.read_csv(BUILDING_LABELS_FILE, index_col='building_id')
            print(f"Loaded building values: {building_values_df.shape}")
            print(f"Loaded building labels: {building_labels_df.shape}")
            
            # Merge values and labels
            building_df = building_values_df.join(building_labels_df)
            print(f"Merged building data shape: {building_df.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading building data file: {e}")
            building_df = None
        except Exception as e:
            print(f"Error processing building data: {e}")
            building_df = None
            
        if building_df is not None and main_event is not None:
            # Combine features
            final_df = combine_building_earthquake_features(building_df, main_event)
            
            # Save the final combined dataset
            print("\n--- Saving Combined Feature Data ---")
            try:
                final_df.to_csv(FINAL_OUTPUT_FILE)
                print(f"Successfully saved combined data to {FINAL_OUTPUT_FILE}")
            except Exception as e:
                print(f"Error saving final data to {FINAL_OUTPUT_FILE}: {e}")
                
            print("\nFirst 5 rows of final data:")
            print(final_df.head())
        else:
             print("\nSkipping final data combination and saving due to errors in previous steps.")

    else:
        print("Preprocessing failed. Exiting.") 