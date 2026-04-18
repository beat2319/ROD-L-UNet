import geopandas as gpd
import pandas as pd
import os
import re
from datetime import datetime

# Create output directory if it doesn't exist
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# Island-specific configuration (same as negative.py)
ISLAND_CONFIG = {
    "Hawaii": {
        "skip_dates": ["05/31/2017", "03/12/2019", "07/03/2019", "12/18/2020", "12/17/2021"],
        "date_buffer_months": 2
    },
    "Lanai": {
        "skip_dates": ["08/16/2019"],
        "date_buffer_months": 2
    },
    "Maui": {
        "skip_dates": ["03/22/2016"],
        "date_buffer_months": 2,
        "year_2019_allowed": ["01/09/2019", "02/08/2019"]
    },
    "Molokai": {
        "skip_dates": [],
        "date_buffer_months": 2
    }
}


def get_year_from_date_str(date_str: str) -> int:
    """Extract year from MM/DD/YYYY date string."""
    return int(date_str.split('/')[-1])


def should_skip_flightline(island: str, year: int, start_date_str: str) -> bool:
    """Returns True if the flightline group should be skipped based on island config."""
    if island not in ISLAND_CONFIG:
        return True  # Skip islands not in config

    config = ISLAND_CONFIG[island]

    # Check standard skip dates
    if start_date_str in config.get("skip_dates", []):
        return True

    # Special Maui 2019 rule
    if island == "Maui" and year == 2019:
        allowed = config.get("year_2019_allowed", [])
        return start_date_str not in allowed

    return False


def get_mortality_date_range(island: str, year: int, start_date_str: str) -> tuple:
    """
    Returns (start_date, end_date) tuple for filtering ohia mortality.

    Special 2016 Hawaii (multi-date ranges):
      - Flightline 01/12/2016 -> 01/11/2016 - 01/16/2016
      - Flightline 02/24/2016 -> 02/23/2016 - 02/25/2016

    All other cases: EXACT date match only
    """
    # Special handling for Hawaii 2016
    if island == "Hawaii" and year == 2016:
        if start_date_str == "01/12/2016":
            return datetime(2016, 1, 11), datetime(2016, 1, 16)
        elif start_date_str == "02/24/2016":
            return datetime(2016, 2, 23), datetime(2016, 2, 25)

    # All other cases: exact date match (same start and end date)
    flight_date = datetime.strptime(start_date_str, '%m/%d/%Y')
    return flight_date, flight_date


def get_target_pairs_from_tifs():
    """Extract (island, date) pairs from existing negative TIF files."""
    pairs = []
    for f in os.listdir('./attachments/negative'):
        if f.startswith('tcc_negative_') and f.endswith('.tif'):
            # Parse: tcc_negative_{island}_{YYYY-MM-DD}.tif
            match = re.match(r'tcc_negative_(\w+)_(\d{4}-\d{2}-\d{2})\.tif', f)
            if match:
                island, date = match.groups()
                # Convert YYYY-MM-DD to MM/DD/YYYY for consistency
                dt = datetime.strptime(date, '%Y-%m-%d')
                date_str = dt.strftime('%m/%d/%Y')
                pairs.append((island.capitalize(), date_str, dt))
    return sorted(pairs, key=lambda x: (x[0], x[2]))


def main():
    target_pairs = get_target_pairs_from_tifs()
    print(f"Found {len(target_pairs)} target (island, date) pairs from negative TIFs")

    processed_count = 0
    skipped_count = 0

    for island, date_str, dt in target_pairs:
        year = dt.year

        # Check skip rules
        if should_skip_flightline(island, year, date_str):
            print(f"Skipping {island} on {date_str} - excluded by island config")
            skipped_count += 1
            continue

        # Get date range for filtering
        start_date, end_date = get_mortality_date_range(island, year, date_str)

        # Load ohia_mortality file
        mortality_path = f"./attachments/ohia_mortality/ohia_mortality_{island.lower()}_{year}.geojson"
        if not os.path.exists(mortality_path):
            print(f"  Warning: Mortality file not found: {mortality_path}")
            skipped_count += 1
            continue

        ohia_mortality = gpd.read_file(mortality_path)

        # The input geojson already has CRS defined (EPSG:32604 for Hawaii)
        # Ensure output is in EPSG:32604, reproject only if needed
        if ohia_mortality.crs is None:
            # If no CRS is defined, assume EPSG:4326 and reproject to EPSG:32604
            ohia_mortality = ohia_mortality.set_crs(epsg=4326, allow_override=True)
            ohia_mortality = ohia_mortality.to_crs(epsg=32604)
        elif ohia_mortality.crs.to_epsg() != 32604:
            # Reproject if not already in EPSG:32604
            ohia_mortality = ohia_mortality.to_crs(epsg=32604)

        # Filter by date range
        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')

        filtered = ohia_mortality[
            (ohia_mortality['detection_timestamp'] >= start_str) &
            (ohia_mortality['detection_timestamp'] <= end_str)
        ]

        if len(filtered) == 0:
            print(f"  Skipping {island} on {date_str} - no matching mortality records in range {start_str} to {end_str}")
            skipped_count += 1
            continue

        # Keep only required columns
        output_gdf = filtered[['PERCENT_AFFECTED_FACTOR', 'geometry']].copy()
        output_gdf['date'] = dt.strftime('%Y-%m-%d')  # Match TIF date format

        # Save output with CRS preserved
        output_path = os.path.join(output_dir, f"positive_{island.lower()}_{dt.strftime('%Y-%m-%d')}.geojson")
        output_gdf.to_file(output_path, driver='GeoJSON', engine='fiona')

        print(f"  Saved: {output_path} ({len(filtered)} records, date range: {start_str} - {end_str}, CRS: {output_gdf.crs})")
        processed_count += 1

    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")


if __name__ == "__main__":
    main()
