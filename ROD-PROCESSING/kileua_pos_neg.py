import geopandas as gpd
import pandas as pd
import fiona
import os

# Enable KML Support
fiona.drvsupport.supported_drivers['KML'] = 'rw'

def process_daily_geojsons(bg_path, pos_path, kml_path, output_dir, bg_time_col='START_DATE_STR', pos_time_col='detection_date'):
    print("Loading master datasets...")
    # Load the layers
    bg_gdf = gpd.read_file(bg_path).to_crs(epsg=4326)
    pos_gdf = gpd.read_file(pos_path).to_crs(epsg=4326)
    kml_gdf = gpd.read_file(kml_path, driver='KML').to_crs(epsg=4326)

    # Convert time columns to actual pandas datetime objects
    bg_gdf[bg_time_col] = pd.to_datetime(bg_gdf[bg_time_col])
    pos_gdf[pos_time_col] = pd.to_datetime(pos_gdf[pos_time_col])

    # Extract just the date part (YYYY-MM-DD) for exact matching, stripping away hours/minutes
    bg_gdf['date_only'] = bg_gdf[bg_time_col].dt.date
    pos_gdf['date_only'] = pos_gdf[pos_time_col].dt.date    

    # The specific dates from your handwritten list
    target_dates_str = [
        '2017-01-26', '2017-01-27', '2017-04-26', '2017-05-31',
        '2017-09-01', '2017-09-26', '2017-09-28', '2017-12-05', '2017-12-06'
    ]
    # Convert string dates to pandas date objects for comparison
    target_dates = [pd.to_datetime(d).date() for d in target_dates_str]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for target_date in target_dates:
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n--- Processing {date_str} ---")

        # 1. Temporal Alignment: Filter to just this specific day
        bg_day = bg_gdf[bg_gdf['date_only'] == target_date]
        pos_day = pos_gdf[pos_gdf['date_only'] == target_date]

        # 2. Clip BOTH background and positive layers by the KML boundary
        bg_clipped = gpd.clip(bg_day, kml_gdf) if not bg_day.empty else gpd.GeoDataFrame()
        pos_clipped = gpd.clip(pos_day, kml_gdf) if not pos_day.empty else gpd.GeoDataFrame()

        # 3. Calculate the Negative (Difference)
        if not bg_clipped.empty and not pos_clipped.empty:
            negative_gdf = bg_clipped.overlay(pos_clipped, how='difference')
        elif not bg_clipped.empty and pos_clipped.empty:
            # If there are no positives today, the negative is just the whole background
            negative_gdf = bg_clipped 
        else:
            # If there's no background data today, the negative is empty
            negative_gdf = gpd.GeoDataFrame(columns=bg_gdf.columns, crs=bg_gdf.crs)

        # 4. Clean up datetime columns for GeoJSON export
        for gdf in [pos_clipped, negative_gdf]:
            if not gdf.empty:
                for col in gdf.columns:
                    if pd.api.types.is_datetime64_any_dtype(gdf[col]) or pd.api.types.is_object_dtype(gdf[col]):
                        gdf[col] = gdf[col].astype(str) # Convert dates/objects to strings

        # Truncate column names to 10 chars (Shapefile limit) AND prevent duplicates
        for gdf in [pos_clipped, negative_gdf]:
            if not gdf.empty:
                new_cols = []
                seen = set()
                for col in gdf.columns:
                    # Leave the protected 'geometry' column alone
                    if col == 'geometry':
                        new_cols.append(col)
                        seen.add(col)
                        continue
                        
                    # Truncate to 10 characters
                    new_col = str(col)[:10]
                    
                    # If this name is already taken, append a number while keeping it under 10 chars
                    counter = 1
                    while new_col in seen:
                        suffix = str(counter)
                        # Trim the base name enough to fit the suffix (e.g., base_na_1)
                        new_col = str(col)[:10 - len(suffix)] + suffix
                        counter += 1
                        
                    seen.add(new_col)
                    new_cols.append(new_col)
                    
                gdf.columns = new_cols

        # 5. Export as Shapefiles
        # Notice we changed the extension to .shp
        pos_out = os.path.join(output_dir, f'positive_{date_str}.shp')
        neg_out = os.path.join(output_dir, f'negative_{date_str}.shp')

        if not pos_clipped.empty:
            # Dropping the driver argument defaults to 'ESRI Shapefile'
            pos_clipped.to_file(pos_out) 
            print(f"Saved {pos_out}")
        else:
            print(f"No positive data found inside KML for {date_str}, skipping positive export.")

        if not negative_gdf.empty:
            negative_gdf.to_file(neg_out)
            print(f"Saved {neg_out}")
        else:
            print(f"No background data found inside KML for {date_str}, skipping negative export.")

# --- Run the function ---
if __name__ == "__main__":
    process_daily_geojsons(
        bg_path='../ROD-PROCESSING/attachments/2017_flightline_buffer.geojson',
        pos_path='../ROD-ML/IGARSS/data/01_raw/vector/year/2017_rod_4326.geojson',
        kml_path='../ROD-COLLECTION/TerraSARX_Hawaii/dims_op_oc_dfd2_506584917_1/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_S_SRA_20160424T042243_20160424T042251/SUPPORT/GEARTH_POLY.kml',
        output_dir='./daily_outputs',  # Folder where all 18 files will be saved
        bg_time_col='START_DATE_STR',    # Background date column
        pos_time_col='detection_date'    # Positive date column
    )