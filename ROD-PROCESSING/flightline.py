import geopandas as gpd
import pandas as pd

coastline = gpd.read_file("../ROD-COLLECTION/attachments/Coastline/coastline.geojson")
coastline = coastline.to_crs("EPSG:32604")

# Mapping between flightline Island property and coastline isle property
island_name_map = {
    'Hawaii': 'Hawaii',
    'Kauai': 'Kauai',
    'Lanai': 'Lanai',
    'Maui': 'Maui',
    'Molokai': 'Molokai',
    'Oahu': 'Oahu',
}

# orbit_124 islands (exclude Kauai, Oahu)
orbit_124_islands = ['Hawaii', 'Lanai', 'Maui', 'Molokai']

for year in range(2016, 2022):
    flightline = gpd.read_file(f'../ROD-COLLECTION/attachments/DMSM/flightlines/flightlines_{year}.geojson')

    # Convert milliseconds to datetime (MM/DD/YYYY format)
    flightline['START_DATE_STR'] = pd.to_datetime(flightline['START_DATE'], unit='ms').dt.strftime('%m/%d/%Y')

    # Project to EPSG:4326 (WGS 84) and Reproject to EPSG:32604 (WGS 84)
    flightline = flightline.set_crs(epsg=4326, allow_override=True)
    flightline = flightline.to_crs(epsg=32604)

    # Create a 2-mile buffer
    flightline['geometry'] = flightline.geometry.buffer(1609.34)

    # Process each island separately
    for island in flightline['Island'].unique():
        # Skip islands not in orbit_124
        if island not in orbit_124_islands:
            continue

        # Filter by island
        island_flightlines = flightline[flightline['Island'] == island]

        # Clip to that island's coastline area
        island_coastline = coastline[coastline['isle'].isin([island_name_map[island]])]
        island_processed = gpd.clip(island_flightlines, island_coastline)

        # Save the processed file
        island_name_lower = island.lower()
        island_processed.to_file(f'./attachments/flightlines/flightlines_{island_name_lower}_{year}_processed.geojson', driver='GeoJSON')
        print(f'Processed flightlines_{island_name_lower}_{year}.geojson with {len(island_processed)} features')
