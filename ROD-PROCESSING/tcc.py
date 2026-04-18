import rioxarray # for the extension to load
import xarray
import rasterio

import geopandas as gpd
from fiona.drvsupport import supported_drivers
import numpy as np
from rasterio import features
import pandas as pd

coastline = gpd.read_file("../ROD-COLLECTION/attachments/Coastline/coastline.geojson")

coastline = coastline.to_crs(32604) 

values_124 = ['Hawaii', 'kahoolawe', 'Lanai', 'Maui', 'Molokai']
orbit_124 = coastline[coastline['isle'].isin(values_124)]

for year in range(2016, 2024):
    tcc = rioxarray.open_rasterio(f"../ROD-COLLECTION/attachments/TCC/nlcd_tcc_hawaii_wgs84_v2023-5_{year}0101_{year}1231.tif")

    tcc_32604 = tcc.rio.reproject("EPSG:32604")

    try:
        # Use all_touched=True to include pixels that touch the geometry
        clipped_tcc = tcc_32604.rio.clip(orbit_124.geometry.values, all_touched=True)
        clipped_tcc.rio.to_raster(raster_path=f"./attachments/tcc/tcc_124_{year}.tif")
        print(f"Successfully processed year {year}")
    except Exception as e:
        print(f"Error for year {year}: {e}")