import rioxarray
import geopandas as gpd
from fiona.drvsupport import supported_drivers
import numpy as np
from rasterio import features
import pandas as pd

# Enable KML support in fiona
supported_drivers['KML'] = 'rw'

# Load the KML and the Raster
mask_gdf = gpd.read_file('../ROD-COLLECTION/TerraSARX_Hawaii/dims_op_oc_dfd2_506584917_1/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_S_SRA_20160424T042243_20160424T042251/SUPPORT/GEARTH_POLY.kml', driver='KML')
raster = rioxarray.open_rasterio('../ROD-ML/IGARSS/data/01_raw/tree_cover/nlcd_tcc_hawaii_2017_v2021-4.tif')

# 1. Load and handle KML quirks
mask_gdf = mask_gdf.explode(index_parts=False) # Break down collections

# 2. Ensure CRS consistency
if mask_gdf.crs is None:
    mask_gdf.set_crs("epsg:4326", inplace=True)
mask_gdf = mask_gdf.to_crs(raster.rio.crs)

# 3. Clip using raw geometry values
clipped_raster = raster.rio.clip(mask_gdf.geometry.values)

# Apply the filter: Keep values > 10 and <= 100
# Others are set to NaN
filtered_raster = clipped_raster.where((clipped_raster > 10) & (clipped_raster <= 100))

# 1. Extract the data and transform
data = filtered_raster.values[0] # Get the first band
transform = filtered_raster.rio.transform()

# 2. Vectorize the non-null pixels
shapes = features.shapes(
    data.astype('float32'), 
    mask=~np.isnan(data), 
    transform=transform
)

# 3. Format into a GeoDataFrame
records = [{"geometry": s, "properties": {"pixel_value": v}} for s, v in shapes]
gdf_final = gpd.GeoDataFrame.from_features(records, crs=clipped_raster.rio.crs)

# 4. Export to GeoJSON
gdf_final.to_file("output.geojson", driver='GeoJSON')