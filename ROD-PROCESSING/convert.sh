#!/bin/bash

# Target CRS from your image
TARGET_EPSG="EPSG:4326"
CURRENT_DIR=$(pwd)

# Explicitly defined paths based on your input
PATHS=(
  "/home/benatk04/Docker/Obsidian/data/vault/Projects/ROD-COLLECTION/TerraSARX_Hawaii/dims_op_oc_dfd2_532227681_2/TSX-1.SAR.L1B/TDX1_SAR__SSC______SM_S_SRA_20170124T042249_20170124T042257/TDX1_SAR__SSC______SM_S_SRA_20170124T042249_20170124T042257.xml"
  "/home/benatk04/Docker/Obsidian/data/vault/Projects/ROD-COLLECTION/TerraSARX_Hawaii/dims_op_oc_dfd2_532316196_2/TSX-1.SAR.L1B/TSX1_SAR__SSC______SM_S_SRA_20170204T042248_20170204T042256/TSX1_SAR__SSC______SM_S_SRA_20170204T042248_20170204T042256.xml"
)

for XML_FILE in "${PATHS[@]}"; do
    echo "------------------------------------------------"
    
    if [ ! -f "$XML_FILE" ]; then
        echo "SKIPPING: File not found at $XML_FILE"
        continue
    fi

    # Create a clean output name based on the filename
    OUTPUT_NAME="$(basename "$XML_FILE" .xml)_UTM4N.tif"
    
    echo "Processing: $(basename "$XML_FILE")"

    # -tps: Thin Plate Spline for better SAR warping
    # -b 1: Selects the Amplitude/Intensity band
    # -q: Quiet mode to hide the PROJ/Latitude warnings
    gdalwarp "$XML_FILE" "$OUTPUT_NAME" \
        -t_srs "$TARGET_EPSG" \
        -tps \
        -r bilinear \
        -of COG \
        -ot Float32 \
        -b 1 \
        -dstnodata 0 \
        -co COMPRESS=DEFLATE \
        -co PREDICTOR=2 \
        -overwrite -q

    echo "DONE! Saved as: $CURRENT_DIR/$OUTPUT_NAME"
done