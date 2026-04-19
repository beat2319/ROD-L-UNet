  Usage:                                        
  python chip_sentinel1.py 2016                                                                                                                                                              
  python chip_sentinel1.py 2020 --label negative --workers 4
                                                                                                                                                                                             
  Key behavior:                                             
  1. Discovers all Sentinel-1 mosaics for the target year and prior year (so early-Jan patches can find Dec revisits)                                                                        
  2. Filters existing patch GeoJSON files (positive and/or negative) by the requested year                                                                                                   
  3. For each GeoJSON file:                                                               
    - Finds the closest Sentinel-1 date to the label date (max 7-day offset)
    - Walks backward 5 more revisits to build a 6-date temporal stack                                                                                                                        
    - Skips if insufficient prior dates exist                        
  4. For each patch in the GeoJSON:                                                                                                                                                          
    - Clips VV (band 1) and VH (band 2) from all 6 mosaics  
    - Computes RVI from linear values: RVI = (4 * VH) / (VV + VH)
    - Converts VV/VH to dB: 10 * log10(value)                    
    - Saves as .npy with shape (6, 3, 256, 256) — channels: [VV_dB, VH_dB, RVI]
                                                                                                                                                                                             
  Output structure:
  data/                                                                                                                                                                                      
    positive_chips/{island}/{year}/positive_chip_{island}_{date}_{patch_id}.npy
    negative_chips/{island}/{year}/negative_chip_{island}_{date}_{patch_id}.npy