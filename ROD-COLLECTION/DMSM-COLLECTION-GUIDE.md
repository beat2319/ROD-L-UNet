- With dataset creator Brian Tuckers approval use the ESRI ArcGIS REST to collect the specific features (both mortality detections (2016-2024) and flightlines(2016-2021))
	- Brian should be updated both soon, but his focus has changed to fence lines analysis
	- [Flightlines](https://services1.arcgis.com/x4h61KaW16vFs7PM/ArcGIS/rest/services/Flightlines_WFL1/FeatureServer/layers)
		- ensure that you filter by layers (0 - 5)
	- [ROD_Mortality](https://services1.arcgis.com/x4h61KaW16vFs7PM/ArcGIS/rest/services/PotentialRODArea_All/FeatureServer/5)
- Now we will use the [collection guide by Jonathan Chang](https://jonathanchang.org/blog/downloading-esri-online-shapefiles/)
	- Its fairly simple once if have the Feature Server found, otherwise requires some quick usage with a browser inspect and then looking for the feature server on the network tab.
		- Will update if I have time on a more detailed guide
	- ```bash
	  pip install esridump
	  ```
	- ```bash
	  esri2geojson "https://services1.arcgis.com/x4h61KaW16vFs7PM/ArcGIS/rest/services/Flightlines_WFL1/FeatureServer/5" flightline_2016.geojson
	  ```