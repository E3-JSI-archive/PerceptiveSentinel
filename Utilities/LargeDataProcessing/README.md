# Large Data Processing Utilities
Some functions that can be used for downloading and processing large amounts of data containing multiple patches.

## Patch Downloader
Generates a shapefile from a area that is read from a geojson file in /data. Splits the area into patches which will be
downloaded separately. The size of the patches can be adjusted here: 
`BBoxSplitter([country_shape], country_crs, (25 * 2, 17 * 2))`

## Sampling
The function `sample_patches` will make a pandas dataframe with one pixel samples from the desired eopatches. The sample includes
the desired features and can use a mask for acquisition.

## Edge Extraction
This function will make a mask which marks the areas where there are **no** edges. The edges are calculated by using a
weighted sum of a features through the whole interval where the weights are determined by the magnitude of the feature 
in the neighbourhood. Then the whole image is thresholded. Optionally the mask can also exclude areas where one
feature is low such as cities and water where the NDVI is low.

## Geopedija Data
Downloads the LPIS data from geopedija transforms it from vector to raster, merges all the groups as specified by the 
`CropData/WP2/SLO_LPIS_grouping.xlsx` where the groups are transformed to numbers and saves it to existing patches. This
currently works only for Slovenia for years 2016 and 2017.

We currently do not have the mapping for group reduction for Austria and Denmark. Once obtained the data download will be
implemented for this countries too.

## Stream Features
The code in the `temporal_features.py` is the work in progress by Filip Koprivec. Once revised it should be integrated
into the eo-learn package and not needed here. File `all_stream_features.py` uses the before mentioned code to compute all 
the Valero features and saves them in the patches.

## Visualization
Some examples of code that can be used to display the patches. It includes a function for coloring the areas with different
ids with random color. Used for displaying clusters or LPIS.

## Fix
This is just the code that was once needed to repair broken eopatches by redownloading them and computing all the features.

