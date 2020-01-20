from Utilities.LargeDataProcessing.PatchDownloader import download_patches, generate_slo_shapefile
import numpy as np
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.ml_tools.utilities import rolling_window
import os
from temporal_features import AddStreamTemporalFeaturesTask

import enum
import os
import sys
import numpy as np
import matplotlib as mpl
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
import geopandas as gpd
import datetime as dt
import time

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import sys
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
from eolearn.core import EOTask, EOPatch, LinearWorkflow, EOWorkflow, Dependency, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.io import S2L1CWCSInput
from eolearn.geometry import VectorToRaster
import os
import datetime
from CropData.eopatches import get_bbox_splitter, get_bbox_gdf
from CropData.workflows import get_create_and_add_lpis_workflow
from CropData.plots import draw_bbox, draw_vector_timeless
from CropData.utilities import get_slovenia_crop_geopedia_idx_to_crop_id_mapping, get_group_id
from CropData.tasks import FixLPIS, CreatePatch, AddGeopediaVectorFeature, AddAreaRatio
from Utilities.LargeDataProcessing.all_stream_features import AddBaseFeatures
from Utilities.LargeDataProcessing.geopedija_data import load_LPIS

path = 'E:/Data/PerceptiveSentinel'
# path = '/home/beno/Documents/test'
gdf, bbox_list = generate_slo_shapefile(path)
download_patches(path, gdf, bbox_list[:81])

# no_patches = 1085
no_patches = 81

#path = '/home/beno/Documents/test'
# path = 'E:/Data/PerceptiveSentinel'

patch_location = path + '/Slovenia/'
load = LoadFromDisk(patch_location)

save_path_location = path + '/Slovenia/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

addStreamNDVI = AddStreamTemporalFeaturesTask(data_feature='NDVI')
addStreamSAVI = AddStreamTemporalFeaturesTask(data_feature='SAVI')
addStreamEVI = AddStreamTemporalFeaturesTask(data_feature='EVI')
addStreamARVI = AddStreamTemporalFeaturesTask(data_feature='ARVI')
addStreamSIPI = AddStreamTemporalFeaturesTask(data_feature='SIPI')
addStreamNDWI = AddStreamTemporalFeaturesTask(data_feature='NDWI')

add_data = S2L1CWCSInput(
    layer='BANDS-S2-L1C',
    feature=(FeatureType.DATA, 'BANDS'),  # save under name 'BANDS'
    resx='10m',  # resolution x
    resy='10m',  # resolution y
    maxcc=0.8,  # maximum allowed cloud cover of original ESA tiles
)

execution_args = []
for id in range(no_patches):
    execution_args.append({
        load: {'eopatch_folder': 'eopatch_{}'.format(id)},
        save: {'eopatch_folder': 'eopatch_{}'.format(id)}
    })

workflow = LinearWorkflow(
    add_data,
    AddBaseFeatures(),
    addStreamNDVI,
    addStreamSAVI,
    addStreamEVI,
    addStreamARVI,
    addStreamSIPI,
    addStreamNDWI,
    # allValid('IS_VALID'),
    # *land_cover_task_array,
    # printPatch(),
    save
)

# workflow.execute(execution_args[0])

start_time = time.time()
# runs workflow for each set of arguments in list
executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
# here you choose how many processes/threads you will run, workers=none is max of processors
executor.run(workers=1, multiprocess=False)

country = 'Slovenia'
# country = 'Austria'
year = 2017

load_LPIS(country, year, path, no_patches)

