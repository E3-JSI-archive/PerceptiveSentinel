from Utilities.LargeDataProcessing.PatchDownloader import download_patches, generate_slo_shapefile
import numpy as np
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.ml_tools.utilities import rolling_window
import os
from temporal_features import AddStreamTemporalFeaturesTask
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
import enum
import os
import sys
import numpy as np
import matplotlib as mpl
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
import geopandas as gpd
import datetime as dt
import time
import cv2
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
from eolearn.features import LinearInterpolation, SimpleFilterTask
from extract_edges import ExtractEdgesTask

path = 'E:/Data/PerceptiveSentinel'
# path = '/home/beno/Documents/test'
# gdf, bbox_list = generate_slo_shapefile(path)
# download_patches(path, gdf, bbox_list[:81])

# no_patches = 1085
no_patches = 1061

# path = '/home/beno/Documents/test'
# path = 'E:/Data/PerceptiveSentinel'

patch_location = path + '/Slovenia/'
load = LoadFromDisk(patch_location, lazy_loading=True)

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

# add_data = S2L1CWCSInput(
#     layer='BANDS-S2-L1C',
#     feature=(FeatureType.DATA, 'BANDS'),  # save under name 'BANDS'
#     resx='10m',  # resolution x
#     resy='10m',  # resolution y
#     maxcc=0.8,  # maximum allowed cloud cover of original ESA tiles
# )

cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
add_clm = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='80m', cm_size_x='80m',
                           cmask_feature='CLM',  # cloud mask name
                           cprobs_feature='CLP'  # cloud prob. map name
                           )

linear_interp = LinearInterpolation(
    (FeatureType.DATA, 'BANDS'),  # name of field to interpolate
    mask_feature=(FeatureType.MASK, 'IS_VALID'),  # mask to be used in interpolation
    bounds_error=False  # extrapolate with NaN's
)


class RemoveUnwantedFeatures(EOTask):
    def __init__(self):
        self.features = [(FeatureType.DATA, 'ARVI_SLOPE'), (FeatureType.DATA, 'EVI_SLOPE'),
                         (FeatureType.DATA, 'NDVI_SLOPE'),
                         (FeatureType.MASK, 'ARVI_EDGE'),
                         (FeatureType.MASK, 'EVI_EDGE'),
                         (FeatureType.MASK, 'GRAY_EDGE'),
                         (FeatureType.MASK, 'NDVI_EDGE'),
                         (FeatureType.MASK_TIMELESS, 'LOW_NDVI'),
                         ]
        base_names = ['ARVI', 'EVI', 'NDVI', 'SAVI', 'SIPI', 'NWDI']
        suffix_name = ['_diff_diff', '_diff_max', '_diff_min', '_max_mean_feature', '_max_mean_len', '_max_mean_surf',
                       '_max_val', '_mean_val', '_min_val', '_neg_len', '_neg_rate', '_neg_surf', '_neg_tran',
                       '_pos_len', '_pos_rate', '_pos_surf', '_pos_tran', '_sd_val']
        good_features = ['NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val',
                         'NDVI_min_val', 'NDWI_max_mean_len',
                         'SAVI_min_val']
        for b in base_names:
            for suff in suffix_name:
                fet = b + suff
                if fet not in good_features:
                    self.features.append((FeatureType.DATA_TIMELESS, fet))

    def execute(self, eopatch):
        for f in self.features:
            eopatch.remove_feature(f[0], f[1])
        return eopatch


class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))


add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(),
                                      'IS_VALID'  # name of output mask
                                      )

execution_args = []
for id in range(no_patches):
    execution_args.append({
        load: {'eopatch_folder': 'eopatch_{}'.format(id)},
        save: {'eopatch_folder': 'eopatch_{}'.format(id)}
    })

structuring_2d = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]
                  ]

segmentation = ExtractEdgesTask(
    edge_features=[
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'EVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((5, 5), 2)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'ARVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((5, 5), 2)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'NDVI',
         "CannyThresholds": (40, 100),
         "BlurArguments": ((5, 5), 2)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'GRAY',
         "CannyThresholds": (5, 40),
         "BlurArguments": ((3, 3), 2)
         }
    ],
    structuring_element=structuring_2d,
    excluded_features=[((FeatureType.DATA, 'NDVI'), 0.3)],
    dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    output_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
    adjust_function=lambda x: cv2.GaussianBlur(x, (9, 9), 5),
    adjust_threshold=0.05,
    yearly_low_threshold=0.8)


class ValidDataFractionPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold


valid_data_predicate = ValidDataFractionPredicate(0.8)
filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

workflow = LinearWorkflow(
    load,
    add_clm,
    add_sh_valmask,
    filter_task,
    linear_interp,
    AddBaseFeatures(),
    # segmentation,
    addStreamNDVI,
    addStreamEVI,
    addStreamARVI,
    addStreamSIPI,
    addStreamSAVI,
    addStreamNDWI,
    RemoveUnwantedFeatures(),
    save
)

# workflow.execute(execution_args[1])

start_time = time.time()
executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
executor.run(workers=None, multiprocess=True)
file = open('timing.txt', 'a')
running = str(
    dt.datetime.now()) + ' cloud mask, stream features NDVI, EVI, ARVI, SIPI, SAVI, NDWI, removal. Running time: {}\n'.format(
    time.time() - start_time)
print(running)
file.write(running)
file.close()
country = 'Slovenia'
# country = 'Austria'
year = 2017

# load_LPIS(country, year, path, no_patches)
