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
from eolearn.geometry import VectorToRaster


class printPatch(EOTask):
    def __init__(self, message="\npatch:"):
        self.message = message

    def execute(self, eopatch):
        print(self.message)
        print(eopatch)
        return eopatch


class allValid(EOTask):

    def __init__(self, mask_name):
        self.mask_name = mask_name

    def execute(self, eopatch):
        # print(eopatch)
        t, w, h, _ = eopatch.data['BANDS'].shape
        eopatch.add_feature(FeatureType.MASK, self.mask_name, np.ones((t, w, h, 1)))
        return eopatch


class LULC(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    CULTIVATED_LAND = (1, 'Cultivated Land', 'xkcd:lime')
    FOREST = (2, 'Forest', 'xkcd:darkgreen')
    GRASSLAND = (3, 'Grassland', 'orange')
    SHRUBLAND = (4, 'Shrubland', 'xkcd:tan')
    WATER = (5, 'Water', 'xkcd:azure')
    WETLAND = (6, 'Wetlands', 'xkcd:lightblue')
    TUNDRA = (7, 'Tundra', 'xkcd:lavender')
    ARTIFICIAL_SURFACE = (8, 'Artificial Surface', 'crimson')
    BARELAND = (9, 'Bareland', 'xkcd:beige')
    SNOW_AND_ICE = (10, 'Snow and Ice', 'black')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3


def normalize_feature(feature):  # Assumes similar max and min throughout different features
    f_min = np.min(feature)
    f_max = np.max(feature)
    if f_max != 0:
        return (feature - f_min) / (f_max - f_min)


def temporal_derivative(data, window_size=(3,)):
    padded_slope = np.zeros(data.shape)
    window = rolling_window(data, window_size, axes=0)

    slope = window[..., -1] - window[..., 0]  # TODO Missing division with time
    padded_slope[1:-1] = slope  # Padding with zeroes at the beginning and end

    return normalize_feature(padded_slope)


class AddBaseFeatures(EOTask):

    def __init__(self, c1=6, c2=7.5, L=1):
        self.c1 = c1
        self.c2 = c2
        self.L = L

    def execute(self, eopatch):
        nir = eopatch.data['BANDS'][..., [7]]
        eopatch.add_feature(FeatureType.DATA, 'NIR', nir)
        blue = eopatch.data['BANDS'][..., [1]]
        red = eopatch.data['BANDS'][..., [3]]

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue + 0.000000001), -1,
                       1)  # TODO nekako boljše to rešit division by 0
        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)
        arvi_slope = temporal_derivative(arvi.squeeze())
        eopatch.add_feature(FeatureType.DATA, 'ARVI_SLOPE', arvi_slope[..., np.newaxis])

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L + 0.000000001)), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)
        evi_slope = temporal_derivative(evi.squeeze())
        eopatch.add_feature(FeatureType.DATA, 'EVI_SLOPE', evi_slope[..., np.newaxis])

        ndvi = np.clip((nir - red) / (nir + red + 0.000000001), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)
        ndvi_slope = temporal_derivative(ndvi.squeeze())
        eopatch.add_feature(FeatureType.DATA, 'NDVI_SLOPE', ndvi_slope[..., np.newaxis])  # ASSUMES EVENLY SPACED

        band_a = eopatch.data['BANDS'][..., 1]
        band_b = eopatch.data['BANDS'][..., 3]
        ndvi = np.clip((band_a - band_b) / (band_a + band_b + 0.000000001), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDWI', ndvi[..., np.newaxis])

        sipi = np.clip((nir - blue) / (nir - red + 0.000000001), 0, 2)  # TODO nekako boljše to rešit division by 0
        eopatch.add_feature(FeatureType.DATA, 'SIPI', sipi)

        Lvar = 0.5
        savi = np.clip(((nir - red) / (nir + red + Lvar + 0.000000001)) * (1 + Lvar), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'SAVI', savi)

        return eopatch


if __name__ == '__main__':

    # no_patches = 1085
    no_patches = 1

    path = '/home/beno/Documents/test'
    # path = 'E:/Data/PerceptiveSentinel'

    patch_location = path + '/Slovenija/'
    load = LoadFromDisk(patch_location)

    save_path_location = path + '/Slovenija/'
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)

    save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    addStreamNDVI = AddStreamTemporalFeaturesTask(data_feature='NDVI')
    addStreamSAVI = AddStreamTemporalFeaturesTask(data_feature='SAVI')
    addStreamEVI = AddStreamTemporalFeaturesTask(data_feature='EVI')
    addStreamARVI = AddStreamTemporalFeaturesTask(data_feature='ARVI')
    addStreamSIPI = AddStreamTemporalFeaturesTask(data_feature='SIPI')
    addStreamNDWI = AddStreamTemporalFeaturesTask(data_feature='NDWI')

    '''
    lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
    lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 11, 1), lulc_cmap.N)

    land_cover_path = path+'/shapefiles/slovenia.shp'

    land_cover = gpd.read_file(land_cover_path)

    land_cover_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    land_cover_array = []
    for val in land_cover_val:
        temp = land_cover[land_cover.lulcid == val]
        temp.reset_index(drop=True, inplace=True)
        land_cover_array.append(temp)
        del temp

    rshape = (FeatureType.MASK, 'IS_VALID')

    land_cover_task_array = []
    for el, val in zip(land_cover_array, land_cover_val):
        land_cover_task_array.append(VectorToRaster(
            feature=(FeatureType.MASK_TIMELESS, 'LULC'),
            vector_data=el,
            raster_value=val,
            raster_shape=rshape,
            raster_dtype=np.uint8))
    '''
    execution_args = []
    for id in range(no_patches):
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(id)},
            save: {'eopatch_folder': 'eopatch_{}'.format(id)}
        })

    workflow = LinearWorkflow(
        load,
        AddBaseFeatures(),
        addStreamNDVI,
        addStreamSAVI,
        addStreamEVI,
        addStreamARVI,
        addStreamSIPI,
        addStreamNDWI,
        # allValid('IS_VALID'),
        # *land_cover_task_array,
        #printPatch(),
        save
    )

    #workflow.execute(execution_args[0])

    start_time = time.time()
    # runs workflow for each set of arguments in list
    executor = EOExecutor(workflow, execution_args, save_logs=True)
    # here you choose how many processes/threads you will run, workers=none is max of processors
    executor.run(workers=None, multiprocess=True)

    file = open('stream_timing.txt', 'a')
    running = str(dt.datetime.now()) + ' Running time: {}\n'.format(time.time() - start_time)
    print(running)
    file.write(running)
    file.close()

    # executor.make_report()