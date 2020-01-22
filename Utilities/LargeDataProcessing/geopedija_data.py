import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

np.set_printoptions(threshold=sys.maxsize)

GEOPEDIA_LPIS_LAYERS = {'Denmark_LPIS_2018': 2051,
                        'Denmark_LPIS_2017': 2051,
                        'Denmark_LPIS_2016': 2051,
                        'Austria_LPIS_2017': 2034,
                        'Austria_LPIS_2016': 2033,
                        'Slovenia_LPIS_2017': 2038,
                        'Slovenia_LPIS_2016': 2037,
                        'Austria_FELDSTUECKE_2017': 2029,
                        'Austria_FELDSTUECKE_2016': 2027}

GEOPEDIA_LPIS_YEAR_NAME = {'Denmark': 'Year',
                           'Slovenia': 'LETO',
                           'Austria': None}

TIME_INTERVAL = {2016: ['2016-01-01', '2016-09-30'],
                 2017: ['2017-01-01', '2017-09-30'],
                 2018: ['2018-01-01', '2018-09-30']}


class printPatch(EOTask):
    def __init__(self, message="\npatch:"):
        self.message = message

    def execute(self, eopatch):
        print(self.message)
        print(eopatch)
        return eopatch


class WorkflowExclude(EOTask):
    # If an area doesn't have any LPIS data the following tasks are omitted
    def __init__(self, *extra_tasks, feature='LPIS_2017', feature_type=FeatureType.VECTOR_TIMELESS):
        self.feature = feature
        self.feature_type = feature_type
        self.extra_tasks = extra_tasks

    def execute(self, eopatch):
        # df = eopatch.vector_timeless['LPIS_2017']
        # with pd.option_context('display.max_rows', 10, 'display.max_columns',
        #                       None):  # more options can be specified also
        #    print(df)
        if self.feature not in eopatch[self.feature_type]:
            return eopatch
        for t in self.extra_tasks:
            eopatch = t(eopatch)
        return eopatch


class AddGroup(EOTask):
    def __init__(self, dictionary, name_of_feature='LPIS_2017'):
        self.name_of_feature = name_of_feature
        self.dictionary = dictionary

    def execute(self, eopatch):
        gdf = eopatch.vector_timeless[self.name_of_feature]
        gdf['GROUP'] = [self.dictionary[i] for i in gdf.SIFKMRS]
        eopatch.vector_timeless[self.name_of_feature] = gdf
        return eopatch


class AddGroup2(EOTask):
    # Ta task je za groupanje po rasterizu
    def __init__(self, dictionary, name_of_feature='LPIS_2017', feature_type=FeatureType.DATA_TIMELESS):
        self.dictionary = dictionary
        self.name_of_feature = name_of_feature
        self.feature_type = feature_type

    def execute(self, eopatch):
        # Maps multiple old values to new reduced group names#########
        print(eopatch[self.feature_type][self.name_of_feature])
        w, h, _ = eopatch[self.feature_type][self.name_of_feature].shape
        # print(eopatch[self.feature_type][self.name_of_feature].shape)
        new_feat = np.zeros((w, h, 1))
        for wid in range(w):
            for hei in range(h):
                before = eopatch[self.feature_type][self.name_of_feature][wid][hei][0]
                # print(before)
                if math.isnan(before):
                    new_feat[wid][hei][0] = float('nan')
                else:
                    new_feat[wid][hei][0] = self.dictionary[before]
        eopatch[self.feature_type][self.name_of_feature] = new_feat
        # print(new_feat)
        # eopatch[self.feature_type][self.name_of_feature] = [
        #     [self.dictionary[y] if y is not math.isnan(y) else float('nan') for y in x]
        #     for x in
        #     eopatch[self.feature_type][self.name_of_feature].squeeze()]
        # eopatch[self.feature_type][self.name_of_feature] = eopatch[self.feature_type][self.name_of_feature][
        #     ..., np.newaxis]
        # gdf = eopatch.vector_timeless[self.name_of_feature]
        # gdf['GROUP'] = [self.dictionary[i] for i in gdf.SIFKMRS]
        # eopatch.vector_timeless[self.name_of_feature] = gdf
        return eopatch


def create_mapping(country):
    # dictionary = dict()
    slo_def = pd.read_excel('../../CropData/WP2/SLO_LPIS_grouping.xlsx')
    slo_def.rename(index=str, columns={"Group 1": "GROUP_1", "SIFKMRS": "CROP_ID"}, inplace=True)
    slo_def['CROP_ID'][99] = '1204'
    # slo_def.CROP_ID = slo_def.CROP_ID.values.astype(int)

    # aus_def = pd.read_excel('../../CropData/WP2/austrian_lpis_sifrant_23_04_2019.xlsx',
    #                       sheet_name='austrian_lpis_sifrant', header=1)
    # aus_def.rename(index=str, columns={"Group 1 ": "GROUP_1", "Austrian": "CROP_ID"}, inplace=True)

    # tog = pd.concat([slo_def, aus_def], sort=False)
    tog = slo_def

    crops = tog.CROP_ID.values.astype(int)
    groups = tog.GROUP_1.values
    unique_groups = np.unique(tog.GROUP_1.values)

    # crops_to_groups = dict(zip(crops, groups))
    groups_to_number = dict(zip(unique_groups, range(len(unique_groups))))
    crops_to_number = dict(zip(crops, [groups_to_number[i] for i in groups]))

    '''
    for i in range(len(tog['CROP_ID'])):
        dictionary[tog['CROP_ID'][i]] = tog['GROUP_1'][i]

    for i in range(len(slo_def['CROP_ID'])):
        dictionary[slo_def['CROP_ID'][i]] = slo_def['GROUP_1'][i]

    for i in range(len(aus_def['CROP_ID'])):
        dictionary[aus_def['CROP_ID'][i]] = aus_def['GROUP_1'][i]
    '''
    return groups_to_number, crops_to_number


class RemoveFeature(EOTask):

    def __init__(self, feature_type, feature_name):
        self.feature_type = feature_type
        self.feature_name = feature_name

    def execute(self, eopatch):
        eopatch.remove_feature(self.feature_type, self.feature_name)
        return eopatch


def load_LPIS(country, year, path, no_patches):
    patch_location = path + '/{}/'.format(country)
    load = LoadFromDisk(patch_location)
    save_path_location = patch_location
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)
    save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # workflow_data = get_create_and_add_lpis_workflow(country, year, save_path_location)

    name_of_feature = 'LPIS_{}'.format(year)

    groups_to_number, crops_to_number = create_mapping(country)

    layer_id = GEOPEDIA_LPIS_LAYERS[f'{country}_LPIS_{year}']
    ftr_name = f'LPIS_{year}'
    year_filter = (GEOPEDIA_LPIS_YEAR_NAME[country], year) if GEOPEDIA_LPIS_YEAR_NAME[country] is not None else None
    add_lpis = AddGeopediaVectorFeature((FeatureType.VECTOR_TIMELESS, ftr_name),
                                        layer=layer_id, year_filter=year_filter, drop_duplicates=True)
    area_ratio = AddAreaRatio((FeatureType.VECTOR_TIMELESS, ftr_name),
                              (FeatureType.SCALAR_TIMELESS, 'FIELD_AREA_RATIO'))
    fixlpis = FixLPIS(feature=name_of_feature, country=country)

    rasterize = VectorToRaster(vector_input=(FeatureType.VECTOR_TIMELESS, name_of_feature),
                               raster_feature=(FeatureType.MASK_TIMELESS, name_of_feature),
                               values=None,
                               values_column='GROUP',
                               raster_shape=(FeatureType.DATA, 'BANDS'),
                               raster_dtype=np.int16,
                               no_data_value=np.nan
                               )

    add_group = AddGroup(crops_to_number, name_of_feature)
    remove_dtf = RemoveFeature(FeatureType.VECTOR_TIMELESS, name_of_feature)

    exclude = WorkflowExclude(area_ratio, fixlpis, add_group, rasterize,
                              remove_dtf)

    workflow = LinearWorkflow(
        load,
        add_lpis,
        exclude,
        save
    )

    execution_args = []
    for i in range(no_patches):
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(i)},
            save: {'eopatch_folder': 'eopatch_{}'.format(i)}
        })
    ##### here you choose how many processes/threads you will run, workers=none is max of processors

    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    # executor.run(workers=None, multiprocess=True)
    executor.run()

    # workflow.execute({load: {'eopatch_folder': 'eopatch_0'},
    #                   save: {'eopatch_folder': 'eopatch_0'}
    #                   })

    # executor = EOExecutor(workflow, [{load: {'eopatch_folder': 'eopatch_0'},
    #                                  save: {'eopatch_folder': 'eopatch_0'}
    #                                  }], save_logs=True, logs_folder='ExecutionLogs')
    # executor.run(workers=None, multiprocess=True)


if __name__ == '__main__':
    # path = 'E:/Data/PerceptiveSentinel'
    path = '/home/beno/Documents/test'
    # no_patches = 1
    no_patches = 1
    country = 'Slovenia'
    # country = 'Austria'
    year = 2017

    load_LPIS(country, year, path, no_patches)
