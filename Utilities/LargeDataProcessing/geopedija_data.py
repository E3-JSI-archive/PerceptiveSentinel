import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
from eolearn.core import EOTask, EOPatch, LinearWorkflow, EOWorkflow, Dependency, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.io import S2L1CWCSInput
from eolearn.geometry import VectorToRaster
import os
import datetime
from eopatches import get_bbox_splitter, get_bbox_gdf
from workflows import get_create_and_add_lpis_workflow
from plots import draw_bbox, draw_vector_timeless
from utilities import get_slovenia_crop_geopedia_idx_to_crop_id_mapping, get_group_id
from tasks import FixLPIS, CreatePatch, AddGeopediaVectorFeature, AddAreaRatio

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
    # Class za pogojno izvajanje eotaskov
    def __init__(self, *extra_tasks, feature='LPIS_2017', feature_type=FeatureType.VECTOR_TIMELESS):
        self.feature = feature
        self.feature_type = feature_type
        self.extra_tasks = extra_tasks

    def execute(self, eopatch):
        if self.feature not in eopatch[self.feature_type]:
            return eopatch
        for t in self.extra_tasks:
            eopatch = t(eopatch)
        return eopatch


class AddGroup(EOTask):
    def __init__(self, dictionary, name_of_feature='LPIS_2017',
                 feature_type=FeatureType.VECTOR_TIMELESS):
        self.name_of_feature = name_of_feature
        self.dictionary = dictionary
        self.feature_type = feature_type

    def execute(self, eopatch):
        if self.name_of_feature not in eopatch[self.feature_type]:
            return eopatch
        gdf = eopatch[self.feature_type][self.name_of_feature]
        gdf['GROUP'] = [self.dictionary[i] for i in gdf.SIFKMRS]
        eopatch.vector_timeless[self.name_of_feature] = gdf
        return eopatch


def load_LPIS(path, no_patches):
    country_name = 'Slovenia'
    year = 2017

    patch_location = path + '/Slovenija/'
    load = LoadFromDisk(patch_location)

    save_path_location = path + '/Slovenija/'
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)
    save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow_data = get_create_and_add_lpis_workflow(country_name, year, save_path_location)

    name_of_feature = 'LPIS_2017'

    ## Group classes
    slo_crop_def = pd.read_excel('./WP2/SLO_LPIS_grouping.xlsx')
    slo_crop_def.rename(index=str, columns={"Group 1": "GROUP_1", "SIFKMRS": "CROP_ID"}, inplace=True)
    slo_crop_def['CROP_ID'][99] = '1204'
    slo_crops = slo_crop_def.CROP_ID.values.astype(int)
    slo_groups = slo_crop_def.GROUP_1.values
    slo_unique_groups = np.unique(slo_crop_def.GROUP_1.values)
    crops_to_groups = dict(zip(slo_crops, slo_groups))
    groups_to_number = dict(zip(slo_unique_groups, range(len(slo_unique_groups))))
    crops_to_number = dict(zip(slo_crops, [groups_to_number[i] for i in slo_groups]))
    ## Workflow all
    tasks = workflow_data.get_tasks()
    add_lpis = tasks['add_lpis']
    area_ratio = tasks['area_ratio']
    fixlpis = FixLPIS(feature=name_of_feature, country=country_name)

    add_group = AddGroup(crops_to_number)
    rasterize = VectorToRaster(vector_input=(FeatureType.VECTOR_TIMELESS, name_of_feature),
                               raster_feature=(FeatureType.MASK_TIMELESS, name_of_feature),
                               values=None,
                               values_column='GROUP',
                               raster_shape=(FeatureType.DATA, 'BANDS'),
                               raster_dtype=np.int16,
                               no_data_value=np.nan
                               )

    exclude = WorkflowExclude(area_ratio, fixlpis, add_group, rasterize)

    workflow = LinearWorkflow(
        load,
        add_lpis,
        # area_ratio,
        # fixlpis,
        # add_group,
        # rasterize,
        exclude,
        save,
    )

    execution_args = []
    for id in range(no_patches):
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(id)},
            save: {'eopatch_folder': 'eopatch_{}'.format(id)}
        })

    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    # here you choose how many processes/threads you will run, workers=none is max of processors
    executor.run(workers=None, multiprocess=True)
    # executor.run()

    # workflow.execute({load: {'eopatch_folder': 'eopatch_0'},
    #                  save: {'eopatch_folder': 'eopatch_0'}
    #                 })


if __name__ == '__main__':
    path = 'E:/Data/PerceptiveSentinel'
    # path = '/home/beno/Documents/test'
    no_patches = 1061

    load_LPIS(path, no_patches)
