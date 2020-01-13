from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from eolearn.io import S2L1CWCSInput, S2L1CWMSInput
from shapely.geometry import Polygon
import os
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
import datetime as dt
import time
from sklearn.utils import resample
import random
import pandas as pd
import collections


def sample_patches(path, no_patches, no_samples, class_feature, mask_feature, features):
    columns = [class_feature[1]] + [x[1] for x in features]
    class_name = class_feature[1]
    sample_dict = []

    for i in range(no_patches):
        eopatch = EOPatch.load('{}/eopatch_{}'.format(path, i))
        _, height, width, _ = eopatch.data['BANDS'].shape
        mask = eopatch[mask_feature[0]][mask_feature[1]].squeeze()
        no_samples = min(height * width, no_samples)
        subsample_id = []
        for h in range(height):
            for w in range(width):
                if mask == None or mask[h][w] == 1:
                    subsample_id.append((h, w))
        subsample_id = random.sample(subsample_id, no_samples)
        for x, y in subsample_id:

            class_value = float(eopatch[class_feature[0]][class_feature[1]][x][y])
            if np.isnan(class_value):
                class_value = float(-1)
            dict_temp = dict(
                [(class_name, class_value)] +
                [(f[1], float(eopatch[f[0]][f[1]][x][y])) for f in features]
            )
            sample_dict.append(dict_temp)
            # print(sample_dict)

    df = pd.DataFrame(sample_dict, columns=columns)
    # print(df)
    class_count = collections.Counter(df[class_feature[1]]).most_common()
    least_common = class_count[-1][1]
    # df_downsampled = df[df[class_name] == least_common]
    df_downsampled = pd.DataFrame(columns=columns)
    names = [name[0] for name in class_count]
    dfs = [df[df[class_name] == x] for x in names]
    for d in dfs:
        nd = resample(d, replace=False, n_samples=least_common)
        df_downsampled = df_downsampled.append(nd)

    return df_downsampled


if __name__ == '__main__':
    # path = 'E:/Data/PerceptiveSentinel'
    path = '/home/beno/Documents/test/Slovenia'

    no_patches = 3
    no_samples = 10000
    class_feature = (FeatureType.MASK_TIMELESS, 'LPIS_2017')
    mask = (FeatureType.MASK_TIMELESS, 'EDGES_INV')
    features = [(FeatureType.DATA_TIMELESS, 'NDVI_mean_val'), (FeatureType.DATA_TIMELESS, 'SAVI_max_val'),
                (FeatureType.DATA_TIMELESS, 'NDVI_pos_surf')]

    samples = sample_patches(path, no_patches, no_samples, class_feature, mask, features)
    print(samples)
