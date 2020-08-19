from sentinelhub import GeopediaFeatureIterator, GeopediaSession
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor
from eolearn.io import SentinelHubDemTask

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from joblib import dump, load
from sklearn import tree
from collections import Counter


def get_slovenia_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Slovenia.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    gpd_session = GeopediaSession()
    to_crop_id = list(GeopediaFeatureIterator(layer='2036', gpd_session=gpd_session))
    to_crop_id = [{'crop_geopedia_idx': code['id'], **code['properties']} for code in to_crop_id]
    to_crop_id = pd.DataFrame(to_crop_id)
    to_crop_id['crop_geopedia_idx'] = pd.to_numeric(to_crop_id.crop_geopedia_idx)

    return to_crop_id


def get_austria_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Austria.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    gpd_session = GeopediaSession()
    to_crop_id = list(GeopediaFeatureIterator(layer='2032', gpd_session=gpd_session))
    to_crop_id = [{'crop_geopedia_idx': code['id'], **code['properties']} for code in to_crop_id]
    to_crop_id = pd.DataFrame(to_crop_id)
    to_crop_id['crop_geopedia_idx'] = pd.to_numeric(to_crop_id.crop_geopedia_idx)
    to_crop_id.rename(index=str, columns={"SNAR_BEZEI": "SNAR_BEZEI_NAME"}, inplace=True)
    to_crop_id.rename(index=str, columns={"crop_geopedia_idx": "SNAR_BEZEI"}, inplace=True)

    return to_crop_id


def get_danish_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Austria.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    gpd_session = GeopediaSession()
    to_crop_id = list(GeopediaFeatureIterator(layer='2050', gpd_session=gpd_session))
    to_crop_id = [{'crop_geopedia_idx': code['id'], **code['properties']} for code in to_crop_id]
    to_crop_id = pd.DataFrame(to_crop_id)
    to_crop_id['crop_geopedia_idx'] = pd.to_numeric(to_crop_id.crop_geopedia_idx)

    return to_crop_id


def get_group_id(crop_group, crop_group_df, group_name='GROUP_1_NAME',
                 group_id='GROUP_1_ID', default_value=0):
    """
    Returns numeric crop group value for specified crop group name. The mapping is obtained from
    the specified crop group pandas DataFrame.
    """
    values = crop_group_df[crop_group_df[group_name] == crop_group][group_id].values
    if len(values) == 0:
        return default_value
    else:
        return values[-1]


class AddBaseFeatures(EOTask):

    def __init__(self, c1=6, c2=7.5, L=1, Lvar=0.5, delta=10 ** -10):
        self.c1 = c1
        self.c2 = c2
        self.L = L
        self.Lvar = Lvar

        # We add a small number that doesn't significantly change the result to avoid divisions by zero
        self.delta = delta

    def execute(self, eopatch):
        nir = eopatch.data['BANDS'][..., [7]]
        blue = eopatch.data['BANDS'][..., [1]]
        red = eopatch.data['BANDS'][..., [3]]
        eopatch.add_feature(FeatureType.DATA, 'NIR', nir)
        eopatch.add_feature(FeatureType.DATA, 'BLUE', blue)
        eopatch.add_feature(FeatureType.DATA, 'RED', red)

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue + self.delta), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L + self.delta)), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)

        ndvi = np.clip((nir - red) / (nir + red + self.delta), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)

        ndwi = np.clip((blue - red) / (blue + red + self.delta), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDWI', ndwi)

        sipi = np.clip((nir - blue) / (nir - red + self.delta), 0, 2)
        eopatch.add_feature(FeatureType.DATA, 'SIPI', sipi)

        savi = np.clip(((nir - red) / (nir + red + self.Lvar + self.delta)) * (1 + self.Lvar), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'SAVI', savi)

        return eopatch


# Color scheme for coloring the classes and legend
class_name_color = {0: ('Not Farmland', 'xkcd:black'),
                    1: ('Grass', 'xkcd:brown'),
                    2: ('Maize', 'xkcd:butter'),
                    3: ('Orchards', 'xkcd:royal purple'),
                    4: ('Other', 'xkcd:white'),
                    5: ('Peas', 'xkcd:spring green'),
                    6: ('Potatoes', 'xkcd:poo'),
                    7: ('Pumpkins', 'xkcd:pumpkin'),
                    8: ('Soybean', 'xkcd:baby green'),
                    9: ('Summer cereals', 'xkcd:cool blue'),
                    10: ('Sun flower', 'xkcd:piss yellow'),
                    11: ('Vegetables', 'xkcd:bright pink'),
                    12: ('Vineyards', 'xkcd:grape'),
                    13: ('Winter cereals', 'xkcd:ice blue'),
                    14: ('Winter rape', 'xkcd:neon blue')}


def draw_histogram(distribution):
    named = dict()
    for no in distribution.keys():
        value = distribution[no]
        named[class_name_color[no][0]] = value

    distribution = named
    plt.bar(range(len(distribution)), list(distribution.values()), align='center')
    plt.xticks(range(len(distribution)), list(distribution.keys()), rotation='vertical')
    plt.show()
