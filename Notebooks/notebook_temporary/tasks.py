import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from sentinelhub import CRS, GeopediaFeatureIterator
from eolearn.core import EOTask, EOPatch, FeatureType
from skimage.morphology import disk, binary_dilation, binary_erosion

from .utilities import get_slovenia_crop_geopedia_idx_to_crop_id_mapping
from .utilities import get_austria_crop_geopedia_idx_to_crop_id_mapping
from .utilities import get_danish_crop_geopedia_idx_to_crop_id_mapping


class ValidDataFractionPredicate:
    """
    Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold


class CreatePatch(EOTask):
    """Creates an empty EOPatch for given BBOX."""

    def execute(self, *, bbox):
        """Returns newly created EOPatch given BBOX. 
        
        :param bbox: specifies the bounding box of the EOPatch. Coordinates must be in
                     the specified coordinate reference system. Required.
        :type bbox: BBox
        :return: new empty EOPatch for given BBOX.
        :rtype: EOPatch
        """
        eopatch = EOPatch()
        eopatch.bbox = bbox

        return eopatch


class FixLPIS(EOTask):
    """
    Fixes known issues of LPIS data stored as vector_timeless feature in the EOPatch.

    Known issues depend on the country and are:
    * Slovenia:
        * column "SIFRA_KMRS" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "Crop type classification for Slovenia" and not CROP ID as the name suggests
            * This task replaces "SIFRA_KMRS" with "SIFKMRS" that truly represents CROP ID
        * CROP IDs are strings and not integers, which represents a problem when burning in the
          LPIS data to raster.
            * This task replaces "204_a" with "1204"
            * column is casted to numeric
    * Austria:
        * column "SNAR_BEZEI" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "Austria LPIS (SNAR_BEZEI)" and not CROP NAME as the name suggests
        * a new column is added "SNAR_BEZEI_NAME" with the CROP NAME as appears in Austrian LPIS data
    * Denmark:
        * columns "CropName" and "PreCropName" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "DK LPIS crop type" and not CROP NAME as the name suggests
        * they are replaced with two new columns "Crop Name" and "PreCrop Name" with the CROP NAME as
          appears in Danish LPIS data

    :param feature: Name of the vector_timeless feature with LPIS data
    :type feature: str
    :param country: Name of the country
    :type country: str
    """

    def __init__(self, feature, country):
        self.feature = feature
        self.country = country
        self.mapping = None

        self._set_mapping()

    def _set_mapping(self):
        if self.country == 'Slovenia':
            self.mapping = get_slovenia_crop_geopedia_idx_to_crop_id_mapping()
        elif self.country == 'Austria':
            self.mapping = get_austria_crop_geopedia_idx_to_crop_id_mapping()
        elif self.country == 'Denmark':
            self.mapping = get_danish_crop_geopedia_idx_to_crop_id_mapping()

    def _fix_slovenian_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature].rename(index=str, columns={"SIFRA_KMRS": "crop_geopedia_idx"},
                                                     inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature].loc[eopatch.vector_timeless[self.feature]['SIFKMRS'] == '204_a',
                                                  'SIFKMRS'] = '1204'
        eopatch.vector_timeless[self.feature]['SIFKMRS'] = pd.to_numeric(
            eopatch.vector_timeless[self.feature]['SIFKMRS'])

    def _fix_austrian_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='SNAR_BEZEI')

    def _fix_danish_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature].rename(index=str, columns={"CropName": "crop_geopedia_idx"}, inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature]['crop_geopedia_idx'] = eopatch.vector_timeless[self.feature][
            'PreCropName']
        self.mapping.rename(index=str, columns={"Crop Name": "PreCrop Name"}, inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature].drop(['crop_geopedia_idx', 'PreCropName'], axis=1, inplace=True)

    def execute(self, eopatch):
        if self.country == 'Slovenia':
            self._fix_slovenian_lpis(eopatch)
        elif self.country == 'Austria':
            self._fix_austrian_lpis(eopatch)
        elif self.country == 'Denmark':
            self._fix_danish_lpis(eopatch)

        return eopatch


class AddGeopediaVectorFeature(EOTask):
    """
    Add vector data from Geopedia.
    """

    def __init__(self, feature, layer, year_filter=None, drop_duplicates=False):
        self.feature_type, self.feature_name = next(self._parse_features(feature)())
        self.layer = layer
        self.drop_duplicates = drop_duplicates
        self.year_col_name = year_filter[0] if year_filter is not None else None
        self.year = year_filter[1] if year_filter is not None else None

    def execute(self, eopatch):
        # convert to 3857 CRS
        bbox_3857 = eopatch.bbox.transform(CRS.POP_WEB)

        # get iterator over features
        gpd_iter = GeopediaFeatureIterator(layer=self.layer, bbox=bbox_3857)

        features = list(gpd_iter)
        if len(features):
            gdf = gpd.GeoDataFrame.from_features(features)
            gdf.crs = {'init': 'epsg:4326'}
            # convert back to EOPatch CRS
            gdf = gdf.to_crs({'init': f'epsg:{eopatch.bbox.crs.value}'})

            if self.year:
                # Filter by years
                gdf = gdf.loc[gdf[self.year_col_name].isin([self.year])]

            if self.drop_duplicates:
                sel = gdf.drop('geometry', axis=1)
                sel = sel.drop_duplicates()
                gdf = gdf.loc[sel.index]

            eopatch[self.feature_type][self.feature_name] = gdf

        return eopatch


class AddAreaRatio(EOTask):
    """
    Calculates the ratio between 
    
    area of all fields (vector data) / total area of the patch.
    
    This information can be used for example to exclude EOPatches with no or very small area of cultivated land.
    """

    def __init__(self, vector_feature, area_feature):
        self.in_feature_type, self.in_feature_name = next(self._parse_features(vector_feature)())
        self.out_feature_type, self.out_feature_name = next(self._parse_features(area_feature)())

    def execute(self, eopatch):
        ratio = np.array([-1.0])
        if self.in_feature_name not in eopatch[self.in_feature_type]:
            eopatch[self.out_feature_type][self.out_feature_name] = ratio
            return eopatch

        gdf = eopatch[self.in_feature_type][self.in_feature_name]
        ratio = np.array([0.0])
        if gdf is not None:
            bbox_poly = Polygon(eopatch.bbox.get_polygon())
            ratio = np.array([np.sum(gdf.area.values) / bbox_poly.area])

        eopatch[self.out_feature_type][self.out_feature_name] = ratio

        return eopatch


class Sen2CorValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a valid data mask.
    The valid data mask is post-processed (optional).

    The Sen2Cor's classification map is asumed to be found in eopatch.mask['SCL']
    """

    def __init__(self, valid_classes, erosion_radius=0, dilation_radius=0):
        self.valid = valid_classes
        self.erosion = erosion_radius
        self.dilation = dilation_radius

    def __call__(self, eopatch):
        sen2cor_valid = np.zeros_like(eopatch.mask['SCL'], dtype=np.bool)

        for valid in self.valid:
            sen2cor_valid = np.logical_or(sen2cor_valid, (eopatch.mask['SCL'] == valid))

        sen2cor_valid = sen2cor_valid.squeeze()
        if self.erosion:
            sen2cor_valid = np.logical_not(
                np.asarray([binary_erosion(np.logical_not(mask), disk(self.erosion)) for mask in sen2cor_valid],
                           dtype=np.bool))

        if self.dilation:
            sen2cor_valid = np.logical_not(
                np.asarray([binary_dilation(np.logical_not(mask), disk(self.dilation)) for mask in sen2cor_valid],
                           dtype=np.bool))

        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool), sen2cor_valid[..., np.newaxis])


class SentinelHubValidData:
    """
    Combine s2cloudless cloud map with `IS_DATA` to define a `VALID_DATA_SH` mask

    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))


class MergeMasks:
    """
    Merges two specified masks with logical and operation and returns it.
    """

    def __init__(self, mask_a, mask_b):
        self.mask_a = mask_a
        self.mask_b = mask_b

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask[self.mask_a].astype(np.bool),
                              eopatch.mask[self.mask_b].astype(np.bool))


class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what], axis=0))

        return eopatch
