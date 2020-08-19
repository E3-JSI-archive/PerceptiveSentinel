import datetime
import itertools as it

import numpy as np
from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.ml_tools.utilities import rolling_window


class AddStreamTemporalFeaturesTask(EOTask):
    # pylint: disable=too-many-instance-attributes
    """ Task that implements and adds to eopatch the spatio-temporal features proposed in [1].
    The features are added to the `data_timeless` attribute dictionary of eopatch.
    [1] Valero et. al. "Production of adynamic cropland mask by processing remote sensing
    image series at high temporal and spatial resolutions" Remote Sensing, 2016.
    """

    def __init__(self, data_feature=(FeatureType.DATA, 'NDVI'), data_index=None,
                 ndvi_feature_name=(FeatureType.DATA, 'NDVI'), mask_data=True, *,
                 features=None,
                #  max_val_feature='max_val', min_val_feature='min_val', mean_val_feature='mean_val',
                #  sd_val_feature='sd_val', diff_max_feature='diff_max', diff_min_feature='diff_min',
                #  diff_diff_feature='diff_diff', max_mean_feature='max_mean',
                #  max_mean_len_feature='max_mean_len', max_mean_surf_feature='max_mean_surf',
                #  pos_surf_feature='pos_surf', pos_len_feature='pos_len', pos_rate_feature='pos_rate',
                #  neg_surf_feature='neg_surf', neg_len_feature='neg_len', neg_rate_feature='neg_rate',
                #  pos_tran_feature='pos_tran', neg_tran_feature='neg_tran',
                 feature_name_prefix=None, window_size=2, interval_tolerance=0.1, base_surface_min=-1.,
                 ndvi_barren_soil_cutoff=0.1):
        """
        :param data_feature: Name of data feature with values that are considered. Default is `'NDVI'`
        :type data_feature: object
        :param data_index: Index of to be extracted from last dimension in `data_feature`. If None, last dimension of
            data array is assumed ot be of size 1 (e.g. as in NDVI). Default is `None`
        :type data_index: int
        :param ndvi_feature_name: Name of data feature with NDVI values for bare soil transition considerations.
        If None, soil transitions are not calculated and set as 0
        :type ndvi_feature_name: obj
        :param mask_data: Flag specifying whether to mask data with `'VALID_DATA'` mask. If `False`, the `'IS_DATA'`
                          mask is used
        :param max_val_feature: Name of feature with computed max
        :type max_val_feature: str
        :param min_val_feature: Name of feature with computed min
        :type min_val_feature: str
        :param mean_val_feature: Name of feature with computed mean
        :type mean_val_feature: str
        :param sd_val_feature: Name of feature with computed standard deviation
        :param sd_val_feature: str
        :param diff_max_feature: Name of feature with computed max difference in a temporal sliding window
        :param diff_max_feature: str
        :param diff_min_feature: Name of feature with computed min difference in a temporal sliding window
        :param diff_min_feature: str
        :param diff_diff_feature: Name of feature with computed difference of difference in a temporal sliding window
        :param diff_diff_feature: str
        :param max_mean_feature: Name of feature with computed max of mean in a sliding window
        :param max_mean_feature: str
        :param max_mean_len_feature: Name of feature with computed length of time interval corresponding to max_mean
        :param max_mean_len_feature: str
        :param max_mean_surf_feature: Name of feature with computed surface under curve corresponding to max_mean
        :param max_mean_surf_feature: str
        :param pos_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is positive
        :param pos_surf_feature: str
        :param pos_len_feature: Name of feature with computed length of time interval corresponding to pos_surf
        :param pos_len_feature: str
        :param pos_rate_feature: Name of feature with computed rate of change corresponding to pos_surf
        :param pos_rate_feature: str
        :param neg_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is negative
        :param neg_surf_feature: str
        :param neg_len_feature: Name of feature with computed length of time interval corresponding to neg_surf
        :param neg_len_feature: str
        :param neg_rate_feature: Name of feature with computed rate of change corresponding to neg_surf
        :param neg_rate_feature: str
        :param pos_tran_feature: Name of feature to be associated to computed feature of argmax values
        :param pos_tran_feature: str
        :param neg_tran_feature: Name of feature to be associated to computed feature of argmax values
        :param neg_tran_feature: str
        :param feature_name_prefix: String to be used as prefix in names for calculated features.
        Default: value of data_feature
        :param feature_name_prefix: str
        :param window_size: Size of sliding temporal window
        :param window_size: int
        :param interval_tolerance: Tolerance for calculation of max_mean family of data features
        :param interval_tolerance: float
        :param base_surface_min: Minimal base value for data, used to more accurately calculate surface under curve.
        Default for indices like values is -1.0.
        :param base_surface_min: float
        :param ndvi_barren_soil_cutoff: Cutoff for bare soil detection
        :type ndvi_barren_soil_cutoff: 0.1
        """
        # pylint: disable=too-many-locals
        self.data_feature = next(iter(self._parse_features(data_feature, default_feature_type=FeatureType.DATA)))
        self.data_index = data_index or 0
        self.mask_data = mask_data
        self.ndvi_feature_name = next(iter(self._parse_features(ndvi_feature_name,
                                                                default_feature_type=FeatureType.DATA)))

        if feature_name_prefix:
            self.feature_name_prefix = feature_name_prefix
            if not feature_name_prefix.endswith("_"):
                self.feature_name_prefix += "_"
        else:
            self.feature_name_prefix = data_feature + "_"

        # self.max_val_feature = self.feature_name_prefix + max_val_feature
        # self.min_val_feature = self.feature_name_prefix + min_val_feature
        # self.mean_val_feature = self.feature_name_prefix + mean_val_feature
        # self.sd_val_feature = self.feature_name_prefix + sd_val_feature
        # self.diff_max_feature = self.feature_name_prefix + diff_max_feature
        # self.diff_min_feature = self.feature_name_prefix + diff_min_feature
        # self.diff_diff_feature = self.feature_name_prefix + diff_diff_feature
        # self.max_mean_feature = self.feature_name_prefix + max_mean_feature
        # self.max_mean_len_feature = self.feature_name_prefix + max_mean_len_feature
        # self.max_mean_surf_feature = self.feature_name_prefix + max_mean_surf_feature
        # self.pos_surf_feature = self.feature_name_prefix + pos_surf_feature
        # self.pos_len_feature = self.feature_name_prefix + pos_len_feature
        # self.pos_rate_feature = self.feature_name_prefix + pos_rate_feature
        # self.neg_surf_feature = self.feature_name_prefix + neg_surf_feature
        # self.neg_len_feature = self.feature_name_prefix + neg_len_feature
        # self.neg_rate_feature = self.feature_name_prefix + neg_rate_feature
        # self.pos_transition_feature = self.feature_name_prefix + pos_transition_feature
        # self.neg_transition_feature = self.feature_name_prefix + neg_transition_feature

        if features:
            self.features = features
        else:
            self.features = AddStreamTemporalFeaturesTask.get_feature_names()

        self.window_size = window_size
        self.interval_tolerance = interval_tolerance
        self.base_surface_min = base_surface_min

        self.ndvi_barren_soil_cutoff = ndvi_barren_soil_cutoff

    def execute(self, eopatch):
        """ Compute spatio-temporal features for input eopatch
        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        if not self.features:
            return eopatch

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        data = eopatch[self.data_feature[0]][self.data_feature[1]][..., self.data_index]
        valid_data_mask = np.ones_like(data)

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data = {}

        for f in AddStreamTemporalFeaturesTask.get_feature_names():
            if f == 'max_val' and f in self.features:
                data[f] = np.ma.MaskedArray.max(madata, axis=0).filled()
            elif f == 'min_val' and f in self.features:
                data[f] = np.ma.MaskedArray.min(madata, axis=0).filled()
            elif f == 'mean_val' and f in self.features:
                data[f] = np.ma.MaskedArray.mean(madata, axis=0).filled()
            elif f == 'sd_val' and f in self.features:
                data[f] = np.ma.MaskedArray.std(madata, axis=0).filled()
            elif f not in ['diff_diff']:
                data[f] = np.empty((h, w))

        # data['max_val'] = np.ma.MaskedArray.max(madata, axis=0).filled()
        # data['min_val'] = np.ma.MaskedArray.min(madata, axis=0).filled()
        # data['mean_val'] = np.ma.MaskedArray.mean(madata, axis=0).filled()
        # data['sd_val'] = np.ma.MaskedArray.std(madata, axis=0).filled()

        # data_diff_max = np.empty((h, w))
        # data_diff_min = np.empty((h, w))
        # # data_diff_diff = np.empty((h, w)) # Calculated later

        # data_max_mean = np.empty((h, w))
        # data_max_mean_len = np.empty((h, w))
        # data_max_mean_surf = np.empty((h, w))

        # data_pos_surf = np.empty((h, w))
        # data_pos_len = np.empty((h, w))
        # data_pos_rate = np.empty((h, w))

        # data_neg_surf = np.empty((h, w))
        # data_neg_len = np.empty((h, w))
        # data_neg_rate = np.empty((h, w))

        # data_pos_tr = np.empty((h, w))
        # data_neg_tr = np.empty((h, w))
        for ih, iw in it.product(range(h), range(w)):
            data_curve = madata[:, ih, iw]
            valid_idx = np.where(~madata.mask[:, ih, iw])[0]

            data_curve = data_curve[valid_idx].filled()

            valid_dates = all_dates[valid_idx]

            if self.any_feature_selected(['diff_max', 'diff_min', 'diff_diff']):
                sw_max = np.max(rolling_window(data_curve, self.window_size), -1)
                sw_min = np.min(rolling_window(data_curve, self.window_size), -1)
                sw_diff = sw_max - sw_min

            if self.any_feature_selected(['diff_max', 'diff_diff']):
                data['diff_max'][ih, iw] = np.max(sw_diff)

            if self.any_feature_selected(['diff_min', 'diff_diff']):
                data['diff_min'][ih, iw] = np.min(sw_diff)

            if not self.any_feature_selected([
                'max_mean', 'max_mean_len', 'max_mean_surf',
                'pos_surf', 'pos_len', 'pos_rate',
                'neg_surf', 'neg_len', 'neg_rate',
                'pos_tran', 'neg_tran'
            ]):
                continue

            sw_mean = np.mean(rolling_window(data_curve, self.window_size), -1)
            max_mean = np.max(sw_mean)

            data['max_mean'][ih, iw] = max_mean

            # Calculate max mean interval
            # Work with mean windowed or whole set?
            workset = data_curve  # or sw_mean, which is a bit more smoothed
            higher_mask = workset >= max_mean - ((1 - self.interval_tolerance) * abs(max_mean))

            # Just normalize to have 0 on each side
            higher_mask_norm = np.zeros(len(higher_mask) + 2)
            higher_mask_norm[1:len(higher_mask)+1] = higher_mask

            # index of 1 that have 0 before them, SHIFTED BY ONE TO RIGHT
            up_mask = (higher_mask_norm[1:] == 1) & (higher_mask_norm[:-1] == 0)

            # Index of 1 that have 0 after them, correct indices
            down_mask = (higher_mask_norm[:-1] == 1) & (higher_mask_norm[1:] == 0)

            # Calculate length of interval as difference between times of first and last high enough observation,
            # in particular, if only one such observation is high enough, the length of such interval is 0
            # One can extend this to many more ways of calculating such length:
            # take forward/backward time differences, interpolate in between (again...) and treat this as
            # continuous problem, take mean of the time intervals between borders...
            times_up = valid_dates[up_mask[:-1]]
            times_down = valid_dates[down_mask[1:]]

            # There may be several such intervals, take the longest one
            times_diff = times_down - times_up
            # if there are no such intervals, the signal is constant,
            # set everything to zero and continue
            if times_diff.size == 0:
                data['max_mean_len'][ih, iw] = 0
                data['max_mean_surf'][ih, iw] = 0

                data['pos_surf'][ih, iw] = 0
                data['pos_len'][ih, iw] = 0
                data['pos_rate'][ih, iw] = 0

                data['neg_surf'][ih, iw] = 0
                data['neg_len'][ih, iw] = 0
                data['neg_rate'][ih, iw] = 0

                if self.ndvi_feature_name:
                    data['pos_tran'][ih, iw] = 0
                    data['neg_tran'][ih, iw] = 0

                continue

            if self.any_feature_selected(['max_mean_len', 'max_mean_surf']):
                max_ind = np.argmax(times_diff)
            
            if 'max_mean_len' in self.features:
                data['max_mean_len'][ih, iw] = times_diff[max_ind]

            if 'max_mean_surf' in self.features:
                fst = np.where(up_mask[:-1])[0]
                snd = np.where(down_mask[1:])[0]

                surface = np.trapz(data_curve[fst[max_ind]:snd[max_ind]+1] - self.base_surface_min,
                                valid_dates[fst[max_ind]:snd[max_ind]+1])
                data['max_mean_surf'][ih, iw] = surface

            # Derivative based features
            # How to approximate derivative?
            if self.any_feature_selected([
                'pos_surf', 'pos_len', 'pos_rate',
                'neg_surf', 'neg_len', 'neg_rate'
            ]):
                derivatives = np.gradient(data_curve, valid_dates)

            # Positive derivative
            if self.any_feature_selected([
                'pos_surf', 'pos_len', 'pos_rate'
            ]):
                pos = np.zeros(len(derivatives) + 2)
                pos[1:len(derivatives)+1] = derivatives >= 0

                pos_der_int, pos_der_len, pos_der_rate, (start, _) = \
                    self.derivative_features(pos, valid_dates, data_curve, self.base_surface_min)

                data['pos_surf'][ih, iw] = pos_der_int
                data['pos_len'][ih, iw] = pos_der_len
                data['pos_rate'][ih, iw] = pos_der_rate

            # Negative derivative
            if self.any_feature_selected([
                'neg_surf', 'neg_len', 'neg_rate'
            ]):
                neg = np.zeros(len(derivatives) + 2)
                neg[1:len(derivatives)+1] = derivatives <= 0

                neg_der_int, neg_der_len, neg_der_rate, (_, end) = \
                    self.derivative_features(neg, valid_dates, data_curve, self.base_surface_min)

                data['neg_surf'][ih, iw] = neg_der_int
                data['neg_len'][ih, iw] = neg_der_len
                data['neg_rate'][ih, iw] = neg_der_rate

            if self.ndvi_feature_name:
                if 'pos_tran' in self.features:
                    data['pos_tran'][ih, iw] = \
                        np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][:start+1, ih, iw, 0] <=
                            self.ndvi_barren_soil_cutoff)
                if 'neg_tran' in self.features:
                    data['neg_tran'][ih, iw] = \
                        np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][end:, ih, iw, 0] <=
                            self.ndvi_barren_soil_cutoff)

        for feature in self.features:
            feature_name = f'{self.feature_name_prefix}{feature}'
            if feature == 'diff_diff':
                eopatch.data_timeless[feature_name] = (data['diff_max'] - data['diff_min'])[..., np.newaxis]
            else:
                eopatch.data_timeless[feature_name] = data[feature][..., np.newaxis]

        # eopatch.data_timeless[self.max_val_feature] = data_max_val[..., np.newaxis]
        # eopatch.data_timeless[self.min_val_feature] = data_min_val[..., np.newaxis]
        # eopatch.data_timeless[self.mean_val_feature] = data_mean_val[..., np.newaxis]
        # eopatch.data_timeless[self.sd_val_feature] = data_sd_val[..., np.newaxis]

        # eopatch.data_timeless[self.diff_max_feature] = data_diff_max[..., np.newaxis]
        # eopatch.data_timeless[self.diff_min_feature] = data_diff_min[..., np.newaxis]
        # eopatch.data_timeless[self.diff_diff_feature] = (data_diff_max - data_diff_min)[..., np.newaxis]

        # eopatch.data_timeless[self.max_mean_feature] = data_max_mean[..., np.newaxis]
        # eopatch.data_timeless[self.max_mean_len_feature] = data_max_mean_len[..., np.newaxis]
        # eopatch.data_timeless[self.max_mean_surf_feature] = data_max_mean_surf[..., np.newaxis]

        # eopatch.data_timeless[self.pos_len_feature] = data_pos_len[..., np.newaxis]
        # eopatch.data_timeless[self.pos_surf_feature] = data_pos_surf[..., np.newaxis]
        # eopatch.data_timeless[self.pos_rate_feature] = data_pos_rate[..., np.newaxis]
        # eopatch.data_timeless[self.pos_transition_feature] = data_pos_tr[..., np.newaxis]

        # eopatch.data_timeless[self.neg_len_feature] = data_neg_len[..., np.newaxis]
        # eopatch.data_timeless[self.neg_surf_feature] = data_neg_surf[..., np.newaxis]
        # eopatch.data_timeless[self.neg_rate_feature] = data_neg_rate[..., np.newaxis]
        # eopatch.data_timeless[self.neg_transition_feature] = data_neg_tr[..., np.newaxis]

        return eopatch

    def get_data(self, patch):
        """Extracts and concatenates newly extracted features contained in the provided eopatch
        :param patch: Input eopatch
        :type patch: eolearn.core.EOPatch
        :return: Tuple of two lists: names of extracted features and their values
        """
        # names = [self.max_val_feature, self.min_val_feature, self.mean_val_feature, self.sd_val_feature,
        #          self.diff_max_feature, self.diff_min_feature, self.diff_diff_feature,
        #          self.max_mean_feature, self.max_mean_len_feature, self.max_mean_surf_feature,
        #          self.pos_len_feature, self.pos_surf_feature, self.pos_rate_feature, self.pos_transition_feature,
        #          self.neg_len_feature, self.neg_surf_feature, self.neg_rate_feature, self.neg_transition_feature]

        names = AddStreamTemporalFeaturesTask.get_feature_names()

        dim_x, dim_y, _ = patch.data_timeless[names[0]].shape

        data = np.zeros((dim_x, dim_y, len(names)))
        for ind, name in enumerate(names):
            data[..., ind] = patch.data_timeless[name].squeeze()

        return names, data

    def any_feature_selected(self, features):
        for feature in features:
            if feature in self.features:
                return True

        return False

    @staticmethod
    def get_feature_names():
        return [
            'max_val', 'min_val', 'mean_val', 'sd_val',
            'diff_max', 'diff_min', 'diff_diff',
            'max_mean', 'max_mean_len', 'max_mean_surf',
            'pos_surf', 'pos_len', 'pos_rate',
            'neg_surf', 'neg_len', 'neg_rate',
            'pos_tran', 'neg_tran'
        ]

    @staticmethod
    def derivative_features(mask, valid_dates, data, base_surface_min):
        """Calculates derivative based features for provided data points selected by
        mask (increasing data points, decreasing data points)
        :param mask: Mask indicating data points considered
        :type mask: np.array
        :param valid_dates: Dates (x-axis for surface calculation)
        :type valid_dates: np.array
        :param data: Base data
        :type data: np.array
        :param base_surface_min: Base surface value (added to each measurement)
        :type base_surface_min: float
        :return: Tuple of: maximal consecutive surface under the data curve,
                           date length corresponding to maximal surface interval,
                           rate of change in maximal interval,
                           (starting date index of maximal interval, ending date index of interval)
        """
        # index of 1 that have 0 before them, shifted by one to right
        up_mask = (mask[1:] == 1) & (mask[:-1] == 0)

        # Index of 1 that have 0 after them, correct indices
        down_mask = (mask[:-1] == 1) & (mask[1:] == 0)

        fst_der = np.where(up_mask[:-1])[0]
        snd_der = np.where(down_mask[1:])[0]
        der_ind_max = -1
        der_int_max = -1

        for ind, (start, end) in enumerate(zip(fst_der, snd_der)):

            integral = np.trapz(
                data[start:end + 1] - base_surface_min,
                valid_dates[start:end + 1])

            if abs(integral) >= abs(der_int_max):
                der_int_max = integral
                der_ind_max = ind

        start_ind = fst_der[der_ind_max]
        end_ind = snd_der[der_ind_max]

        der_len = valid_dates[end_ind] - valid_dates[start_ind]
        der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0

        return der_int_max, der_len, der_rate, (start_ind, end_ind)
