import itertools
import time
import warnings

import numpy as np

from eolearn.core import EOTask

from IPython.core.debugger import Tracer
def rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def derivative_features(mask, valid_dates, data, base_surface_min):
    # index of 1 that have 0 before them, SHIFTED BY ONE TO RIGHT
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
            valid_dates[start:end+1])
        if abs(integral) >= abs(der_int_max):
            der_int_max = integral
            der_ind_max = ind
    if der_ind_max == -1:
        return 0, 0, 0, (0, len(mask))
    start_ind = fst_der[der_ind_max]
    end_ind = snd_der[der_ind_max]

    der_len = valid_dates[end_ind] - valid_dates[start_ind]
    der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0

    return der_int_max, der_len, der_rate, (start_ind, end_ind)


warnings.simplefilter('error', UserWarning)


class AddTemporalDataFeaturesTask(EOTask):
    """ Task that implements and adds to eopatch the spatio-temporal features proposed in [1].


    The features are added to the `data_timeless` attribute dictionary of eopatch.

    [1] Valero etal. "Production of adynamic cropland mask by processing remote sensingimage series at high temporal and
    spatial resolutions" Remote Sensing, 2016.

    """

    def __init__(self, data_feature='NDVI', data_index=None, ndvi_feature_name='NDVI', mask_data=True, *,
                 max_val_feature='max_val', min_val_feature='min_val', mean_val_feature='mean_val',
                 sd_val_feature='sd_val', diff_max_feature='diff_max', diff_min_feature='diff_min',
                 diff_diff_feature='diff_diff', max_mean_feature='max_mean_feature',
                 max_mean_len_feature='max_mean_len', max_mean_surf_feature='max_mean_surf',
                 pos_surf_feature='pos_surf', pos_len_feature='pos_len', pos_rate_feature='pos_rate',
                 neg_surf_feature='neg_surf', neg_len_feature='neg_len', neg_rate_feature='neg_rate',
                 pos_transition_feature='pos_tran', neg_transition_feature='neg_tran',
                 feature_name_prefix=None, window_size=2, interval_tolerance=0.1, base_surface_min=-1.,
                 ndvi_barren_soil_cutoff=0.1):
        """
        :param data_feature: Name of data feature with values that are considered. Default is `'NDVI'`
        :type data_feature: str
        :param data_index: Index of to be extracted from last dimension in `data_feature`. If None, last dimension of
            data array is assumed ot be of size 1 (e.g. as in NDVI). Default is `None`
        :type data_index: int
        :param ndvi_feature_name: Name of data feature with NDVI values for bare soil transition considerations.
        If None, soil transitions are not calculated and set as 0
        :type ndvi_feature_name: str
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
        :param pos_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param pos_transition_feature: str
        :param neg_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param neg_transition_feature: str
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

        self.data_feature = data_feature
        self.data_index = data_index or 0
        self.mask_data = mask_data
        self.ndvi_feature_name = ndvi_feature_name

        if feature_name_prefix:
            self.feature_name_prefix = feature_name_prefix
            if not feature_name_prefix.endswith("_"):
                self.feature_name_prefix += "_"
        else:
            self.feature_name_prefix = data_feature + "_"

        self.max_val_feature = self.feature_name_prefix + max_val_feature
        self.min_val_feature = self.feature_name_prefix + min_val_feature
        self.mean_val_feature = self.feature_name_prefix + mean_val_feature
        self.sd_val_feature = self.feature_name_prefix + sd_val_feature
        self.diff_max_feature = self.feature_name_prefix + diff_max_feature
        self.diff_min_feature = self.feature_name_prefix + diff_min_feature
        self.diff_diff_feature = self.feature_name_prefix + diff_diff_feature
        self.max_mean_feature = self.feature_name_prefix + max_mean_feature
        self.max_mean_len_feature = self.feature_name_prefix + max_mean_len_feature
        self.max_mean_surf_feature = self.feature_name_prefix + max_mean_surf_feature
        self.pos_surf_feature = self.feature_name_prefix + pos_surf_feature
        self.pos_len_feature = self.feature_name_prefix + pos_len_feature
        self.pos_rate_feature = self.feature_name_prefix + pos_rate_feature
        self.neg_surf_feature = self.feature_name_prefix + neg_surf_feature
        self.neg_len_feature = self.feature_name_prefix + neg_len_feature
        self.neg_rate_feature = self.feature_name_prefix + neg_rate_feature
        self.pos_transition_feature = self.feature_name_prefix + pos_transition_feature
        self.neg_transition_feature = self.feature_name_prefix + neg_transition_feature

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
        if self.mask_data:
            valid_data_mask = eopatch.mask[self.mask_data]
        else:
            valid_data_mask = eopatch.mask['IS_DATA']

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        data = eopatch.data[self.data_feature][..., self.data_index]

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data_max_val = np.ma.MaskedArray.max(madata, axis=0).compressed().reshape((h, w))
        data_min_val = np.ma.MaskedArray.min(madata, axis=0).compressed().reshape((h, w))
        data_diff_val = data_max_val - data_min_val
        data_mean_val = np.ma.MaskedArray.mean(madata, axis=0).compressed().reshape((h, w))
        data_sd_val = np.ma.MaskedArray.std(madata, axis=0).compressed().reshape((h, w))
        #Tracer()()
        data_diff_max = np.empty((h, w))
        data_diff_min = np.empty((h, w))
        # data_diff_diff = np.empty((h, w)) # Calculated later

        data_max_mean = np.empty((h, w))
        data_max_mean_len = np.empty((h, w))
        data_max_mean_surf = np.empty((h, w))

        data_pos_surf = np.empty((h, w))
        data_pos_len = np.empty((h, w))
        data_pos_rate = np.empty((h, w))

        data_neg_surf = np.empty((h, w))
        data_neg_len = np.empty((h, w))
        data_neg_rate = np.empty((h, w))

        data_pos_tr = np.empty((h, w))
        data_neg_tr = np.empty((h, w))
        print("WORKING")
        t = time.time()
        tt = t
        for ih, iw in itertools.product(range(h), range(w)):
            if iw == 0:
                print(ih, time.time() - tt)
                tt = time.time()
            data_curve = madata[:, ih, iw]
            valid_idx = np.where(~madata.mask[:, ih, iw])[0]
            # Get rid of mask
            data_curve = data_curve[valid_idx].compressed()
            #assert not np.any(np.isnan(data_curve))
            valid_dates = all_dates[valid_idx]
            sw_max = np.max(rolling_window(data_curve, self.window_size), -1)
            sw_min = np.min(rolling_window(data_curve, self.window_size), -1)

            sw_diff = sw_max - sw_min

            data_diff_max[ih, iw] = np.max(sw_diff)
            data_diff_min[ih, iw] = np.min(sw_diff)

            sw_mean = np.array(np.mean(rolling_window(data_curve, self.window_size), -1))

            max_mean = np.max(sw_mean)

            data_max_mean[ih, iw] = max_mean

            # Calculate max mean interval
            # Work with mean windowed or whole set?
            workset = data_curve  # or sw_mean, which is a bit more smoothed
            higher_mask = workset >= (max_mean - (self.interval_tolerance * data_diff_val[ih, iw]))

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

            # There may be several such intervals (which is highly unlikely), take the longest one
            times_diff = times_down - times_up
            max_ind = np.argmax(times_diff)

            data_max_mean_len[ih, iw] = times_diff[max_ind]

            fst = np.where(up_mask[:-1])[0]
            snd = np.where(down_mask[1:])[0]

            # assert valid_dates[snd[max_ind]+1] - valid_dates[fst[max_ind]] == times_diff[max_ind]

            surface = np.trapz(data_curve[fst[max_ind]:snd[max_ind]+1] - self.base_surface_min,
                               valid_dates[fst[max_ind]:snd[max_ind]+1])

            data_max_mean_surf[ih, iw] = surface

            # Derivative based features
            # How to approximate derivative?
            derivatives = np.gradient(data_curve, valid_dates)

            # Positive derivative
            pos = np.zeros(len(derivatives) + 2)
            pos[1:len(derivatives)+1] = derivatives >= 0

            pos_der_int, pos_der_len, pos_der_rate, (start, _) = derivative_features(pos, valid_dates, data_curve,
                                                                                     self.base_surface_min)

            data_pos_surf[ih, iw] = pos_der_int
            data_pos_len[ih, iw] = pos_der_len
            data_pos_rate[ih, iw] = pos_der_rate

            neg = np.zeros(len(derivatives) + 2)
            neg[1:len(derivatives)+1] = derivatives <= 0

            neg_der_int, neg_der_len, neg_der_rate, (_, end) = derivative_features(neg, valid_dates, data_curve,
                                                                                   self.base_surface_min)

            data_neg_surf[ih, iw] = neg_der_int
            data_neg_len[ih, iw] = neg_der_len
            data_neg_rate[ih, iw] = neg_der_rate

            if self.ndvi_feature_name:
                data_pos_tr[ih, iw] = np.any(eopatch.data[self.ndvi_feature_name][:start+1,ih, iw, 0] <= self.ndvi_barren_soil_cutoff)
                data_neg_tr[ih, iw] = np.any(eopatch.data[self.ndvi_feature_name][end:,ih, iw, 0] <= self.ndvi_barren_soil_cutoff)
        #Tracer()()
        eopatch.data_timeless[self.max_val_feature] = data_max_val[..., np.newaxis]
        eopatch.data_timeless[self.min_val_feature] = data_min_val[..., np.newaxis]
        eopatch.data_timeless[self.mean_val_feature] = data_mean_val[..., np.newaxis]
        eopatch.data_timeless[self.sd_val_feature] = data_sd_val[..., np.newaxis]

        eopatch.data_timeless[self.diff_max_feature] = data_diff_max[..., np.newaxis]
        eopatch.data_timeless[self.diff_min_feature] = data_diff_min[..., np.newaxis]
        eopatch.data_timeless[self.diff_diff_feature] = (data_diff_max - data_diff_min)[..., np.newaxis]

        eopatch.data_timeless[self.max_mean_feature] = data_max_mean[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_len_feature] = data_max_mean_len[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_surf_feature] = data_max_mean_surf[..., np.newaxis]

        eopatch.data_timeless[self.pos_len_feature] = data_pos_len[..., np.newaxis]
        eopatch.data_timeless[self.pos_surf_feature] = data_pos_surf[..., np.newaxis]
        eopatch.data_timeless[self.pos_rate_feature] = data_pos_rate[..., np.newaxis]
        eopatch.data_timeless[self.pos_transition_feature] = data_pos_tr[..., np.newaxis]

        eopatch.data_timeless[self.neg_len_feature] = data_neg_len[..., np.newaxis]
        eopatch.data_timeless[self.neg_surf_feature] = data_neg_surf[..., np.newaxis]
        eopatch.data_timeless[self.neg_rate_feature] = data_neg_rate[..., np.newaxis]
        eopatch.data_timeless[self.neg_transition_feature] = data_neg_tr[..., np.newaxis]
        print(time.time() - t)
        return eopatch
