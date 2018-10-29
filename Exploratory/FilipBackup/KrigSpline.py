import numpy as np
from eolearn.features import InterpolationTask
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor
from eolearn.core import EOTask, EOPatch, FeatureType

from dateutil import parser
from datetime import timedelta
from scipy import interpolate

import pdb

class PixelWiseInterpolationTask(InterpolationTask):


    def interpolate_data(self, data, times, resampled_times):
        """ Interpolates data feature

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :type data: numpy.ndarray
        :param times: Array of reference times relative to the first timestamp
        :type times: numpy.array
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :type resampled_times: numpy.array
        :return: Array of interpolated values
        :rtype: numpy.ndarray
        """
        # get size of 2d array t x nobs
        ntimes, nobs = data.shape

        # mask representing overlap between reference and resampled times
        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))

        # define time values as linear mono-tonically increasing over the observations
        const = int(self.filling_factor * (np.max(times) - np.min(times)))
        temp_values = (times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :]).astype(np.float64)
        res_temp_values = (resampled_times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :]).astype(np.float64)

        # initialise array of interpolated values
        new_data = data if self.resample_range is None else np.full((len(resampled_times), nobs),
                                                                    np.nan, dtype=data.dtype)
        # array defining index correspondence between reference times and resampled times
        ori2res = np.arange(ntimes, dtype=np.int32) if self.resample_range is None else np.array(
            [np.abs(resampled_times - o).argmin()
             if np.min(resampled_times) <= o <= np.max(resampled_times) else None for o in times])

        # find NaNs that start or end a time-series
        row_nans, col_nans = np.where(self._get_start_end_nans(data))
        nan_row_res_indices = np.array([index for index in ori2res[row_nans] if index is not None], dtype=np.int32)
        nan_col_res_indices = np.array([True if index is not None else False for index in ori2res[row_nans]],
                                       dtype=np.bool)
        if nan_row_res_indices.size:
            # mask out from output values the starting/ending NaNs
            res_temp_values[nan_row_res_indices, col_nans[nan_col_res_indices]] = np.nan
        # if temporal values outside the reference dates are required (extrapolation) masked them to NaN
        res_temp_values[~time_mask, :] = np.nan

        # build 1d array for interpolation. Spline functions require monotonically increasing values of x, so .T is used
        import pdb
        #pdb.set_trace()
        for obs in range(nobs):
            if not obs % 10000:
                print(obs)
            valid = ~np.isnan(data[:, obs])
            #valid_res_temp = ~np.isnan(data[:,obs])
            cur_x = times[valid]
            cur_y = data[valid, obs]
            #pdb.set_trace()
            obs_interpolating_func = self.get_interpolation_function(cur_x, cur_y)
            #pdb.set_trace()
            #res = obs_interpolating_func(res_temp_values[valid_res_temp,obs])
            res = obs_interpolating_func(resampled_times[:, np.newaxis])
            new_data[:, obs] = res

        #input_x = temp_values.T[~np.isnan(data).T]
        #input_y = data.T[~np.isnan(data).T]
        #pdb.set_trace()
        # build interpolation function
        #interp_func = self.get_interpolation_function(input_x, input_y)

        # interpolate non-NaN values in resampled time values
        #new_data[~np.isnan(res_temp_values)] = interp_func(res_temp_values[~np.isnan(res_temp_values)])

        # return interpolated values
        return new_data


class KrigingObject:

    def __init__(self, times, series, **kwargs):
        self.regressor = GaussianProcessRegressor(**kwargs)
        # Should normalize by max in resample time to be totally consistent, but this works fine (0.03% error in testing)
        self.normalizing_factor = max(times) - min(times)
        #print(times)
        #print(series)
        #print(self.normalizing_factor)
        self.regressor.fit(times.reshape(-1, 1)/self.normalizing_factor, series)
        self.call_args = kwargs.get("call_args", {})

    def __call__(self, X, **kwargs):
        call_args = self.call_args.copy()
        call_args.update(kwargs)
        #print(X)
        #print(self.regressor.predict(X.reshape(-1, 1)/self.normalizing_factor, **call_args))
        #import pdb
        #pdb.set_trace()

        return self.regressor.predict(X.reshape(-1, 1)/self.normalizing_factor, **call_args)


class KrigingInterpolation(PixelWiseInterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `sklearn.gaussian_process.GaussianProcessRegressor`
    """
    def __init__(self, feature, **kwargs):
        super().__init__(feature, KrigingObject, **kwargs)


class BSplineInterpolation(PixelWiseInterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.BSpline`
    """
    def __init__(self, feature, *, spline_degree=3, **kwargs):
        super().__init__(feature, interpolate.make_interp_spline, k=spline_degree, **kwargs)