from eolearn.io import SentinelHubDemTask
from eolearn.core import LoadTask, SaveTask, OverwritePermission, LinearWorkflow, FeatureType, EOExecutor, EOTask
import os
from scipy import ndimage
import numpy as np


class AddGradientTask(EOTask):
    def __init__(self, elevation_feature, result_feature):
        self.feature = elevation_feature
        self.result_feature = result_feature

    def execute(self, eopatch):
        elevation = eopatch[self.feature[0]][self.feature[1]].squeeze()
        gradient = ndimage.gaussian_gradient_magnitude(elevation, 1)
        eopatch.add_feature(self.result_feature[0], self.result_feature[1], gradient[..., np.newaxis])

        return eopatch


if __name__ == '__main__':
    # path = 'E:/Data/PerceptiveSentinel'
    path = '/home/beno/Documents/test/Slovenia/'
    size_small = (337, 333)
    size_big = (505, 500)

    load = LoadTask(path, lazy_loading=True)
    save_path_location = path
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)
    save = SaveTask(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    dem = SentinelHubDemTask((FeatureType.DATA_TIMELESS, 'DEM'), size=size_big)
    grad = AddGradientTask((FeatureType.DATA_TIMELESS, 'DEM'), (FeatureType.DATA_TIMELESS, 'INCLINATION'))

    workflow = LinearWorkflow(
        load,
        dem,
        grad,
        save
    )

    no_patches = 1061

    execution_args = []
    for i in range(no_patches):
        i = i + 2
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(i)},
            save: {'eopatch_folder': 'eopatch_{}'.format(i)},
        })
    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    executor.run()



