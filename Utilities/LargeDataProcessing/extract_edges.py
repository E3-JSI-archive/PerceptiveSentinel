import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk, EOExecutor

import cv2
from scipy.ndimage.measurements import label
import os


class ExtractEdgesTask(EOTask):

    def __init__(self,
                 edge_features,
                 structuring_element,
                 excluded_features,
                 dilation_mask,
                 erosion_mask,
                 output_feature,
                 adjust_function,
                 adjust_threshold,
                 yearly_low_threshold):

        self.edge_features = edge_features
        self.structuring_element = structuring_element
        self.excluded_features = excluded_features
        self.dilation_mask = dilation_mask
        self.erosion_mask = erosion_mask
        self.output_feature = output_feature
        self.adjust_function = adjust_function
        self.adjust_threshold = adjust_threshold
        self.yearly_low_threshold = yearly_low_threshold

    def extract_edges(self, eopatch, feature_type, feature_name, low_threshold, high_threshold, blur):

        image = eopatch[feature_type][feature_name]
        t, w, h, _ = image.shape
        all_edges = np.zeros((t, w, h))
        for time in range(t):
            image_one = image[time]
            edge = self.one_edge(image_one, low_threshold, high_threshold, blur)
            all_edges[time] = edge
        eopatch.add_feature(FeatureType.MASK, feature_name + '_EDGE', all_edges[..., np.newaxis])
        return all_edges

    def one_edge(self, image, low_threshold, high_threshold, blur):
        ##########QUICK NORMALIZATION -  SHOULD BE LATER IMPROVED / MOVED SOMEWHERE ELSE
        f_min = np.min(image)
        f_max = np.max(image)
        image = (image - f_min) / f_max * 255
        image = image.squeeze()
        kernel_size, sigma = blur
        smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)
        edges = cv2.Canny(smoothed_image.astype(np.uint8), low_threshold, high_threshold)
        return edges > 0

    def filter_unwanted_areas(self, eopatch, feature, threshold):
        # Returns mask of areas that should be excluded (low NDVI etc...)
        bands = eopatch[feature[0]][feature[1]]
        t, w, h, _ = bands.shape
        mask = np.zeros((w, h))

        for time in range(t):
            fet = eopatch[feature[0]][feature[1]][time].squeeze()
            mask_cur = fet <= threshold
            mask_cur = cv2.dilate((mask_cur * 255).astype(np.uint8), self.dilation_mask * 255)
            mask_cur = cv2.erode((mask_cur * 255).astype(np.uint8), self.erosion_mask * 255)
            mask = mask + mask_cur

        mask = (mask / t) > self.yearly_low_threshold
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'LOW_' + feature[1], mask[..., np.newaxis])
        return mask

    def normalize_feature(self, feature):
        f_min = np.min(feature)
        f_max = np.max(feature)
        return (feature - f_min) / (f_max - f_min)

    def execute(self, eopatch):

        bands = eopatch.data['BANDS']
        t, w, h, _ = bands.shape

        no_feat = len(self.edge_features)
        edge_vector = np.zeros((no_feat, t, w, h))
        for i in range(no_feat):
            arg = self.edge_features[i]
            one_edge = self.extract_edges(eopatch, arg['FeatureType'], arg['FeatureName'],
                                          arg['CannyThresholds'][0], arg['CannyThresholds'][1], arg['BlurArguments'])
            v1 = eopatch[arg['FeatureType']][arg['FeatureName']].squeeze()
            v1 = self.normalize_feature(v1)
            v1 = [self.adjust_function(x) for x in v1]
            edge_vector[i] = one_edge * v1

        edge_vector1 = np.sum(edge_vector, (0, 1))
        edge_vector1 = edge_vector1 / (t * len(self.edge_features))
        edge_vector = edge_vector1 > self.adjust_threshold

        for unwanted, threshold in self.excluded_features:
            mask = self.filter_unwanted_areas(eopatch, unwanted, threshold)

            edge_vector = np.logical_or(edge_vector, mask)

        edge_vector = 1 - edge_vector
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.output_feature[1], edge_vector[..., np.newaxis])
        return eopatch


# Example
if __name__ == '__main__':

    b_low = 10
    b_high = 40

    structuring_2d = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]
                      ]

    segmentation = ExtractEdgesTask(
        edge_features=[
            {"FeatureType": FeatureType.DATA,
             "FeatureName": 'EVI',
             "CannyThresholds": (40, 80),
             "BlurArguments": ((5, 5), 2)
             },

            {"FeatureType": FeatureType.DATA,
             "FeatureName": 'ARVI',
             "CannyThresholds": (40, 80),
             "BlurArguments": ((5, 5), 2)
             },
            {"FeatureType": FeatureType.DATA,
             "FeatureName": 'NDVI',
             "CannyThresholds": (40, 100),
             "BlurArguments": ((5, 5), 2)
             },
            {"FeatureType": FeatureType.DATA,
             "FeatureName": 'GRAY',
             "CannyThresholds": (5, 40),
             "BlurArguments": ((3, 3), 2)
             }
        ],
        structuring_element=structuring_2d,
        excluded_features=[((FeatureType.DATA, 'NDVI'), 0.3)],
        dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        output_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
        adjust_function=lambda x: cv2.GaussianBlur(x, (9, 9), 5),
        adjust_threshold=0.05,
        yearly_low_threshold=0.8)


    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


    class Preprocess(EOTask):

        def execute(self, eopatch):
            img = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]] * 3.5, 0, 1)
            t, w, h, _ = img.shape
            gray_img = np.zeros((t, w, h))
            print(img[0].shape)
            for time in range(t):
                img0 = np.clip(eopatch[FeatureType.DATA]['BANDS'][time][..., [2, 1, 0]] * 3.5, 0, 1)
                img = rgb2gray(img0)
                gray_img[time] = (img * 255).astype(np.uint8)

            eopatch.add_feature(FeatureType.DATA, 'GRAY', gray_img[..., np.newaxis])
            print(eopatch)
            return eopatch


    no_patches = 1

    path = '/home/beno/Documents/test'
    # path = 'E:/Data/PerceptiveSentinel'

    patch_location = path + '/Slovenia/'
    load = LoadFromDisk(patch_location, lazy_loading=True)

    save_path_location = path + '/Slovenia/'
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)
    save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    execution_args = []
    for id in range(no_patches):
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(id)},
            save: {'eopatch_folder': 'eopatch_{}'.format(id)}
        })

    workflow = LinearWorkflow(
        load,
        Preprocess(),
        segmentation,
        save
    )

    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    # here you choose how many processes/threads you will run, workers=none is max of processors
    executor.run(workers=None, multiprocess=False)

    # display()
