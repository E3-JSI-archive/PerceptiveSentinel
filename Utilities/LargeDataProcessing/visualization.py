import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Sampling import sample_patches
from PIL import Image


def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape
    print(image.shape)
    if colors is None:
        labels = np.max(image)
        labels = 0 if np.isnan(labels) else int(labels)
        colors = np.array([[0, 0, 0]])
        for _ in range(labels + 40):
            n_col = np.array([[random.randint(15, 255), random.randint(15, 255), random.randint(15, 255)]])
            colors = np.concatenate((colors, n_col), axis=0)

    new_image = np.zeros((w, h, 3))
    for x in range(w):
        for y in range(h):
            a = image[x][y]
            a = 0 if np.isnan(a) else a + 1
            c = colors[int(a)]
            new_image[x][y] = c
            # new_image[x][y] = colors[image[x][y]]

    return new_image / 255


def display():
    path = '/home/beno/Documents/test'
    # path = 'E:/Data/PerceptiveSentinel'
    patch_no = 2
    eopatch = EOPatch.load(path + '/Slovenia/eopatch_{}'.format(patch_no), lazy_loading=True)
    # print(eopatch)
    # print(eopatch.mask_timeless['LPIS_2017'].squeeze())

    # plt.subplot(2, 3, 1)
    #seg = eopatch.mask_timeless['EDGES_INV'].squeeze()
    # plt.imshow(seg)
    # plt.show()
    # print(seg.shape)
    # plt.imshow(seg, cmap='gray')

    # plt.subplot(2, 3, 2)
    #seg1 = eopatch.mask_timeless['LOW_NDVI'].squeeze()
    #print(seg.shape)
    # plt.imshow(seg1, cmap='gray')

    # plt.subplot(2, 3, 3)
    #seg3 = eopatch.mask_timeless['LPIS_2017'].squeeze()
    # plt.imshow(seg)
    # plt.show()
    # print(seg.shape)
    # plt.imshow(color_patch(seg3))

    # plt.subplot(2, 3, 4)
    # seg = eopatch.mask_timeless['LPIS_2017'].squeeze()
    # plt.imshow(seg)
    # plt.show()
    # print(seg.shape)
    # plt.imshow(Image.blend(Image.fromarray(color_patch(seg)), Image.fromarray(seg), alpha=0.5))
    elevation = eopatch.data_timeless['DEM'].squeeze()
    incline = eopatch.data_timeless['INCLINE'].squeeze()

    cmap = matplotlib.colors.ListedColormap(np.random.rand(23, 3))
    n_time = 5
    image = np.clip(eopatch.data['BANDS'][n_time][..., [3, 2, 1]] * 3.5, 0, 1)
    #mask = np.squeeze(eopatch.mask_timeless['LPIS_2017'])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 15))
    ax0.imshow(image)
    ax1.imshow(elevation)
    ax2.imshow(incline)
    # print(image)
    # seg = seg*255
    # image[:, :, 0] = image[:, :, 0] * seg
    # image[:, :, 1] = image[:, :, 1] * seg
    # image[:, :, 2] = image[:, :, 2] * seg

    # ax1.imshow(seg, cmap='gray')
    # ax1.imshow(mask, cmap=cmap, alpha=0.8)
    # ax1.imshow(seg, cmap='gray', alpha=0.2)

    # path = '/home/beno/Documents/test/Slovenia'
    #
    # no_patches = patch_no + 1
    # no_samples = 10000
    # class_feature = (FeatureType.MASK_TIMELESS, 'LPIS_2017')
    # mask = (FeatureType.MASK_TIMELESS, 'EDGES_INV')
    # features = [(FeatureType.DATA_TIMELESS, 'NDVI_mean_val'), (FeatureType.DATA_TIMELESS, 'SAVI_max_val'),
    #             (FeatureType.DATA_TIMELESS, 'NDVI_pos_surf')]
    # samples_per_class = 10
    # debug = True
    #
    # samples = sample_patches(path, no_patches, no_samples, class_feature, mask, features, samples_per_class, debug)
    # print(samples)
    # for index, row in samples.iterrows():
    #    ax1.plot(row['x'], row['y'], 'ro', alpha=1)

    plt.show()


if __name__ == '__main__':
    display()
