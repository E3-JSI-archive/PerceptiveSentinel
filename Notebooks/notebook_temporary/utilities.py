import datetime
import math
import os
import string

import numpy as np
import pandas as pd
from eolearn.core import EOPatch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sentinelhub import GeopediaFeatureIterator, GeopediaSession
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm.notebook import tqdm


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
    values = crop_group_df[crop_group_df[group_name]==crop_group][group_id].values
    if len(values)==0:
        return default_value
    else:
        return values[-1]


def get_eopatch_ids(id_bottom_left, id_top_right, bbox_grid):
    """Generate 2D array of eopatch IDs from bottom-left and top-right eopatch IDs.
    IDs are identified from eo-patch bounding box grid provided in the form of
    geopandas data frame.

    :param id_bottom_left: Bottom-left eopatch ID
    :type id_bottom_left: int
    :param id_top_right: Top-right eopatch ID
    :type id_top_right: int
    :param bbox_grid: Grid of eopatch bounding boxes in Geopandas data frame format
    :type bbox_grid: DataFrame
    :return: 2D array of Eopatch IDs
    :rtype: numpy.array
    """
    coords_bl = list(bbox_grid.loc[id_bottom_left, ['index_x', 'index_y']])
    coords_tr = list(bbox_grid.loc[id_top_right, ['index_x', 'index_y']])

    ids = []
    for y in range(coords_tr[1], coords_bl[1] - 1, -1):
        row = []
        for x in range(coords_bl[0], coords_tr[0] + 1):
            indices = list(bbox_grid[bbox_grid['index_x'] == x][bbox_grid['index_y'] == y].index)
            row.append(indices[0] if len(indices) else None)
        ids.append(row)

    return np.array(ids)


def plot_grid(id_grid, patch_dir, img_func, img_func_args={}, imshow_args={}, title=None, colorbar=None, size=20):
    """Plot grid of eopatches defined by grid of eopatch IDs and plotting function.

    :param id_grid: 2D array of Eopatch IDs
    :type id_grid: numpy.array
    :param patch_dir: Path to the directory where eopatches are stored.
        Eopatch directory names should be formated as `eopatch_N`, where `N` is eopatch ID.
    :type patch_dir: string
    :param img_func: Function that prepares image data from eopatch
    :type img_func: function
    :param img_func_args: `func` arguments, defaults to {}
    :type img_func_args: dict, optional
    :param imshow_args: `imshow_args` arguments, defaults to {}
    :type imshow_args: dict, optional
    :param title: Suptitle, defaults to None
    :type title: str, optional
    :param colorbar: Colorbar settings, i.e. `ticks` and `labels`, defaults to None
    :type colorbar: dict or None, optional
    :param size: Plot size in inches
    :type size: int
    """
    grid_shape = id_grid.shape
    fig, axs = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1])
    patch_dirs = [os.path.join(patch_dir, f'eopatch_{patch_id}') for patch_id in id_grid.ravel()]
    patch_count = len(patch_dirs)
    aspect_ratio = 0

    for i, patch_dir in enumerate(tqdm(patch_dirs)):
        # Load each patch separately.
        eopatch = EOPatch.load(patch_dir, lazy_loading=True)

        # Plot image.
        ax = axs[i // grid_shape[1]][i % grid_shape[1]]
        im = ax.imshow(img_func(eopatch, **img_func_args), **imshow_args)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        # ax.axis('off')

        # Set aspect ratio based on patch and region shapes.
        if not aspect_ratio:
            data_key = list(eopatch.data.keys())[0]
            patch_shape = eopatch.data[data_key].shape
            width = patch_shape[2] * grid_shape[1]
            height = patch_shape[1] * grid_shape[0]
            aspect_ratio = height / width

        del eopatch

    fig.set_size_inches(size, size * aspect_ratio + 4 * int(bool(colorbar)))
    
    if title:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
        plt.tight_layout()

    fig.subplots_adjust(wspace=0, hspace=0)

    if title:
        fig.suptitle(title, va='top', fontsize=20)

    # Legend
    if colorbar:
        cb = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            orientation='horizontal',
            pad=0.01,
            aspect=100
        )
        cb.ax.tick_params(labelsize=20)

        if isinstance(colorbar, dict):
            if 'ticks' in colorbar:
                cb.set_ticks(colorbar['ticks'])
            if 'labels' in colorbar:
                cb.ax.set_xticklabels(
                    colorbar['labels'],
                    ha='right',
                    rotation_mode='anchor',
                    rotation=45,
                    fontsize=15
                )


def img_rgb(eopatch, date=None):
    """Prepares data for RGB plot for given eopatch.

    :param eopatch: Eopatch
    :type eopatch: EOPatch
    :param date: Reference date, defaults to None
    :type date: datetime or None, optional
    :return: Image data
    :rtype: numpy.array
    """
    time_frame_idx = 0

    if date:
        # Get time frame index closest to the given date.
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        dates = np.array(eopatch.timestamp)
        time_frame_idx = np.argsort(abs(date - dates))[0]

    image = eopatch.data['BANDS'][time_frame_idx][..., [3, 2, 1]]

    return np.clip(image * 3.5, 0, 1)


def img_feature(eopatch, feature):
    """Prepares data for selected timeless mask from given eopatch.

    :param eopatch: Eopatch
    :type eopatch: EOPatch
    :param feature: Feature
    :type feature: FeatureType
    :return: Image data
    :rtype: numpy.array
    """
    return eopatch[feature[0]][feature[1]].squeeze()


def img_diff(eopatch, feature, reference_feature):
    return eopatch.mask_timeless[reference_feature].squeeze() != eopatch.mask_timeless[feature].squeeze()


def plot_roi(country, country_grid, eopatch_ids=None, title=None, size=20):
    """Plot Region Of Interest.

    :param country: Country data
    :type country: GeoDataFrame
    :param country_grid: Country split into bounding boxes, each representing
        one EOPatch
    :type country_grid: GeoDataFrame
    :param eopatch_ids: 2D array of eopatch IDs, defaults to None
    :type eopatch_ids: numpy.array, optional
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param size: plot size, defaults to 20
    :type size: int, optional
    """
    roi_grid = None
    
    if eopatch_ids is not None:
        roi_grid = country_grid.iloc[eopatch_ids.ravel()]

    country_grid_x = country_grid['index_x'].max() + 1
    country_grid_y = country_grid['index_y'].max() + 1

    ratio = country_grid_y / country_grid_x

    fig, ax = plt.subplots(figsize=(size, size * ratio))
    country.plot(ax=ax, facecolor='#eef1f5', edgecolor='#666', alpha=1.0)
    country_grid.plot(ax=ax, facecolor='#ffffff', edgecolor='#666', alpha=0.4)

    if roi_grid is not None:
        roi_grid.plot(ax=ax, facecolor='#f86a6a', edgecolor='#f86a6a', alpha=0.2)

    for idx, row in country_grid.iterrows():
        bbox = row.geometry
        ax.text(
            bbox.centroid.x,
            bbox.centroid.y,
            idx,
            ha='center',
            va='center',
            fontdict={'fontsize': 8}
        )

    if title:
        plot_title = title
    else:
        plot_title = f'{country.iloc[0]["name"]} ({len(country_grid)} eopatches)'

    ax.set_title(plot_title, fontdict={'fontsize': 20})

    plt.axis('off')
    plt.tight_layout()


def plot_features(path, features=None, size=20, max_cols=4):

    eopatch = EOPatch.load(path, lazy_loading=True)

    if features:
        data_timeless_keys = features
    else:
        data_timeless_keys = list(eopatch.data_timeless.keys())

    num_features = len(data_timeless_keys)

    if not num_features:
        print('Nothing to plot')
        return

    num_cols = min(max_cols, num_features)
    num_rows = math.ceil(num_features / num_cols)

    count = 0

    fig = plt.figure(figsize=(20, 20 / num_cols * num_rows))
    # fig.suptitle(f'eopatch_{idx}', fontsize=26)


    for key in data_timeless_keys:
        count += 1
        ax = plt.subplot(num_rows, num_cols, count)

        image = eopatch.data_timeless[key].squeeze()
        image -= image.min()
        image *= (255.0/image.max())
        plt.imshow(image)
        # plt.imshow(eopatch.data_timeless[key].squeeze())

        plt.xticks([])
        plt.yticks([])
        ax.set_aspect("auto")
        plt.title(key, fontsize=20)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    del eopatch


def plot_confusion_matrix(y_test, y_pred, labels, title=None, size=10):
    ticks = [i for i in range(len(labels))]
    cm = confusion_matrix(
        y_test,
        y_pred,
        ticks,
        normalize='pred'
    )

    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(cm, cmap='viridis')
    plt.colorbar(im,fraction=0.0457, pad=0.04)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')

    if title:
        plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.show()
    plt.tight_layout()


def evaluate_grid(eopatch_ids, eopatch_root_path, feature, reference_feature, labels=None, title=None):
    y_true = []
    y_pred = []

    for eopatch_id in eopatch_ids.ravel():
        eopatch_path = os.path.join(eopatch_root_path, f'eopatch_{eopatch_id}')
        eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
        y_true += list(eopatch.mask_timeless[reference_feature].squeeze().ravel())
        y_pred += list(eopatch.mask_timeless[feature].squeeze().ravel())
        del eopatch

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy score: {accuracy}')

    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 score: {f1}')

    if labels:
        plot_confusion_matrix(
            np.array(y_true) - 1,
            np.array(y_pred) - 1,
            labels,
            title=title,
            size=10
        )


def abbreviate(s):
    """Abbreviate given string.

    :param s: Input string
    :type s: str
    :return: Abbreviated string
    :rtype: str
    """
    return ''.join([c for c in s if c in string.ascii_uppercase])
