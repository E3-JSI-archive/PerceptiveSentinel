import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyproj

from DataRequest.DataRequest import TulipFieldRequest


# If major edits are in order, just create new WMS instance (dont forget to update instanceid)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plot_cloud_mask(mask, figsize=(15, 15), cmap="gray", normalize=False,
                    plot_cmap=False, cmap_frac=0):
    """
    Utility function for plotting a binary cloud mask.
    """
    plt.figure(figsize=figsize)
    plot = plt.subplot(1, 1, 1)
    if not normalize:
        vmin_vmax = {"vmin": 0, "vmax": 1}
    else:
        vmin_vmax = dict()
    pl = plot.imshow(mask, cmap=cmap, **vmin_vmax)
    if plot_cmap:
        # Legend
        def fmt(x, pos):
            return "{} %".format(int(round(x * 100)))

        cbar = plt.colorbar(pl, format=ticker.FuncFormatter(fmt),
                            fraction=cmap_frac or 1, pad=0.04)
        cbar.set_label("Clud saturation")
    # Todo: plot nice cmap


def plot_image(data, factor=1. / 255, cmap=None, figsize=(15,7)):
    """
    Utility function for plotting RGB images. The numpy arrays returned by the WMS and WCS requests have channels
    ordered as Blue (`B02`), Green (`B03`), and Red (`B04`) therefore the order has to be reversed before ploting
    the image.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    rgb = data.astype(np.float32)
    if len(rgb.shape) == 3 and rgb.shape[2] == 3:
        rgb = rgb[..., [2, 1, 0]]

    img = rgb*factor
    plt.imshow(img, cmap=cmap)


def plot_timeseries(data, factor=1. / 255, cmap="gray"):
    """
    Utility function for ploting timeseries type data.
    """

    return plot_image(np.array([data]), cmap=cmap)


def plot_timeseries_line(data, vis, spec="ro", new=True, datesp=None,figsize=(15,15),flatten=True, **kwargs):
    if new:
        fig = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if datesp is not None:
            def ext_month(a):
                return f"{a.day}.{a.month}.{a.year}"

            xt = list(map(ext_month, datesp))
            k = 10
            filt = np.arange(0, len(datesp), k)
            plt.xticks(filt, xt[0::k])
    if flatten and len(data.shape) > 1:
        plt.plot(vis[0], rgb2gray(data)[vis], spec, **kwargs)
    else:
        plt.plot(vis[0], data[vis], spec, **kwargs)


def get_timeseries_delete(mask, true_c, bands, dates, cloud_masks, x_ind, y_ind,
                          band_ind):
    # Filter images
    # Choose 0 band as reference for non photographed
    nonzero_image_indices = np.nonzero(bands[:, x_ind, y_ind, 0])
    nonzero_cloud_indices = np.nonzero(cloud_masks[:, x_ind, y_ind])
    both_nonzero = np.intersect1d(nonzero_image_indices, nonzero_cloud_indices)

    return bands[both_nonzero, x_ind, y_ind, band_ind], dates[both_nonzero]


def convert_to_dd(degms):
    deg, other = degms.split("°")
    minn, other = other.split("'")
    sec, way = other.split('"')

    return (float(deg) + float(minn) / 60 + float(sec) / 3600) * [-1, 1][
        way == "N" or way == "E"]


def overlay_cloud_mask(image, mask=None, factor=1. / 255, nrows=1, ncols=1,
                       figsize=(15, 15), mask_maps=None):
    """
    Utility function for plotting RGB images with binary mask overlayed. The numpy arrays returned
    by the Sentinel Hub's WMS and WCS requests have channels ordered as Blue (`B02`), Green (`B03`),
    and Red (`B04`) therefore the order has to be reversed before ploting the image.
    """
    plt.subplots(nrows, ncols, figsize=figsize, frameon=False)
    rgb = np.zeros(image.shape, dtype=np.uint8)
    rgb = image[..., [2, 1, 0]]

    plt.imshow(rgb * factor)
    if mask is not None:
        if mask_maps is None:
            mask = [mask]
            mask_maps = [(0, 255, 255)]
        for (msk, (r,g,b, *a)) in zip(mask, mask_maps):
            try:
                a = a[0]
            except:
                a = 100
            if isinstance(msk, list):
                cloud_image = np.zeros((image.shape[0], image.shape[1], 4),
                                       dtype=np.uint8)
                cloud_image[list(zip(*msk))] = np.asarray([r, g, b, a],
                                                   dtype=np.uint8)
            else:
                cloud_image = np.zeros((msk.shape[0], msk.shape[1], 4),
                                       dtype=np.uint8)
                cloud_image[msk == 1] = np.asarray([r,g,b, a], dtype=np.uint8)
            plt.imshow(cloud_image)
