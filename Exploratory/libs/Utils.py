import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

"""
Utils.py
Utils functions for data plotting and timeseries visualization. 
Mostly adapted from sentinelhub-py example methods and modified to suit purpose
https://github.com/sentinel-hub/sentinelhub-py 
"""

def plot_cloud_mask(mask, figsize=(15, 15), cmap="gray", normalize=False,
                    plot_cmap=False, cmap_frac=0, label="Cloud Saturation"):
    """
    Plot (arbitrary) 2d image mask. Mask may be non boolean, in that case
    color-map plot is produces
    :param mask: 2d image (numpy array)
    :param figsize: figure size
    :param cmap: color map to be used
    :param normalize: normalize colors in colormap
    :param plot_cmap: plot color map legend
    :param cmap_frac: fraction of image represented by cmap legend
    :param label: image label
    :return: None
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
        if label:
            cbar.set_label(label)


def plot_timeseries_line(data, vis, spec="ro", new=True, datesp=None):
    """
    Plot time-series in temporal scatter plot, will only plot data of the form
    data[vis]
    :param data: Full data to be plotted
    :param vis: filter for plotting data
    :param spec: plot type specification
    :param new: start new plot (or use existing one)
    :param datesp: datetime objects used for xticks
    :return: None
    """
    if new:
        fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
        if datesp is not None:
            def ext_month(a):
                return f"{a.day}.{a.month}.{a.year}"

            xt = list(map(ext_month, datesp))
            k = 10
            filt = np.arange(0, len(datesp), k)
            plt.xticks(filt, xt[0::k])
    plt.plot(vis[0], data[vis], spec)


def convert_to_dd(degms):
    """
    Converts google maps coordinates to decimal coordinates
    :param degms:
    :return:
    """
    deg, other = degms.split("°")
    minn, other = other.split("'")
    sec, way = other.split('"')

    return (float(deg) + float(minn) / 60 + float(sec) / 3600) * [-1, 1][
        way == "N" or way == "E"]


def overlay_cloud_mask(image, mask=None, factor=1. / 255, figsize=(15, 15),
                       mask_maps=None):
    """
    Utility function for plotting RGB images with binary mask overlayed. The numpy arrays returned
    by the Sentinel Hub's WMS and WCS requests have channels ordered as Blue (`B02`), Green (`B03`),
    and Red (`B04`) therefore the order has to be reversed before ploting the image.

    If multiple mask_maps are provided, more maps are overlaid each with corresponding color
    :param image: Base image in True Color to be plotted
    :param mask: Mask or list of masks to be overlaid (2d numpy boolean arrays)
    :param factor: color scaling factor for base
    :param figsize: Figure size
    :param mask_maps: List of rgb values for each mask corresponding to color
    for each mask
    :return: None
    """
    plt.subplots(1, 1, figsize=figsize, frameon=False)
    rgb = image[..., [2, 1, 0]]

    plt.imshow(rgb * factor)
    if mask is not None:
        if mask_maps is None:
            mask = [mask]
            mask_maps = [(0, 255, 255)]
        for (msk, (r, g, b, *a)) in zip(mask, mask_maps):
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
                cloud_image[msk == 1] = np.asarray([r, g, b, a], dtype=np.uint8)
            plt.imshow(cloud_image)
