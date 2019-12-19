import matplotlib as mpl
import numpy as np

from shapely.geometry import Polygon
from matplotlib import patches, patheffects

# Defintion of Sen2Cor's scene classification color map
# To be used in matplotlib plots
scl_cmap = mpl.colors.ListedColormap(['black', 'red', 'dimgray', 'saddlebrown', 'green', 'yellow', 'blue', 'darkgray',
                                      'lightgray', 'white', 'skyblue', 'violet'])
scl_cmap.set_over('black')
scl_cmap.set_under('black')

bounds = np.arange(-0.5, 12, 1).tolist()
scl_norm = mpl.colors.BoundaryNorm(bounds, scl_cmap.N)

scl_classes = {'NO_DATA': 0,
               'SATURATED_OR_DEFECTIVE': 1,
               'DARK_AREA_PIXELS': 2,
               'CLOUD_SHADOWS': 3,
               'VEGETATION': 4,
               'NOT_VEGETATED': 5,
               'WATER': 6,
               'UNCLASSIFIED': 7,
               'CLOUD_MEDIUM_PROBABILITY': 8,
               'CLOUD_HIGH_PROBABILITY': 9,
               'THIN_CIRRUS': 10,
               'SNOW': 11}
# END Sen2Cor color map definition


def get_extent(eopatch):
    """
    Returns the extent of the patch.
    """
    return [eopatch.bbox.min_x, eopatch.bbox.max_x, eopatch.bbox.min_y, eopatch.bbox.max_y]


def draw_outline(o, lw, foreground='black'):
    """
    Adds outline to the matplotlib patch.
    """
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground=foreground), patheffects.Normal()])
    

def draw_poly(ax, poly, color='r', lw=2, outline=True):
    """
    Draws Polygon.
    """
    if poly is None:
        return
    if poly.exterior is None:
        return
    
    x, y = poly.exterior.coords.xy
    xy = np.moveaxis(np.array([x, y]), 0, -1)
    
    patch = ax.add_patch(patches.Polygon(xy, closed=True, edgecolor=color, fill=False, lw=lw))
    if outline:
        draw_outline(patch, 4)
    

def draw_bbox(ax, eopatch, color='r', lw=2, outline=True):
    """
    Draws EOPatch' BBOX. 
    """
    draw_poly(ax, Polygon(eopatch.bbox.get_polygon()), color=color, lw=lw, outline=outline)


def draw_vector_timeless(ax, eopatch, vector_name, color='b', alpha=0.5):
    """
    Draws all polygons from EOPatch' timeless vector geopandas data frame.
    """
    eopatch.vector_timeless[vector_name].plot(ax=ax, color=color, alpha=alpha)


def draw_true_color(ax, eopatch, time_idx, feature_name='BANDS-S2-L2A', bands=[3,2,1], factor=3.5, grid=True):
    """
    Draws true color image for given time stamp.
    """
    ax.imshow(np.clip(eopatch.data[feature_name][time_idx][..., bands] * factor, 0, 1), extent=get_extent(eopatch))
    if grid:
        ax.grid()

    ax.set_title(f'{feature_name} {eopatch.timestamp[time_idx]}')


def draw_scene_classification(ax, eopatch, time_idx, feature_name='SCL', grid=True):
    """
    Draws Sen2Cor's scene classification.
    """
    ax.imshow(eopatch.mask[feature_name][time_idx].squeeze(), cmap=scl_cmap, norm=scl_norm, extent=get_extent(eopatch))
    if grid:
        ax.grid()

    ax.set_title(f'{feature_name} {eopatch.timestamp[time_idx]}')


def draw_mask(ax, eopatch, time_idx, feature_name, grid=True, vmin=0, vmax=1):
    """
    Draws valid data map.
    """
    if time_idx is None:
        mask = eopatch.mask_timeless[feature_name].squeeze()
    else:
        mask = eopatch.mask[feature_name][time_idx].squeeze()

    ax.imshow(mask, extent=get_extent(eopatch), vmin=vmin, vmax=vmax)
    if grid:
        ax.grid()

    title = f'{feature_name} {eopatch.timestamp[time_idx]}' if time_idx is not None else f'{feature_name}'
    ax.set_title(title)
