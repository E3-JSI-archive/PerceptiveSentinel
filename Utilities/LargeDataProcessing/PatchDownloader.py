from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from eolearn.io import S2L1CWCSInput, S2L1CWMSInput
from shapely.geometry import Polygon
import os
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
import datetime as dt
import time


def generate_slo_shapefile(path):
    DATA_FOLDER = os.path.join('data')

    area = gpd.read_file(os.path.join(DATA_FOLDER, 'svn_buffered.geojson'))

    # Convert CRS to UTM_33N
    country_crs = CRS.UTM_33N
    area = area.to_crs(crs={'init': CRS.ogc_string(country_crs)})

    # Get the country's shape in polygon format
    country_shape = area.geometry.values.tolist()[-1]

    # Plot country
    area.plot()
    plt.axis('off');

    # Create the splitter to obtain a list of bboxes
    bbox_splitter = BBoxSplitter([country_shape], country_crs, (25 * 2, 17 * 2))

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    path_out = path + '/shapefiles'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list]
    idxs_y = [info['index_y'] for info in info_list]

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                           crs={'init': CRS.ogc_string(country_crs)},
                           geometry=geometry)

    shapefile_name = path_out + '/slovenia.shp'
    gdf.to_file(shapefile_name)

    return gdf, bbox_list


def download_patches(path, shp, bbox_list):
    add_data = S2L1CWCSInput(
        layer='BANDS-S2-L1C',
        feature=(FeatureType.DATA, 'BANDS'),  # save under name 'BANDS'
        resx='10m',  # resolution x
        resy='10m',  # resolution y
        maxcc=0.8,  # maximum allowed cloud cover of original ESA tiles
    )
    path_out = path + '/Slovenija/'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    save = SaveToDisk(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow = LinearWorkflow(
        add_data,
        save
    )

    time_interval = ['2017-01-01', '2017-12-31']  # time interval for the SH request

    execution_args = []
    for idx, bbox in enumerate(bbox_list):
        execution_args.append({
            add_data: {'bbox': bbox, 'time_interval': time_interval},
            save: {'eopatch_folder': 'eopatch_{}'.format(idx)}
        })
    start_time = time.time()
    executor = EOExecutor(workflow, execution_args, save_logs=True)
    executor.run(workers=1, multiprocess=False)
    file = open('timing.txt', 'a')
    running = str(dt.datetime.now()) + 'Running time: {}'.format(time.time() - start_time)
    print(running)
    file.write(running)
    file.close()


if __name__ == '__main__':
    path = 'E:/Data/PerceptiveSentinel'
    gdf, bbox_list = generate_slo_shapefile(path)
    download_patches(path, gdf, bbox_list)
