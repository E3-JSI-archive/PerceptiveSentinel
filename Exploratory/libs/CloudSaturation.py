import datetime
import pyproj

import numpy as np

from sentinelhub.data_request import WmsRequest, WcsRequest
from sentinelhub.constants import MimeType
from sentinelhub.common import BBox, CRS
from sentinelhub import CustomUrlParam

from s2cloudless import S2PixelCloudDetector

from .DataRequest.DataRequest import TulipFieldRequest

"""
CloudSaturation.py
CloudSaturation class for aggregation of Sentinel-Hub requests on the same 
coordinate window. Combines data and cloud analysis.
"""

# Get your own instance id and put it into file
INSTANCE_ID = open("INSTANCE_ID").read()


def to_epsg3857(latlong_wgs84):
    """
    Converts coordinates to epsg3857 projection
    :param latlong_wgs84: lat long in wgs84 system
    :return: epsg3857 coordinate box
    """
    if len(latlong_wgs84) != 2:
        return [to_epsg3857([latlong_wgs84[0], latlong_wgs84[1]]),
                to_epsg3857([latlong_wgs84[2], latlong_wgs84[3]])]
    epsg3857 = pyproj.Proj(init='epsg:3857')
    wgs84 = pyproj.Proj(init='EPSG:4326')
    return list(
        pyproj.transform(wgs84, epsg3857, latlong_wgs84[1], latlong_wgs84[0]))


class CloudSaturation:
    """
    Helper class to aggregate cloud data and acquired images of the same
    (sensibly small) bounding box
    """

    # Sample tulip layers connected to geopedia data
    LAYERS = {'tulip_field_2016': 'ttl1904', 'tulip_field_2017': 'ttl1905',
              'arable_land_2017': 'ttl1917'}

    class MemoData:
        """
        POD class for holding already downloaded and processed data
        (mostly cloud masks and band indices)
        """

        def __init__(self, total_mask_w, true_color, bands, all_bands, dates,
                     cloud_masks):
            """

            :param total_mask_w: 2d np array of of [0., 1.] values specifying cloud coverage of sample pixel
            :param true_color: N * W * H * 3 array of True Color data
            :param bands: cloud detection bands N * H' * W' * B' array of bands
            for cloud detection (N is number of acquisitions,
            H' is scaled height in pixels, W' is scaled width in pixels,
            B' number of bands used for cloud detection (10 or 13))
            :param all_bands: N * H * W * B array of acquired bands data
            (N is number of acquisitions, H is height, W is width,
            B is number of bands (ussualy 13))
            :param dates: List of N datetime.datetime objects corresponding to date and time of sample acquisition
            :param cloud_masks: N*H*W array of [0,1] masking values of clouds
            """
            self.total_mask_w = total_mask_w
            self.true_color = np.array(true_color)
            self.bands = np.array(bands)
            self.all_bands = all_bands
            self.dates = np.array(dates)
            self.cloud_masks = np.array(cloud_masks)

    def __init__(self,
                 coordinates,
                 start_time='2014-12-01',
                 end_time=datetime.datetime.now().strftime('%Y-%m-%d'),
                 data_folder_name="data/",
                 cloud_scale=6,
                 res=(10, 10),
                 redownload=False,
                 instance_id=INSTANCE_ID,
                 cloud_detector_config=None
                 ):

        self.coordinates = coordinates
        self.bbox = BBox(bbox=self.coordinates, crs=CRS.WGS84)

        self.data_folder_name = data_folder_name
        self.start_time = start_time
        self.end_time = end_time
        self.time_range = (start_time, end_time)
        self.redownload = redownload
        self.instance_id = instance_id
        if cloud_detector_config is None:
            self.cloud_detector_config = {'threshold': 0.4, 'average_over': 4,
                                          'dilation_size': 2, 'all_bands': True}
        else:
            self.cloud_detector_config = cloud_detector_config
        self.res_x, self.res_y = res
        self.cloud_res_x = cloud_scale * self.res_x
        self.cloud_res_y = cloud_scale * self.res_y
        self.cloud_scale = cloud_scale
        self.memo_data = None  # type: CloudSaturation.MemoData

    def create_requests(self):
        """
        Creates basic requests:
        True color request for simpler data visualization
        Scaled all bands request for cloud detection
        All bands request for data processing
        :return: Tuple[tc_request, scaled_cloud_request, all_bands_request]
        """
        # Request true color data for visualization in same resolution as full data
        wms_true_color_request = WcsRequest(layer='TRUE_COLOR',
                                            bbox=self.bbox,
                                            data_folder=self.data_folder_name,
                                            time=self.time_range,
                                            resx=str(self.res_x) + "m",
                                            resy=str(self.res_y) + "m",
                                            image_format=MimeType.PNG,
                                            instance_id=self.instance_id,
                                            custom_url_params={
                                                CustomUrlParam.SHOWLOGO: False,
                                            },
                                            )
        # Scaled cloud detection must be run on images a bit bigger to get
        # boundary pixels with better accuracy
        # and to eliminate missing pixels on the edge
        cloud_bbox = list(self.coordinates)
        # Note: Numbers are not precise
        cloud_bbox[2] += 0.001
        cloud_bbox[3] += 0.001

        # Create cloud request
        wms_bands_request = WcsRequest(layer='ALL-BANDS',
                                       data_folder=self.data_folder_name,
                                       custom_url_params={
                                           CustomUrlParam.SHOWLOGO: False},
                                       bbox=BBox(bbox=cloud_bbox,
                                                 crs=CRS.WGS84),
                                       time=self.time_range,
                                       resx=str(self.cloud_res_x) + "m",
                                       resy=str(self.cloud_res_y) + "m",
                                       image_format=MimeType.TIFF_d32f,
                                       instance_id=self.instance_id)

        # Download all bands
        wms_all_bands_request = WcsRequest(layer='ALL-BANDS',
                                           data_folder=self.data_folder_name,
                                           custom_url_params={
                                               CustomUrlParam.SHOWLOGO: False},
                                           bbox=BBox(bbox=self.bbox,
                                                     crs=CRS.WGS84),
                                           time=self.time_range,
                                           resx=str(self.res_x) + "m",
                                           resy=str(self.res_y) + "m",
                                           image_format=MimeType.TIFF_d32f,
                                           instance_id=self.instance_id)

        return wms_true_color_request, wms_bands_request, wms_all_bands_request

    def load_data(self):
        """
        Prepares and conducts request downloading
        :return: Tuple[true_color_data, cloud_bands_data, all_bands_data, image_acquisition_dates]
        """
        true_color_request, bands_request, all_bands_request = self.create_requests()

        t_c_data = true_color_request.get_data(save_data=True,
                                               redownload=self.redownload)
        print("Saved True color")
        bands_data = bands_request.get_data(save_data=True,
                                            redownload=self.redownload)
        print("Saved cloud detection bands")
        all_bands_data = all_bands_request.get_data(save_data=True,
                                                    redownload=self.redownload)
        print("Saved bands")
        return t_c_data, bands_data, all_bands_data, true_color_request.get_dates()

    @staticmethod
    def upscale_image(img, factor: int):
        """
        Rescale image with suitable integral factor
        :param img:
        :param factor: Scaling factor
        :return: Scaled image
        """
        return np.kron(img, np.ones((factor, factor)))

    @staticmethod
    def get_image_mask(image):
        """
        Converts [0,255] range image to mask [0,1] valued, used for converting
        results of cloud detection to simple image masks easier for plotting,
        manipulation and handling.
        Value of 255 is considered 1 all other values are considered 0
        :param image: 2d image with coefficients [0,255]
        :return: 2d image with same dimensions treated as mask
        """
        return (image == 255).all(axis=2)

    def get_cloud_saturation_mask(self):
        """
        Main acquisition method
        Acquires required data, generates cloud detections masks and saves them
        in memo_data for further use
        :return: Tuple[cloud_saturation_mask, true_color_bands, all_bands,
        data_acquisiton_dates, cloud_masks]
        """
        cloud_detector = S2PixelCloudDetector(**self.cloud_detection_config)
        true_color, bands, all_bands, dates = self.load_data()
        print("Downloaded")
        cloud_masks_orig = cloud_detector.get_cloud_masks(np.array(bands))
        # upscale cloud masks
        cloud_masks = []
        print(cloud_masks_orig[0].shape)
        for i in range(len(cloud_masks_orig)):
            cloud_masks.append(
                self.upscale_image(cloud_masks_orig[i], self.cloud_scale))
        cloud_masks = np.array(cloud_masks)
        print("Detected")
        # Images might be slightly out of scale, crop them
        x, y, _ = true_color[0].shape
        cloud_masks = cloud_masks[:, -x:, :y]
        off_image_detection_mask = sum(map(self.get_image_mask, true_color))
        # Just sum how many times we detect white pixels

        full_cloud_mask = sum(cloud_masks)
        # Get total mask by dividing number of cloud detections by number of all sensible pixels
        total_mask_w = (full_cloud_mask / (
                len(cloud_masks) - off_image_detection_mask)).astype(float)

        self.memo_data = CloudSaturation.MemoData(total_mask_w, true_color,
                                                  bands, all_bands, dates,
                                                  cloud_masks)

        # TODO: Refactor me to some nicer object
        return (total_mask_w, np.array(true_color), np.array(all_bands),
                np.array(dates), np.array(cloud_masks)
                )

    def get_full_index_timeseries(self, index_id: str):
        """
        Acquires time series for specifiec index corresponding to object bouinding box
        :param index_id: WMS configurator index id
        :return: N*H*W*B' array of index data (B' corresponds to number of different values returned by index)
        """
        # If cloud mask is not calculated, first acquire data and calculate cloud masks
        if self.memo_data is None:
            print("Cloud mask not found")
            self.get_cloud_saturation_mask()

        # Create request for corresponding index and preset tile configurations
        wms_index_request = WcsRequest(layer=index_id,
                                       data_folder=self.data_folder_name,
                                       custom_url_params={
                                           CustomUrlParam.SHOWLOGO: False},
                                       bbox=self.bbox,
                                       time=self.time_range,
                                       resx=str(self.res_x) + "m",
                                       resy=str(self.res_y) + "m",
                                       image_format=MimeType.TIFF_d32f,
                                       instance_id=self.instance_id)
        data = wms_index_request.get_data(save_data=True,
                                          redownload=self.redownload)

        return np.array(data)

    def filter_index_timeseries(self, index_timeseries, x_ind, y_ind):
        if self.memo_data is None:
            self.get_cloud_saturation_mask()
        # Filter images
        # Choose 0 band as reference for non photographed
        nonzero_image_indices = np.nonzero(
            self.memo_data.bands[:, x_ind, y_ind, 0])
        nonzero_cloud_indices = np.nonzero(
            self.memo_data.cloud_masks[:, x_ind, y_ind])
        both_nonzero = np.intersect1d(nonzero_image_indices,
                                      nonzero_cloud_indices)

        return index_timeseries[both_nonzero, x_ind, y_ind], \
               self.memo_data.dates[both_nonzero]

    def get_cloud_filter(self, x_ind, y_ind):
        """
        Extracts cloud filter for sample image pixel
        :param x_ind: x_coordinate
        :param y_ind: y_coordinate
        :return: list of cloud masks for corresponding image pixel
        """
        if self.memo_data is None:
            self.get_cloud_saturation_mask()
        return np.nonzero(self.memo_data.cloud_masks[:, x_ind, y_ind])

    def get_visible_filter(self, x_ind, y_ind):
        """
        Extracts visibility filter for sample image pixel
        :param x_ind: x_coordinate
        :param y_ind: y_coordinate
        :return: list of visibility masks for corresponding image pixel
        """
        return np.nonzero(1 - self.get_cloud_filter(x_ind, y_ind))

    def get_tulip_mask(self, layer_name: str):
        """
        Acquires geopedia tulip masks with layer_name for corresponding bounding box
        and produces masks and nonzero indices filter for easeier use
        :param layer_name: Geopedia layer name
        :return: tulip_field_data, tulip_field_mask, nonzero_indices
        """
        geopedia_layer_name = CloudSaturation.LAYERS[layer_name]

        h = self.memo_data.true_color.shape[1]
        w = self.memo_data.true_color.shape[2]
        tulipFields = TulipFieldRequest(bbox=to_epsg3857(self.coordinates),
                                        width=w, height=h, crs=3857,
                                        layer=geopedia_layer_name)
        # Values [255,255,255] on some pixel mean, that there WAS NO field at that pixel
        tulip_field = tulipFields.get_data()[0]
        # Values 1 means NO tulips
        # There sure is a better way, but this works
        tulip_field_mask = 1 - np.array(
            np.dot(tulip_field[..., :], [1, 1, 1]) / 765, dtype=int)
        # tulip_field_mask = 1 - tulip_field_mask ## -> 1 tulips, 0 no tulips

        nonzero_indices = list(zip(*np.nonzero(tulip_field_mask)))

        return tulip_field, tulip_field_mask, nonzero_indices
