from eolearn.core import EOTask, FeatureType
import os
import requests
from time import sleep
import rasterio
import json
import zipfile
import wget
import geopandas as gpd


class GeoVilleServices(EOTask):
    """
    Calls EO-services from GeoVille.
    https://services.geoville.com/
    """
    def __init__(self, email, password, data_dict=None, outpath=None):
        
        self.auth_url = "https://services.geoville.com/api-token-auth/"
        self.task_url = 'https://services.geoville.com/api/v1/tasks/'
        
        self.email = email
        self.password = password
        
        self.data_dict = data_dict

        self.name = self.data_dict['name']
        self.begin = self.data_dict['begin']
        self.end = self.data_dict['end']
        
        if not outpath:
            self.outpath = "./GeoVille_{}_{}_{}.tif".format(self.name, self.begin, self.end)
        else:
            self.outpath = outpath
        
    def login(self):
        headers = {
            'Content-type': 'application/json',
        }

        data = json.dumps({"email": self.email, "password": self.password})
        response = requests.post(self.auth_url, headers=headers, data=data)
        output = json.loads(response.text)
        return output["token"]

    def retrieve_task_metadata(self, token, uuid):
        headers = {
            'Authorization': 'Bearer {}'.format(token),
        }

        response = requests.get(self.task_url + uuid, headers=headers)
        output = json.loads(response.text)
        return output

    def download_task(self, token, uuid):
        headers = {
            'Authorization': 'Bearer {}'.format(token),
        }

        response = requests.get(self.task_url + uuid + "/download", headers=headers)
        url = response.text.split('"')[4].replace("\\", "")
        if "zip" in url:
            self.outpath = self.outpath.replace("tif", "zip")
        wget.download(url, out=self.outpath)

        if "zip" in self.outpath:
            zip_ref = zipfile.ZipFile(self.outpath, 'r')
            zip_ref.extractall(os.path.dirname(self.outpath))
            zip_ref.close()
            shp_path = os.path.join(os.path.dirname(self.outpath), 
                                    [f for f in zip_ref.namelist() if f.endswith(".shp")][0])
            
            data_out = gpd.read_file(shp_path)
            
        else:
            with rasterio.open(self.outpath) as dataset:
                data_out = dataset.read()

        return data_out

    def submit_task(self, token, data_dict):
        headers = {
            'Authorization': 'Bearer {}'.format(token),
        }

        data = json.dumps(data_dict)
        response = requests.post(self.task_url, headers=headers, data=data)
        output = json.loads(response.text)
        return output["uuid"]

    def execute(self, eopatch):
        token = self.login()
        uuid = self.submit_task(token, self.data_dict)

        running = True
        while running:
            metadata = self.retrieve_task_metadata(token, uuid)
            if metadata['status'] == 'FINISHED':        
                eopatch.add_feature(FeatureType.META_INFO, self.name + "_metadata", {**self.data_dict, **metadata})
                output = self.download_task(token, uuid)
                if "zip" in self.outpath:
                    eopatch.add_feature(FeatureType.VECTOR_TIMELESS, self.name, output)
                else:
                    eopatch.add_feature(FeatureType.DATA_TIMELESS, self.name, output)

                running = False
            elif metadata['status'] == 'FAILED':
                print("ERROR: {}".format(metadata))
                running = False
            elif metadata['status'] == 'RUNNING':
                print("#", sep='', end='', flush=True)
                sleep(10)  # 30 seconds

        return eopatch

########################################################################################################################
# USAGE                                                                                                                #
########################################################################################################################
# email = None
# password = None

# data_dict = {"name":"fieldparcel","source":1,"cloudmask":"","maskvalues":[0,1,2,3,8,9,11],"begin":"2017-01-01",
# "end":"2017-12-31","wcs":False,"zoom_factor":4,"npoints":500,"nreps":10,"maxNDVImin":8000,"segment_diff_max_ratio":0,
# "border_diff_mult":1.5,"segment_percentile":25,"border_percentile":25,"debug":False,
# "field":{"type":"FeatureCollection","name":"Absam_AOI_WGS84",
# "crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::4326"}},
# "features":[{"type":"Feature","properties":{"id":0},
# "geometry":{"type":"Polygon","coordinates":[[[11.517448529775444,47.30097050303895],
# [11.51432643118898,47.31530806434312],[11.549095256356405,47.320599469996615],[11.556190934961993,47.30722562211768],
# [11.56158365070225,47.28595519741436],[11.559738774264786,47.275942625096825],[11.54469593562093,47.271320798763],
# [11.519009579068674,47.26361685785759],[11.515319826193774,47.279119896552174],
# [11.514468344761091,47.285377600517975],[11.517448529775444,47.30097050303895]]]}}]}}

# add_gv_service = GeoVilleServices(email, password, data_dict=data_dict)
# workflow = LinearWorkflow(add_gv_service)
# result = workflow.execute()
