import pickle
import geopandas as gpd
from pathlib import Path

ROOT_FILE_NAMES = {'Austria': 'Austria_EPSG:32633_30x58_991',
                   'Denmark': 'Denmark_EPSG:32633_36x45_626',
                   'Slovenia': 'Slovenia_EPSG:32633_17x25_276'}

VALID_YEARS = {'Austria': [2016, 2017],
               'Denmark': [2016, 2017, 2018],
               'Slovenia': [2016, 2017]}


def check_year(country, year):
    """
    Checks if year is valid or not.
    """
    if year not in VALID_YEARS[country]:
        raise ValueError(f'Invalid year! Pick one among {VALID_YEARS[country]}')


def check_country(country):
    """
    Checks if country is valid or not.
    """
    if country not in ROOT_FILE_NAMES.keys():
        raise ValueError(f'Invalid country {country}! Pick one among {ROOT_FILE_NAMES.keys()}')


def check_file(file):
    """
    Checks if file exists.
    """
    if not file.is_file():
        raise ValueError(f'File {file} does not exist!')
        

def get_bbox_splitter(country, path):
    """
    Opens pickled BBOXSplitter and returns it.
    
    :param country: country name
    :type country: string
    :param path: path to directory with pickle file
    :type path: Path    
    """
    check_country(country)
    
    file = path/f'{ROOT_FILE_NAMES[country]}.pickle'
    check_file(file)
    
    with open(file, 'rb') as fp:
        bbox_splitter = pickle.load(fp)
        return bbox_splitter


def get_bbox_gdf(country, path):
    """
    Opens shapefile with defined EOPatches (bboxes, indices).
    
    :param country: country name
    :type country: string
    :param path: path to directory with shape file
    :type path: Path    
    """
    check_country(country)
    
    file = path/f'{ROOT_FILE_NAMES[country]}.shp'
    check_file(file)
    
    gdf = gpd.read_file(str(file))
    
    return gdf


def save_bbox_gdf(gdf, country, path):
    """
    Saves shapefile with defined EOPatches (bboxes, indices).

    :param gdf: Geopandas data frame to be saved
    :type gdf: Geopandas data frame
    :param country: country name
    :type country: string
    :param path: path to directory with shape file
    :type path: Path
    """
    check_country(country)

    file = path / f'{ROOT_FILE_NAMES[country]}.shp'
    check_file(file)

    gdf.to_file(str(file))
