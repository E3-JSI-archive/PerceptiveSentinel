from eolearn.core import LinearWorkflow, FeatureType, SaveToDisk, OverwritePermission, LoadFromDisk
from eolearn.io import S2L2AWCSInput, AddSen2CorClassificationFeature
from eolearn.mask import get_s2_pixel_cloud_detector, AddCloudMaskTask, AddValidDataMaskTask
from eolearn.features import SimpleFilterTask

from .tasks import CreatePatch, AddGeopediaVectorFeature, AddAreaRatio
from .tasks import SentinelHubValidData, Sen2CorValidData, MergeMasks, ValidDataFractionPredicate

from sentinelhub import MimeType

from datetime import timedelta


GEOPEDIA_LPIS_LAYERS = {'Denmark_LPIS_2018': 2051,
                        'Denmark_LPIS_2017': 2051,
                        'Denmark_LPIS_2016': 2051,
                        'Austria_LPIS_2017': 2034,
                        'Austria_LPIS_2016': 2033,
                        'Slovenia_LPIS_2017': 2038,
                        'Slovenia_LPIS_2016': 2037,
                        'Austria_FELDSTUECKE_2017': 2029,
                        'Austria_FELDSTUECKE_2016': 2027}


GEOPEDIA_LPIS_YEAR_NAME = {'Denmark': 'Year',
                           'Slovenia': 'LETO',
                           'Austria': None}

TIME_INTERVAL = {2016: ['2016-01-01', '2016-09-30'],
                 2017: ['2017-01-01', '2017-09-30'],
                 2018: ['2018-01-01', '2018-09-30']}


def get_create_and_add_lpis_workflow(country, year, out_path):
    """
    Creates an EOWorkflow that:
    1. creates an empty patch
    2. add LPIS vector data fetched from Geopedia 
    3. calculate the ratio between area of all fields and area of EOPatch
    4. save newly created EOPatch to disk
    """
    print(f'Preparing EOWorkflow for {country} and year {year}:')
    
    # 1. create empty patch
    create = CreatePatch()
    print(f'   1. Creating empty EOPatch')

    # 2. Add LPIS vector data
    layer_id = GEOPEDIA_LPIS_LAYERS[f'{country}_LPIS_{year}']
    ftr_name = f'LPIS_{year}'
    year_filter = (GEOPEDIA_LPIS_YEAR_NAME[country], year) if GEOPEDIA_LPIS_YEAR_NAME[country] is not None else None
    
    add_lpis = AddGeopediaVectorFeature((FeatureType.VECTOR_TIMELESS, ftr_name),
                                        layer=layer_id, year_filter=year_filter, drop_duplicates=True)
    print(f'   2. Adding LPIS vector data to feature {ftr_name}')
    print(f'      -> Geopedia layer ID: {layer_id}')
    if year_filter is not None:
        print(f'      -> with year filter applied: {year_filter}')

    # 3. Add Area Ratio
    area_ratio = AddAreaRatio((FeatureType.VECTOR_TIMELESS, ftr_name),
                              (FeatureType.SCALAR_TIMELESS, 'FIELD_AREA_RATIO'))
    print(f'   3. Calcuating the field area ratio')

    # 4. Save EOPatch to disk
    out_dir = str(out_path + '/' + country + '/' + str(year))
    save = SaveToDisk(out_dir, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    print(f'   4. Saving EOPatches to: {out_dir}')
    
    workflow = LinearWorkflow(create, add_lpis, area_ratio, save, 
                              task_names={create: 'create', add_lpis: 'add_lpis',
                                          area_ratio: ' area_ratio', save: 'save'})
    
    return workflow


def get_add_l2a_data_workflow(data):
    """
    Creates an workflow that:
    1. loads existing EOPatch
    2. adds sen2cor scene classification map
    3. adds L2A data (all 12 bands)
    4. adds s2cloudless cloud masks
    5. determines `L2A_VALID` - map of valid observations (t,h,w,1) based on L2A SCL map
        * pixels are marked as valid, if they're tagged as
        `[DARK_AREA_PIXELS, VEGETATION, NOT_VEGETATED, WATER, UNCLASSIFIED]`
        * performs opening with disk with radius 11 on `L2A_VALID`
    6. determines `L1C_VALID` - map of valid observations (t,h,w,1) based on s2cloudless cloud map
        * pixels are marked as valid, if they're tagged as not cloud
    7. saves EOPatch to disk
    """
    # 1. loads existing EOPatch
    load = LoadFromDisk(str(data))

    # 2. add L2A
    add_l2a = S2L2AWCSInput(layer='BANDS-S2-L2A', resx='10m', resy='10m', maxcc=0.8, time_difference=timedelta(hours=2),
                            raise_download_errors=False)

    # 3. add sen2cor's scene classification map and snow probability map
    add_scl = AddSen2CorClassificationFeature(sen2cor_classification='SCL', layer='TRUE-COLOR-S2-L2A',
                                              image_format=MimeType.TIFF_d32f, raise_download_errors=False)

    # 4. add s2cloudless cloud mask
    cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
    add_clm = AddCloudMaskTask(cloud_classifier,
                               'BANDS-S2CLOUDLESS',
                               cm_size_y='160m',
                               cm_size_x='160m',
                               cmask_feature='CLM')

    # create valid data masks
    scl_valid_classes = [2, 4, 5, 6, 7]

    # 5. and 6. add L2A and L1C valid data masks
    add_l1c_valmask = AddValidDataMaskTask(SentinelHubValidData(), 'L1C_VALID')
    add_l2a_valmask = AddValidDataMaskTask(Sen2CorValidData(scl_valid_classes, 6, 22), 'L2A_VALID')
    add_valmask = AddValidDataMaskTask(MergeMasks('L1C_VALID', 'L2A_VALID'), 'VALID_DATA')

    # 3. keep only frames with valid data fraction over 70%
    valid_data_predicate = ValidDataFractionPredicate(0.7)
    filter_task = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'), valid_data_predicate)

    # save
    save = SaveToDisk(str(data), compress_level=1, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow = LinearWorkflow(load, add_l2a, add_scl, add_clm, add_l1c_valmask, add_l2a_valmask, add_valmask,
                              filter_task, save,
                              task_names={load: 'load', add_l2a: 'add_L2A', add_scl: 'add_SCL', add_clm: 'add_clm',
                                          add_l1c_valmask: 'add_L1C_valmask', add_l2a_valmask: 'add_L2A_valmask',
                                          add_valmask: 'add_valmask', filter_task: ' filter_task', save: 'save'})

    return workflow
