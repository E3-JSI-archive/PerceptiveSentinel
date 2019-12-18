from sentinelhub import GeopediaFeatureIterator, GeopediaSession
import pandas as pd


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
