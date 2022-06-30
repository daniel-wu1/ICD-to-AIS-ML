import numpy as np
import pandas as pd
import random
import math


def tidy_icd_code(icd_codes):
    # Split strings into separate columns (wide format)
    # So the csv data of patient injuries has the patient number and a list of ICD injuries sustained in one cell... so what we do is split that up
    icd_codes = icd_codes.ICD9CODE.str.split(' ', expand=True)    
    # Convert to long format
    # So you have clean data now instead of
    clean_icd_codes = pd.DataFrame(icd_codes.stack()).reset_index().rename(columns={'level_0':'key',0:'icd9_code'}).drop(columns=['level_1'])
    return(clean_icd_codes)


def trim_codes(icd_codes):
    # Remove all codes that are not 'D' codes
    icd_codes = icd_codes[icd_codes.icd9_code.str.contains('D')].reset_index(drop=True)
    # Remove leading 'D' cuz u know it's implied
    icd_codes['icd9_code'] = icd_codes.icd9_code.apply(lambda x: x.lstrip('D'))
    return(icd_codes)


def ICD9_AIS08_map(_map, icd_codes, name):
    # Merge map onto ICD codes
    if name == "SMT":
        left_column = 'icd9_code'
        right_column = 'ais_code'
    else:
        left_column = 'CODE'
        right_column = 'AIS_CODE'
    icd_codes = icd_codes.merge(_map[[left_column, right_column]], how='left', left_on='icd9_code', right_on=left_column)
    # Sort in ascending order
    # Note that the keys are already in order ? seemingly, what gets ordered instead is the AIS_CODE from lowest to highest
    # convert codes to numbers
    if name == 'SMT':
        icd_codes = icd_codes.sort_values(['key',left_column])
        icd_codes[right_column] = icd_codes.ais_code.astype(np.float64)
    else:
        icd_codes = icd_codes.sort_values(['key',right_column])
        icd_codes[right_column] = icd_codes.AIS_CODE.astype(np.float64)   
    # fill in unmapped codes with '-1'
    icd_codes = icd_codes.fillna(-1)
    # fill in uspecified codes with 0
    icd_codes = icd_codes.replace(-100000, 0)
    return(icd_codes)


def convert_codes_to_lists(icd_codes):
    try:
        icd_codes = icd_codes.groupby('key')['AIS_CODE'].apply(list).reset_index(name='AIS_CODE').drop(columns=['key'])
    except:
        icd_codes = icd_codes.groupby('key')['ais_code'].apply(list).reset_index(name='ais_code').drop(columns=['key'])
    return(icd_codes)