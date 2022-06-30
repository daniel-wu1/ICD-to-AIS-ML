import pandas as pd
import math

def create_inclusive_df(years, prefix_age, prefix_ecodes, prefix_pcodes, prefix_dcodes, truncate_ais, age_cat, min_age):
    '''
    This function takes a list of years in int or string format (YYYY)
    and produces a pandas data frame that contains:
    - Inc_key 8 number identifier for a patient
    - Age prefixed by an A
    - Year of admission as YYYY
    - One ICD9 ecode
    - ICD9 procedure codes in a list
    - ICD9 diagnosis codes in a list
    - AIS05 codes in a list
    
    Parameters:
        years - list of years
        prefix_age - add 'A' designator to age
        prefix_ecodes - add 'E' designator to E-codes
        prefix_pcodes - add 'P' designator to P-codes
        prefix_dcodes - add 'D' designator to D-codes
        turncate_ais - remove severity designation of AIS codes
        age_cat - place ages in bins
        min_age - minimum age of patient
    Returns:
        data frame containing the above columns
    '''

    # Obtain a csv pathlist for the AGES for every year
    pathlist = __get_pathlist_type('age', years)
    # For each path read and add to a dataframe
    pt_df =  __get_df_from_pathlist(pathlist, ["INC_KEY", "AGE"], True) # Will be final df
    pt_df = pt_df.astype({'AGE':'int'})
    pt_df = pt_df[pt_df['AGE'] >= min_age]
    if age_cat:
        pt_df = __bin_ages(pt_df)
    pt_df = pt_df.astype({'AGE':'str'})
    if prefix_age:
        pt_df['AGE'] = 'A' + pt_df.AGE
    
    print("Length from age:")
    print(len(pt_df))
    
    
    # Obtain a csv pathlist for the ECODES for every year
    pathlist = __get_pathlist_type('ecode', years)
    # For each path read and add to a dataframe
    dcode_df = __get_df_from_pathlist(pathlist, ["INC_KEY", "ECODE"])
    pt_df = pt_df.merge(dcode_df, on='INC_KEY')
    # filter out cases not coded in ICD-9
    pt_df = pt_df[pt_df.ECODE!='-1']
    if prefix_ecodes:
        pt_df['ECODE'] = 'E' + pt_df.ECODE
        
    print("\nLength after adding E codes:")
    print(len(pt_df))
        
        
    # Obtain a csv pathlist for the PCODES for every year
    pathlist = __get_pathlist_type('pcode', years)
    # For each path read and add to a dataframe
    pcode_df = __get_df_from_pathlist(pathlist)
    if prefix_pcodes:
        pcode_df['PCODE'] = 'P' + pcode_df.PCODE
    # transform PCODES into list for each patient
    pcode_df = pcode_df.groupby('INC_KEY')['PCODE'].apply(list).reset_index(name='PCODES')
    pt_df = pt_df.merge(pcode_df, on='INC_KEY')
    
    print("\nLength after adding P codes:")
    print(len(pt_df))
    
    
    # Obtain a csv pathlist for the DCODES for every year
    pathlist = __get_pathlist_type('dcode', years)
    # For each path read and add to a dataframe
    dcode_df = __get_df_from_pathlist(pathlist)
    if prefix_dcodes:
        dcode_df['DCODE'] = 'D' + dcode_df.DCODE
    # remove blank results, not sure why I am getting blank codes
    dcode_df = dcode_df[~(dcode_df.DCODE=='')]
    # remove 'V' codes
    dcode_df = dcode_df[~dcode_df.DCODE.str.contains('V', na=False)]
    # transform DCODES into list for each patient
    dcode_df = dcode_df.groupby('INC_KEY')['DCODE'].apply(list).reset_index(name='DCODES')
    pt_df = pt_df.merge(dcode_df, on='INC_KEY')
    
    print("\nLength after adding D codes:")
    print(len(pt_df))
    
    # Obtain a csv pathlist for the AIS05 for every year
    pathlist = __get_pathlist_type('ais05', years)
    # For each path read and add to a dataframe
    if not truncate_ais:
        ais_df = __get_df_from_pathlist(pathlist, ["INC_KEY", "PREDOT", "SEVERITY"])
        ais_df["AIS05CODE"] = ais_df['PREDOT'].astype(str) +"."+ ais_df["SEVERITY"]
    else:
        ais_df = __get_df_from_pathlist(pathlist, ["INC_KEY", "PREDOT"])
        ais_df.rename(columns={'PREDOT':'AIS05CODE'}, inplace=True)
    # Remove none codes
    ais_df = ais_df[~ais_df.AIS05CODE.isnull()].reset_index(drop=True)
    # transform PCODES into list for each patient
    ais_df = ais_df.groupby('INC_KEY')['AIS05CODE'].apply(list).reset_index(name='AIS05CODE')
    pt_df = pt_df.merge(ais_df, on='INC_KEY')
    
    print("\nLength after adding ais codes:")
    print(len(pt_df))
    
    return pt_df


def __get_df_from_pathlist(pathlist, col_list=None, first=False):
    '''
    This function takes a list of paths with the same data
    type and a list of columns to grab and converts that list
    of paths into an appended dataframe
    
    Parameters:
        pathlist: list of string paths to csv's
        col_list: list of columns to obtain
            - if None will get all
        first: if True, will add year of admission
            - use on first call
    Returns:
        pandas dataframe from csv data
    '''
    # For each path read and add to a dataframe
    df_list = []
    for path in pathlist:
        if col_list is None:
            df = pd.read_csv(path, dtype=str)
        else:
            df = pd.read_csv(path, dtype=str, usecols=col_list)
        if first:
            year_index = path.index('2')
            year = path[year_index:(year_index+4)]
            df['YOADMIT'] = year
        df_list.append(df) 
    df = pd.concat(df_list)        
    return(df)


def __get_pathlist_type(dtype, years):
    '''
    This function takes a string of data type and list of years
    and produces a pathlist of data files to be accessed
    for that data type over those years
    
    Parameters:
        dtype: String that can be age, yoadmit, ecode, pcode, ais05, dcode
        years: list of years
    Returns:
        list of paths to load into a dataframe
    '''
    pathlist = []
    for year in years:
        if dtype == 'dcode':
            path = "../NTDB/PUF AY " + str(year) + "/CSV/PUF_DCODE.csv"
        elif dtype == 'ecode':
            path = "../NTDB/PUF AY " + str(year) + "/CSV/PUF_ECODE.csv"
        elif dtype == 'pcode':
            path = "../NTDB/PUF AY " + str(year) + "/CSV/PUF_PCODE.csv"
        elif dtype == 'age':
            path = "../NTDB/PUF AY " + str(year) + "/CSV/PUF_DEMO.csv"
        elif dtype == 'ais05':
            path = "../NTDB/PUF AY " + str(year) + "/CSV/PUF_AISP05CODE.csv"
        else:
            raise Exception("dtype must be one of: age, yoadmit, ecode, pcode, ais05, dcode")
        # Add for each file we need to access
        pathlist.append(path)
    return pathlist


def __bin_ages(pt_df):
    '''
    This function categorizes ages by brackets
    
    Parameters:
        pt_df dataframe containing AGE as an int
    Returns:
        li
    '''
    # cut points and labels
    age_bins = [0,10,20,30,40,50,60,70,80,90,110]
    age_labels = ['00_09','10_19','20_29','30_39','40_49','50_59','60_69','70_79','80_89','90_99']
    # get categories
    pt_df['AGEBIN'] = pd.cut(pt_df.AGE, bins=age_bins, labels=age_labels)
    # replace age with categories
    pt_df['AGE'] = pt_df.AGEBIN
    # remove extra columns
    pt_df = pt_df.drop(columns=['AGEBIN'])
    return pt_df


def write_pt_dat(pt_df, columns, output_file):
    '''
    The function takes a dataframe and output the values in all columns in
    as values separated with spaces.
    
    Arguments:
        pt_df - dataframe with patient data
        columns - columns with data to include
        output_file - name of output file
    Returns:
        None
    '''
    # select only columns of interest
    pt_df = pt_df[columns].copy()
    # loop through all columns
    for col in columns:
        #check if column contains list
        if isinstance(pt_df[col].reset_index(drop=True)[0], list):
            # Turn all elements in list to strings
            pt_df[col] = pt_df[col].apply(lambda x: __stringify_list(x))
            # Sort the list
            pt_df[col] = pt_df[col].apply(lambda x: sorted(x))
            # Convert list to string
            pt_df[col] = pt_df[col].apply(lambda x: " ".join(x))
        else:
            # convert values to string
            pt_df[col] = pt_df[col].apply(lambda x: str(x))    
    # merge columns
    pt_df = pt_df.apply(lambda x: " ".join(x), axis=1)
    pt_df.to_csv(output_file,index=False, header=False)
    
    
def __stringify_list(_list):
    '''
    Function takes all the elements within _list and converts
    them to strings using the str() function
    
    Arguments:
        _list: list of items that can be converted to string
    Returns:
        A new list containing only strings
    '''
    new_list = []
    for item in _list:
        new_list.append(str(item)) 
    return new_list
        
    