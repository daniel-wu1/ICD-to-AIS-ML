import numpy as np
import pandas as pd
import sqlite3
import random
import math
import icd9cms
import icd10
import matplotlib.pyplot as plt


def get_ais_dot_map(ais_codes):
    ais_dot = pd.Series(ais_codes.code.values, index=ais_codes.predot).to_dict()
    ais_dot[0] = 0.0
    ais_dot[-1] = -1.0
    return ais_dot


def process_ais_codes(test_res, ais_dot):
    # convert string to list
    test_res['AIS_obs'] = test_res.AIS_obs.str.split(" ")
    test_res['AIS_pred'] = test_res.AIS_pred.str.split(" ")
    # replace predicted NaN (no prediction for patient) with 0 
    test_res['AIS_pred'] = [ ["0"] if x is np.NaN else x for x in test_res['AIS_pred']]
    # replace <unk> (unknown code encountered) with -1
    test_res['AIS_pred'] = test_res.apply(lambda x: ["-1" if val=="<unk>" else val for val in x['AIS_pred']], axis=1)
    # convert to numbers
    test_res['AIS_obs'] = test_res.apply(lambda x: [int(val) for val in x['AIS_obs']], axis=1)
    test_res['AIS_pred'] = test_res.apply(lambda x: [int(val) for val in x['AIS_pred']], axis=1)
    # convert map string to number
    test_res['AIS_map'] = test_res.AIS_map.str.replace('[','').str.replace(']','')
    test_res['AIS_map'] = test_res.apply(lambda x: [np.float64(val) for val in str(x['AIS_map']).split(",")], axis=1)
    # add post-dot code
    test_res['AIS_obs'] = test_res.apply(lambda x: [ais_dot[val] for val in x['AIS_obs']], axis=1)
    test_res['AIS_pred'] = test_res.apply(lambda x: [ais_dot[val] for val in x['AIS_pred']], axis=1)
    return(test_res)


def match_ais(ais1, ais2):
    '''
    This function matches AIS codes from two lists with proximation.  The overarching concept
    is to match codes based on 1) exact matches, 2) same body region-same severity, 
    3) same body region-different severity, 4) different body region-same severity, 
    5) remaining codes are paired in decreasing severity, 6) unmatched codes are then added.
    
    Parameters:
        ais1 - list of AIS codes
        ais2 - list of AIS codes
    Returns:
        Pandas dataframe with matched codes
    '''
    # df for matches
    matches = pd.DataFrame(columns={'codes1','code2'})
    
    # df for data
    df1 = pd.DataFrame({'ais_code':ais1, 'curr_prec':ais1}).sort_values('ais_code')
    df2 = pd.DataFrame({'ais_code':ais2, 'curr_prec':ais2}).sort_values('ais_code')

    # determine body region
    df1['region'] = np.floor(df1.ais_code/100000)
    df2['region'] = np.floor(df2.ais_code/100000)
    
    # determine severity
    df1['severity'] = np.round(df1.curr_prec%1 * 10)
    df2['severity'] = np.round(df2.curr_prec%1 * 10)

    # move severity to the 2nd MSD
    df1['curr_prec'] = np.floor(df1.region * 1000000 + df1.severity * 100000 + df1.ais_code%100000)
    df2['curr_prec'] = np.floor(df2.region * 1000000 + df2.severity * 100000 + df2.ais_code%100000)    
    
    # matching steps #1-3, loop through all digits
    for digit in range(0,7):

        # loop through ais1 codes
        for i in range(0,len(df1)):

            # loop through ais2 codes
            for j in range(0,len(df2)):

                # check for match
                if df1.curr_prec[i] == df2.curr_prec[j]:

                    # match found, add to match list
                    matches = matches.append({'code1':df1.ais_code[i], 'code2':df2.ais_code[j]}, ignore_index=True)

                    # remove rows with matched codes from dfs
                    df1 = df1.drop(index=i, axis=1)
                    df2 = df2.drop(index=j, axis=1).reset_index(drop=True)

                    # stop searching for code
                    break

        # decrease precision, on first loop this removes severity
        df1['curr_prec'] = np.floor(df1.curr_prec/10)
        df2['curr_prec'] = np.floor(df2.curr_prec/10)

        # reset index for list1
        df1 = df1.reset_index(drop=True) 

    # matching step #4 - find same severity in different body regions    
        
    # assign severity to current precision
    df1['curr_prec'] = df1.severity
    df2['curr_prec'] = df2.severity
    
    # sort based on codes
    df1 = df1.sort_values('ais_code').reset_index(drop=True)
    df2 = df2.sort_values('ais_code').reset_index(drop=True)
    
    #loop through all unmatched codes in list 1
    for i in range(0,len(df1)):
        
        # loop through unmatched codes in list 2
        for j in range(0,len(df2)):
                
            # check for match
            if df1.curr_prec[i] == df2.curr_prec[j]:

                # match found, add to match list
                matches = matches.append({'code1':df1.ais_code[i], 'code2':df2.ais_code[j]}, ignore_index=True)

                # remove rows with matched codes from dfs
                df1 = df1.drop(index=i, axis=1)
                df2 = df2.drop(index=j, axis=1).reset_index(drop=True)

                # stop searching for code
                break
                
    # matching step #5-6, sequentially assign unmatched codes from based on decreasing AIS severity

    # arrange in decreasing severity
    df1 = df1.sort_values('curr_prec', ascending=False).reset_index(drop=True)
    df2 = df2.sort_values('curr_prec', ascending=False).reset_index(drop=True)
  
    #loop through all unmatched codes in list 1
    for i in range(0,len(df1)):

        # check if list 2 still has any numbers
        if len(df2) > 0:

            # add to match list
            matches = matches.append({'code1':df1.ais_code[i], 'code2':df2.ais_code[0]}, ignore_index=True)

            # remove rows with matched codes from df
            df2 = df2.drop(index=0, axis=1).reset_index(drop=True)

        # else no more codes in list 2
        else:

            # add to match list
            matches = matches.append({'code1':df1.ais_code[i], 'code2':None}, ignore_index=True)

        # remove code from list 1
        df1 = df1.drop(index=i, axis=1)

    # assign unmatched to any remaining codes list 2
    for j in range(0,len(df2)):

        # add to match list
        matches = matches.append({'code1':None, 'code2':df2.ais_code[j]}, ignore_index=True)
        
    # remove values that are undefined
    matches['code1'] = matches.code1.apply(lambda x: None if x<=0 else x)
    matches['code2'] = matches.code2.apply(lambda x: None if x<=0 else x)

    # arrange by decrease AIS severity
    matches['severity1'] = matches.code1%1
    matches = matches.sort_values('code1').sort_values('severity1',ascending=False)[['code1','code2']].reset_index(drop=True)

    return matches


def calc_ISS(codes_list, ais_codes, NISS=False, mapped_codes=False):
    '''
    This function accepts a list of AIS codes and returns the ISS.  This is based
    on the six body regions method.
    
    Parameters:
        ais_codes - list of AIS codes
        NISS - True if the new injury severity score method should be used
        
    Returns:
        ISS or NISS
    '''
    
    # dataframe for code and region info
    codes_df = pd.DataFrame(codes_list, columns=['code'])
    
    # add region to codes
    if mapped_codes == False:
        codes_df = codes_df.merge(ais_codes[['code','region']], how='left', on='code')
    else:
        codes_df['region'] = ((codes_df.code/10000)%10).astype(int)
    
    # get severity and severity squared
    codes_df['severity'] = ((codes_df.code*10)%10).astype(int)
    codes_df['severity_sq'] = np.square(codes_df.severity)
       
    # check if any severity is 9, then unknown
    if (codes_df.severity==9).any():
        # IS is unknown
        IS = np.nan

    # else check if any severity is 6, then automatic 75
    elif (codes_df.severity==9).any():
        # IS is max (75)
        IS = 75
        
    # calculate injury severity
    else:
        # check if using NISS, highest severity in any region
        if NISS:
            # get 3 highest severity codes
            codes_df = codes_df.sort_values('severity', ascending=False).reset_index(drop=True).head(3)

            # calculate injury severity
            IS = sum(codes_df.severity_sq)
            
        # else using ISS, highest severity in different body regions
        else:
            # get highest severity codes in different body regions
            codes_df = codes_df.sort_values('severity', ascending=False).groupby('region').head(1).reset_index(drop=True)
            
            # get 3 highest severity codes
            codes_df = codes_df.sort_values('severity', ascending=False).reset_index(drop=True).head(3)
            
            # calculate injury severity
            IS = sum(codes_df.severity_sq)
        
    return IS



def eval_matches(codes_df):
    '''
    This function determines the level of match  between paired
    lists of AIS codes.  

    Parameter:
        codes_df - dataframe with matched list of AIS codes, two columns of AIS codes
        
    Returns:
        Dataframe with match level added as one-hot encoding.  The levels are:
            exact - number of exact matches
            same_reg_same_sev - same body region, same severity, but not exact match
            same_reg_diff_sev - same body region, different severity 
            diff_reg_same_sev - same body region, different severity 
            diff_reg_diff_sev - different body region, different severity, but matched
            unmatched_obs - number of unmatched observed codes
            unmatched_pred - number of unmatched predicted codes
    '''
    
    # make sure column names are correct
    codes_df = codes_df.rename(columns={codes_df.columns[0]:'obs',codes_df.columns[1]:'pred'})
    
    # fill in NaN with 0
    codes_df = codes_df.fillna(0)
    
    # add region to codes
    codes_df['reg_obs'] = np.floor(codes_df.obs/100000).astype(int)
    codes_df['reg_pred'] = np.floor(codes_df.pred/100000).astype(int)
    
    # get severity and severity squared
    codes_df['sev_obs'] = ((codes_df.obs*10)%10).astype(int)
    codes_df['sev_pred'] = ((codes_df.pred*10)%10).astype(int)
    
    # evaluate for exact matches
    codes_df['exact'] = codes_df.apply(lambda x: 1 if x['obs']==x['pred'] else 0, axis=1)
    
    # evaluate for same region, same severity, but not exact match
    codes_df['same_reg_same_sev'] = codes_df.apply(lambda x: 1 if ((x['exact']==0) & \
                                                                   (x['reg_obs']==x['reg_pred']) & \
                                                                   (x['sev_obs']==x['sev_pred'])) else 0, axis=1)
    
    # evaluate for same region, different severity
    codes_df['same_reg_diff_sev'] = codes_df.apply(lambda x: 1 if ((x['reg_obs']==x['reg_pred']) & \
                                                                   (x['sev_obs']!=x['sev_pred'])) else 0, axis=1)
    
    # evaluate for different region, same severity
    codes_df['diff_reg_same_sev'] = codes_df.apply(lambda x: 1 if ((x['reg_obs']!=x['reg_pred']) & \
                                                                   (x['sev_obs']==x['sev_pred'])) else 0, axis=1)
    
    # evaluate for different region, different severity, but not completely unmatched
    codes_df['diff_reg_diff_sev'] = codes_df.apply(lambda x: 1 if ((x['reg_obs']!=x['reg_pred']) & \
                                                                   (x['sev_obs']!=x['sev_pred']) & \
                                                                   (x['obs']!=0) & (x['pred']!=0)) else 0, axis=1)
    
    # evaluate for unmatched codes
    codes_df['unmatched_obs'] = codes_df.apply(lambda x: 1 if x['pred']==0 else 0, axis=1)    
    codes_df['unmatched_pred'] = codes_df.apply(lambda x: 1 if x['obs']==0 else 0, axis=1) 
        
    return codes_df


def match_stats(codes_df, ais_codes, mapped_codes=False):
    '''
    This function calculates stats of matched lists of AIS codes.  
        
    Parameter:
        codes_df - dataframe with matched list of AIS codes, two columns ('obs' and 'pred')
        
    Returns:
        dataframe with results on one row.  These stats include:
            num_obs - number of observed injuries
            num_pred - number of predicted injuries
            mais_obs - maximum AIS severity observed
            main_pred - maximum AIS severity predicted
            ISS_obs - observed ISS
            ISS_pred - predicted ISS
            exact - number of exact matches
            same_reg_same_sev - number of codes in same body region, same severity, but not exact match
            same_reg_diff_sev - number of codes in same body region, different severity 
            diff_reg_same_sev - number of codes in same body region, different severity 
            unmatched_obs - number of unmatched observed codes
            unmatched_pred - number of unmatched predicted codes
    '''     
    # evaluate matches
    codes_df = eval_matches(codes_df)
    
    #display(codes_df)
    
    # get non-zero codes
    codes_obs = codes_df[codes_df.obs!=0]['obs'].values
    codes_pred = codes_df[codes_df.pred!=0]['pred'].values
    
    # create df for results and populate with number of codes
    results = pd.DataFrame({'num_obs':[len(codes_obs)], 'num_pred':[len(codes_pred)]})
    
    #print(codes_obs)
    
    # calculate ISS
    results['iss_obs'] = calc_ISS(codes_obs, ais_codes, mapped_codes)
    results['iss_pred'] = calc_ISS(codes_pred, ais_codes, mapped_codes)
    results['iss_equal'] = [1 if results.iss_obs[0] == results.iss_pred[0] else 0]
    results['iss_16_equal'] = [1 if (results.iss_obs[0]>=16) == (results.iss_pred[0]>=16) else 0]
    
    # determine MAIS
    results['mais_obs'] = max(codes_df.sev_obs)
    results['mais_pred'] = max(codes_df.sev_pred)
    results['mais_equal'] = [1 if results.mais_obs[0] == results.mais_pred[0] else 0]
    results['mais_3_equal'] = [1 if (results.mais_obs[0]>=3) == (results.mais_pred[0]>=3) else 0]
    
    # count types of matches
    results['exact'] = sum(codes_df.exact)
    results['same_reg_same_sev'] = sum(codes_df.same_reg_same_sev)
    results['same_reg_diff_sev'] = sum(codes_df.same_reg_diff_sev)
    results['diff_reg_same_sev'] = sum(codes_df.diff_reg_same_sev)
    results['diff_reg_diff_sev'] = sum(codes_df.diff_reg_diff_sev)
    results['unmatched_obs'] = sum(codes_df.unmatched_obs)
    results['unmatched_pred'] = sum(codes_df.unmatched_pred)
    
    # calculate total unmatched (either obs or pred, and different region/different severity)
    results['unmatched'] = results.diff_reg_diff_sev[0] + results.unmatched_obs[0] + results.unmatched_pred[0] 
    
    return results

#######################################################
# Functions Below here only used in viewing attention #
#######################################################
def display_attn(attn_df, showmat=True, colorbar=True, d_only=False):
    '''
    This function displays attention weights for translations.
    Parameters:
        showmat - print attention dataframe
        attn_df - dataframe with attention weights
        colorbar - whether to add colorbar
    Returns:
        None - outputs matrix and plots
    '''
    # display matrix as dataframe
    if showmat:
        display(attn_df)
    
    # remove all non-D-codes if applicable
    if d_only:
        d_cols = [x for x in attn_df.columns  if x.startswith('D')]
        
        attn_df = attn_df[['AIS']+d_cols]
        
    # get weights
    weights = attn_df.iloc[:,1:].values

    # get icd and ais codes
    icd_codes = attn_df.columns[1:]
    ais_codes = attn_df.AIS.values      
    
    # display matrix
    plt.matshow(weights, cmap=plt.get_cmap('gray'))

    # add labels
    plt.xticks(np.arange(0, len(icd_codes), 1), icd_codes)
    plt.yticks(np.arange(0, len(ais_codes), 1), ais_codes)
    
    # add color bar if applicable
    if colorbar:
        plt.colorbar()
        
    plt.show()
    
    
def icd_mat(mat, icd_version=9):
    icd_codes = mat.columns[1:]
    print(icd_codes)
    return icd_to_text(icd_codes)
    
    
    
def icd_to_text(icd_list, icd_version=9):
    '''
    This script takes a list of ICD codes and returns the text diagnoses.
    Parameters:
        icd_list - ICD codes to translate
    Returns:
        list of text diagnoses
    '''
    # list for text
    icd_txt = []
    
    # loop through all codes
    for code in icd_list:
        # check version of ICD
        if icd_version==9:
            # remove prefix from non-E/P codes
            if (code[0] == 'E') or (code[0] == 'P'):
                icd_trans = icd9cms.search(code)
            else:
                icd_trans = icd9cms.search(code[1:])
            # check if not valid code
            if icd_trans == None:
                icd_txt.append(code + ': code not found')
                
            else:
                # if no long description use short description
                if icd_trans.long_desc == None:
                    icd_txt.append(code + ": " + icd_trans.short_desc)
                else:
                    icd_txt.append(code + ": " + icd_trans.long_desc)
    return icd_txt


def ais_to_text(ais_list, ais_lu, ais_version='2005'):
    '''
    This script takes a list of AIS codes and returns the text diagnoses.
    Parameters:
       ais_list - AIS codes to translate
    Returns:
        list of text diagnoses
    '''
    
    # list for text
    ais_txt = []
    
    # loop through all codes
    for code in ais_list:    

        if code != '</s>':
            
            # convert code to number
            code = int(code)
        
            if ais_version=='2005':
            
                if code in ais_lu.predot.values:
                
                    ais_txt.append(str(code) + ": " + ais_lu[ais_lu.predot==code].Description.values[0])
                
                else:
                
                    ais_txt.append(str(code) + ": code not found")
                
    return ais_txt


def ais_split(ais_str, ais_lu, ais_version='2005'):

    ais_split = ais_str.values[0].split(" ")

    ais_num = [int(x) for x in ais_split]

    return ais_to_text(ais_num, ais_lu)