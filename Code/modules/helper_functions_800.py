import torch
import random
import pandas as pd
import math


# Batch size is the NUMBER OF PATIENTS we are analyzing at once
def get_list_of_spliced_matrices(row, col, data, pt_num, dic):
    # Start at row 0
    current_row = 0
    end = 0
    # If pt_num is < batch size then 1 is the num_sparse_matrices, otherwise divy up into batches (sparse matrices)
    if batch_size < pt_num:
        # So if batch size >= pt_num then divide pt_num by batch size, that's the number of sparse matrices
        num_sparse_matrices = int(math.ceil(pt_num/(batch_size)))
    else:
        num_sparse_matrices = 1
        
    # Now go thru the determined number of sparse matrices and add batches of matrices
    sparse_batch_list = []
    for i in range(0, num_sparse_matrices):
        current_pt = row[current_row]
        # If we add batch_size to current row and we are below pt_num then good, lets keep going
        # Now lets add a smaller sparse coo tensor to the batch list
        rows = []
        cols = []
        datas = []
        last_pt = 0
        if (current_pt + batch_size) < pt_num:
            end = row.index((current_pt + batch_size))
            
        # Else end is pt_num
        else:
            end = row.index(pt_num-1)+1

        while current_row < end:
            subtractor = ((batch_size*i))
            rows.append((row[current_row] - subtractor))
            last_pt = (row[current_row] - subtractor)
            cols.append(col[current_row])
            datas.append(data[current_row])
            current_row += 1
            
        missing_codes = list(map(int, hlp.find_missing(cols, dic)))
        cols.extend(missing_codes) # Add last code instead 
        #print("Missing codes length: " + str(len(missing_codes)))
        for i in range(len(missing_codes)):
            rows.append(last_pt)
            datas.append(0)
        
        # If list len is not batch size, add fake pts with no codes
        if last_pt != batch_size and num_sparse_matrices > 1:
            for imaginary_pt in range(last_pt, batch_size):
                rows.append(imaginary_pt)
                cols.append(0)
                datas.append(0)
                   
        tmp_tensor = torch.sparse_coo_tensor([rows,cols], 
                                datas, 
                                dtype=torch.float)
                
        sparse_batch_list.append(tmp_tensor.detach().clone())
        del rows
        del cols
        del datas
        del tmp_tensor
            
    return sparse_batch_list



def get_list_avg(list_):
    return(sum(list_) / len(list_))

# Function that gets submatrix in dense format
def get_dense_submat(full_sparse_mat, start, end=False):
    # If only getting 1 row
    if end is False:
        return torch.index_select(full_sparse_mat.cuda(),0,torch.tensor(start).cuda())
    # Else return subsection
    return torch.index_select(full_sparse_mat.cuda(),0,torch.tensor(list(range(start, end))).cuda()).to_dense()


def get_unique_codes(dfs):
    # empty list for codes
    unique_codes = []
    # loop through all dataframes
    for df in dfs:  
        # loop through all rows
        for i in range(0,len(df)):
             # extract line and split into 
            line = df.iloc[i].str.split(" ").values[0]
            # loop through terms
            for j in line:
                # check if terms in already in dictionary
                if j not in unique_codes:
                    # add term to dictionary
                    unique_codes.append(j)

    # sort values
    unique_codes.sort()
    return unique_codes
            
    
def find_missing(list_, dict_):
    values = dict_.values()
    return(list(set(list(values)).difference(list_)))
    
    
# Here were creating a three lists
# PT    Code    Value
# 1      2        1
# 1      15       1

# The pt maps to pt number
# Code maps to what information from the dict maps to
# Value will be 1 to say that the patient has this code (?is this necessary?)
# These are to be plugged into torch.sparse_coo_tensor() to create a sparse matrix
def decode_df_coo(df, dic):
    # decoded array
    row = []
    col = []
    data = []
    last = 0
    # loop through all rows (this would be the patients)
    for i in range(len(df)):
        last = i
        # extract line and split into various values if multiple for example (age, ecode, pcode, dcode)
        line = df.iloc[i].str.split(" ").values[0]
        # decoded line
        line_d = []
        # loop through terms for each patient (age, ecode, pcode, dcode, ais_codes. etc)
        for j in line:
            # create new decoded line
            row.append(i) # Add a row or another row for that patient
            col.append(dic[j]) # Add the mapping that the code maps to (for example A10_20 may map to 1)
            data.append(1) # Append a 1 to encode that the patient has this code
            
    if len(col) != len(dic):
        # We need to add the extra columns that are not already part of the dataset
        #start = time.process_time()
        #print('Getting missing icd codes', flush=True)
        missing_icd_codes = list(map(int, find_missing(col, dic)))
        #print("Time to get missing icd: " + str(time.process_time() - start))
        col.extend(missing_icd_codes)
        print("Missing codes length: " + str(len(missing_icd_codes)))
        for i in range(len(missing_icd_codes)):
            row.append(last)
            data.append(0)
    
    # return dictionary and decoded array
    return row, col, data


# Function to test sparse matrices against orig_dfs
def test_sparse_matrix(sparse_matrix, orig_df, dict_):
    # Get the icd codes by patient, store in list
    num_slice = 3000
    num_pts = len(sparse_matrix)
    data = []
    if num_pts > 50000:
        end = num_pts - num_slice
        random_index = random.sample(range(0, end), 1)[0]
        for i in range(random_index, random_index+num_slice):
            code_string = ''
            for index in sparse_matrix[i].coalesce().indices()[0]:
                if code_string == '':
                    code_string = code_string + dict_[int(index)]
                else:
                    code_string = code_string + ' ' + dict_[int(index)]
            data.append(code_string)
        
        # Get the original rows by random index
        orig_df = orig_df.loc[random_index:(random_index+num_slice)]
        orig_df = orig_df.reset_index()
        orig_df = orig_df.drop('index', axis=1)
    else:
        for i in range(0, num_pts):
            code_string = ''
            for index in sparse_matrix[i].coalesce().indices()[0]:
                if code_string == '':
                    code_string = code_string + dict_[int(index)]
                else:
                    code_string = code_string + ' ' + dict_[int(index)]
            data.append(code_string)
        
    # Create a df
    matrix_df = pd.DataFrame(data)
    matrix_df.columns = ['icd_code']
    ne = (matrix_df != orig_df).any(1)
    return ne, matrix_df

# Function to test sparse matrices against orig_dfs
def test_sparse_matrices(sparse_matrices, orig_df, dict_):
    # Get the icd codes by patient, store in list
    data = []
    for mat in sparse_matrices: 
        num_pts = len(mat)
        for i in range(0, num_pts):
            code_string = ''
            for index in mat[i].coalesce().indices()[0]:
                if code_string == '':
                    code_string = code_string + dict_[int(index)]
                else:
                    code_string = code_string + ' ' + dict_[int(index)]
            data.append(code_string)
        
    # Create a df
    matrix_df = pd.DataFrame(data)
    matrix_df.columns = ['icd_code']
    ne = (matrix_df != orig_df).any(1)
    return ne, matrix_df


def print_diff(ne):
    diff_flag = False
    i=0
    for flag in ne:
        if flag == 'True':
            diff_flag = True
            print("Difference at index " + str(i))
        i+=1

    if diff_flag == False:
        print("Dataframes are identical")