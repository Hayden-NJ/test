import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

def oswalk_file_paths(top_path,extension=['.c000']):
    file_path_list = []
    for i in os.walk(top_path):
        if i[2]:
            for j in i[2]:
                if os.path.splitext(j)[-1] in extension:
                    concatenate_path = os.path.join(i[0],j)
                    file_path_list.append(concatenate_path)
    return file_path_list

def split_df(paths,per_n,out_folder,split_a,split_b,columns):
    div = divmod(len(paths), per_n)
    if div[1] != 0:
        iters = div[0] + 1
    else:
        iters = div[0]
    for i in range(iters):
        gc.collect()
        sub_folder = os.path.join(out_folder,f'temp_{i}of{iters}')
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder, mode=0o777)
            print(f'creat folder_{sub_folder}')
        else:
            print(f'{sub_folder} exists already')
            continue
        sub_paths = paths[per_n*(i): per_n*(i+1)]
        df_list = []
        for k,path in enumerate(sub_paths):
            df = pd.read_parquet(path)
            if k%10 == 9:
                print(f'load dataframe{k}')
            df_list.append(df)
        con_df = pd.concat(df_list, axis=0)
        print('con_df_shape',con_df.shape)
        split_a_unique=con_df[split_a].unique()
        print(split_a_unique)
        sub_df_shape = []
        for tid in split_a_unique:
            c_df = con_df[con_df['train_id'].isin([tid])]
            split_b_unique=c_df[split_b].unique()
            for vid in tqdm(split_b_unique):
                cc_df = c_df[c_df['v_id'].isin([vid])][columns]
                sub_df_shape.append(cc_df.shape[0])
                out_file = os.path.join(sub_folder, f'temp_iter{i}_{tid}_v{vid}id.csv')
                cc_df.to_csv(out_file,index=False)
        print('cut_df_shape', sum(sub_df_shape))


split_a = 'train_id'
split_b = 'v_id'
columns = ['signal_value','vtime']
out_folder = '/data/zc/split_vid/temp'
split_df(paths,30,out_folder,split_a,split_b,columns)


