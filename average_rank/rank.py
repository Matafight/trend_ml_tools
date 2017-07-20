# _*_ coding:utf-8 _*_

#需要传入目录名，之后遍历该目录下的所有文件，利用正则表达式提取类别，不过文件的命名应当符合这样的规律 index_***.txt
import matplotlib.pyplot as plt
import pandas as pd
import os
import re


def extract_names(files):
    pattern = re.compile(r'index_(\w+).txt')
    fname = []
    for item in files:
        match = pattern.match(item)
        if match!=None:
            print(match.group(1))
            fname.append(match.group(1))
    return fname

def read_ranks(path):
    tar_dir = './ranks/'+path
    if not os.path.isdir(tar_dir):
        os.mkdir(tar_dir)
    files = [f for f in os.listdir(tar_dir)]
    names = extract_names(files)
    rank_dict={}
    for i,item in enumerate(files):
        tar_dir = './ranks/'+path+'/'+item
        cur_rank_name = names[i]+'_rank' 
        cur_rank={}
        with open(tar_dir) as fh:
            lines = fh.readlines()
            rank = 1
            for line in lines:
                cur_rank[line] = rank
                rank +=1
        rank_dict[cur_rank_name] = cur_rank
    return rank_dict

def handle_missing(rank_dict):
    keys = rank_dict.keys()
    #寻找长度最大的作为基准
    max_len = 0
    max_key = None
    for item in keys:
        if len(rank_dict[item])>max_len:
            max_len = len(rank_dict[item])
            max_key = item
    
    for item in keys:
        if item != max_key:
            for feat in rank_dict[max_key]:
                if feat not in rank_dict[item]:
                    rank_dict[item][feat] = None
            print(len(rank_dict[item])) 
    return rank_dict





def comb_ranks(rank_dict,path):
    tar_dir = './ranks/'+path
    keys = rank_dict.keys()
    count = 0
    df_list = []
    for key in keys:
        cur_keys=[]
        cur_values=[]
        for item in rank_dict[key]:
            cur_keys.append(item)
            cur_values.append(rank_dict[key][item])

        df = pd.DataFrame({'features':cur_keys,key:cur_values})
        df_list.append(df)
    total_df = None
    for i,df in enumerate(df_list):
        if i==0:
            total_df = df
            continue
        static_name = df.columns[df.columns!='features'][0]
        print(df.columns)
        print(static_name)
        total_df[static_name] = 0
        for i,row in df.iterrows():
            total_df.loc[total_df['features']==row['features'],static_name] = row[static_name]
    total_df['aver_rank'] = total_df.mean(axis=1).values
    print(total_df.columns)
    total_df.sort_values(by = ['aver_rank'],axis=0,inplace=True)
    total_df.to_csv('./ranks/'+path+'/comb_rank.csv',index= False)


            

    

if __name__ == '__main__':
    path = 'version2_features_201707'
    rank_dict = read_ranks(path)
    rank_dict = handle_missing(rank_dict)
    comb_ranks(rank_dict,path)

