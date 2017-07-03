# _*_ coding:utf-8 _*_
import pandas as pd
import os
import xgboost as xgb
from os import listdir
import numpy as np
from os.path import isfile,join
import argparse
import ConfigParser

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()


def main_func(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    source_dirs = cf.get('source','source_dirs')
    print(source_dirs.split(','))
    source_dirs = source_dirs.split(',')
    train_labels_path = cf.get('label','train_labels_path')
    test_labels_path=cf.get('label','test_labels_path')
    print(source_dirs) 
    dest_dirs = cf.get('dest','dest_dirs')
    data_df = read_features(source_dirs)
    read_labels(data_df,train_labels_path,test_labels_path,dest_dirs)

def read_features(dirs):
    for row,cur_dir in enumerate(dirs):
        print(cur_dir)
        only_files = [f for f in listdir(cur_dir)]

        fir_ind = 0
        if row == 0:
            while os.path.isdir(only_files[fir_ind]):
                fir_ind +=1
            data = pd.read_csv(cur_dir+only_files[fir_ind])
            cur_only_files = only_files[fir_ind+1:]
        else:
            cur_only_files = only_files
        count = 0
        datalist = []
        for mfile in cur_only_files:
            if not os.path.isdir(cur_dir+mfile):
                count +=1
                mypath = cur_dir+mfile
                try:
                    data1 = pd.read_csv(mypath)
                except:
                    print(mypath)
                data1.drop('id',axis=1,inplace=True)
                datalist.append(data1)
    datalist.append(data)
    data_df = pd.concat(datalist,axis=1)
    return data_df

def read_labels(data,train_labels_path,test_labels_path,dest_dirs):
    train_input  = pd.read_csv(train_labels_path)
    test_input = pd.read_csv(test_labels_path)
    train_index = train_input['id']
    train_label = train_input['malware']

    train_data = data.loc[data['id'].isin(train_index)]
    test_index = test_input['id']
    test_label = test_input['malware']
    test_data = data.loc[data['id'].isin(test_index)]

    train_data.index= train_label.index
    train_data['label'] = train_label.values
    test_data.index = test_label.index
    test_data['label'] = test_label.values
    test_data.to_csv(dest_dirs+'/test_flags_normalized.csv',index = False)
    train_data.to_csv(dest_dirs+'/train_flags_normalized.csv',index = False)




if __name__ == "__main__":
    parser = arg_parser()
    main_func(parser.conf)

