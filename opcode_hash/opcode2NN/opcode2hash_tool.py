#coding:utf-8
import argparse
import ConfigParser
import pandas as pd
import numpy as np
import hashlib
import sha3
import abc
import os
import re
from tqdm import tqdm
from tools import dirlist2,get_csr_labels,save2xgdata,to_NN2
from opcode2hash import batch_str2bits
from multiprocessing import Pool
import multiprocessing
from functools import partial

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    dirs = cf.get('source','source_dirs')
    train_path = cf.get('source','train')
    test_path = cf.get('source','test')
    algorithm = cf.get('setup','algorithm')
    bits = int(cf.get('setup','bits'))
    split_train_test = int(cf.get('setup','split_train_test'))
    split_good_bad = int(cf.get('setup','split_good_bad'))
    ignore_split = int(cf.get('setup','ignore_split'))
    dest_dir = cf.get('dest','dest_dirs')

    param = {'dirs':dirs,'dest_dir':dest_dir,'ignore_split':ignore_split,'algorithm':algorithm,'bits':bits,'train_path':train_path,'test_path':test_path,'split_train_test':split_train_test,'split_good_bad':split_good_bad}
    return param


def opcode2files(filelist,labels,tar_dir,sub_dir_name,param):

    n_path = os.path.join(tar_dir,sub_dir_name)
    if not os.path.isdir(n_path):
        os.mkdir(n_path)

    if 0==len(filelist):
        return
    batch_map = batch_str2bits()
    file_path = batch_map.listfiles2csv(filelist,labels,n_path,param)
    #to_NN
    df = pd.read_csv(file_path)
    new_col = df.columns[df.columns!='opcode_name']
    data = df[new_col].as_matrix()
    paths = df['opcode_name'].values.tolist()
    num_col = data.shape[1]
    features = data[:,:num_col-1]
    labels = data[:,num_col-1]
    paths = [os.path.normcase(os.path.abspath(i)) for i in paths]
    paths = np.array(paths)
    NN_name = os.path.join(n_path,'NN_features.txt')
    NN_label_name = os.path.join(n_path,'NNAI.txt')
    to_NN2(data=features,label=labels,path=paths,NN_name=NN_name,NN_label_name=NN_label_name)
    #delete csv files
    os.remove(file_path)
    
# get train test opcode from train_test_split.csv
def get_train_test(param):
    df_train = pd.read_csv(param['train_path'])
    df_test = pd.read_csv(param['test_path'])

    df_train_files = df_train['id'].values
    df_test_files = df_test['id'].values


    train_opcode_pos_path = []
    train_opcode_neg_path = []
    test_opcode_pos_path = []
    test_opcode_neg_path = []
    train_missing_opcode = []
    test_missing_opcode=[]
    all_files = {}
    all_files = dirlist2(param['dirs'],all_files)
    all_files_keys = all_files.keys()
    #可以采用多进程加速
    print('Searching files in train.csv')
    for item in tqdm(df_train_files):
        #judge if item exists
        item2 = item + '.opcode'
        if item2 in all_files_keys:
            df_item = df_train.loc[df_train['id']==item]
            if(df_item['malware'].values[0]==0):
                train_opcode_neg_path.append(all_files[item2])
            else:
                train_opcode_pos_path.append(all_files[item2])
        else:
            train_missing_opcode.append(item)

    print('Missing opcode of training: %f'%len(train_missing_opcode))
    print('Searching files in test.csv')
    for item in tqdm(df_test_files):
        item2 = item + '.opcode'
        if item2 in all_files_keys:
            df_item = df_test.loc[df_test['id']==item]
            if(df_item['malware'].values[0]==0):
                test_opcode_neg_path.append(all_files[item2])
            else:
                test_opcode_pos_path.append(all_files[item2])
        else:
            test_missing_opcode.append(item)
    print('Missing opcode of testing: %f'%len(test_missing_opcode))
        
    label_train_neg = len(train_opcode_neg_path)*[0] 
    label_train_pos =  len(train_opcode_pos_path)*[1] 
    label_test_neg =  len(test_opcode_neg_path)*[0] 
    label_test_pos = len(test_opcode_pos_path)*[1]
    if param['split_train_test'] == 0 or param['ignore_split']==1:
        if param['split_good_bad'] ==0 or param['ignore_split']==1:
            #save all in one file, all_opcode.csv
            all_opcode_files = train_opcode_neg_path + train_opcode_pos_path + test_opcode_neg_path + test_opcode_pos_path
            label_list = label_train_neg + label_train_pos+label_test_neg + label_test_pos
            opcode2files(all_opcode_files,label_list,param['dest_dir'],'all_opcode_NN',param)
        if param['split_good_bad'] ==1 or param['ignore_split']==1:
            #split int good and bad
            #normal.csv / bad.csv
            good_opcode_files = train_opcode_neg_path + test_opcode_neg_path
            label_good = label_train_neg + label_test_neg
            bad_opcode_files = train_opcode_pos_path + test_opcode_pos_path
            label_bad = label_train_pos + label_test_pos
            opcode2files(good_opcode_files,label_good,param['dest_dir'],'normal_opcode_NN',param)
            opcode2files(bad_opcode_files,label_bad,param['dest_dir'],'malware_opcode_NN',param)
    if param['split_train_test'] == 1 or param['ignore_split']==1:
        if param['split_good_bad'] ==0 or param['ignore_split']==1:
            #train_opcode.csv, test_opcode.csv
            train_opcode_files = train_opcode_pos_path + train_opcode_neg_path
            label_train = label_train_pos + label_train_neg

            test_opcode_files = test_opcode_pos_path + test_opcode_neg_path
            label_test = label_test_pos + label_test_neg
            opcode2files(train_opcode_files,label_train,param['dest_dir'],'train_opcode_NN',param)
            opcode2files(test_opcode_files,label_test,param['dest_dir'],'test_opcode_NN',param)
        if param['split_good_bad'] ==1 or param['ignore_split']==1:
            #train_malware.csv,train_malware.csv,test_malware.csv,test_normal.csv
            opcode2files(train_opcode_pos_path,label_train_pos,param['dest_dir'],'train_malware_NN',param)
            opcode2files(train_opcode_neg_path,label_train_neg,param['dest_dir'],'train_normal_NN',param)
            opcode2files(test_opcode_pos_path,label_test_pos,param['dest_dir'],'test_malware_NN',param)
            opcode2files(test_opcode_neg_path,label_test_neg,param['dest_dir'],'test_normal_NN',param)





if __name__ == '__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    get_train_test(param)