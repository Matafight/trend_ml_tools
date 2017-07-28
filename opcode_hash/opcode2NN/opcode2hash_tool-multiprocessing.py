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
import datetime
import time
import shutil
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    dirs = cf.get('source_opcode2hash','source_dirs')
    train_path = cf.get('source_opcode2hash','train')
    test_path = cf.get('source_opcode2hash','test')
    algorithm = cf.get('setup_opcode2hash','algorithm')
    bits = int(cf.get('setup_opcode2hash','bits'))
    split_train_test = int(cf.get('setup_opcode2hash','split_train_test'))
    split_good_bad = int(cf.get('setup_opcode2hash','split_good_bad'))
    ignore_split = int(cf.get('setup_opcode2hash','ignore_split'))
    dest_dir = cf.get('dest_opcode2hash','dest_dirs')

    algorithm_path = algorithm +'_'+str(bits)
    dest_dir = os.path.join(dest_dir,algorithm_path)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    param = {'dirs':dirs,'dest_dir':dest_dir,'ignore_split':ignore_split,'algorithm':algorithm,'bits':bits,'train_path':train_path,'test_path':test_path,'split_train_test':split_train_test,'split_good_bad':split_good_bad}
    return param


def opcode2files(filelist,labels,tar_dir,sub_dir_name,param):

    n_path = os.path.join(tar_dir,sub_dir_name)
    if not os.path.isdir(n_path):
        os.mkdir(n_path)

    if 0==len(filelist):
        return None,n_path
    batch_map = batch_str2bits()
    file_path = batch_map.listfiles2csv(filelist,labels,n_path,param)
    return file_path,n_path
    #delete csv files
    #os.remove(file_path)
    
def csv2NN(file_path,n_path):
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
    NN_label_name = os.path.join(n_path,'NN_AI.txt')
    to_NN2(data=features,label=labels,path=paths,NN_name=NN_name,NN_label_name=NN_label_name)

    
#n_path means the place where you want to save this NN.txt
def combineCSV(file_paths,n_path):
    file_paths = [i for i in file_paths if i!=None]
    if file_paths == None  or len(file_paths)==0:
        return None
    df_list = []
    for item in file_paths:
        df = pd.read_csv(item)
        df_list.append(df)
    all_df = pd.concat(df_list,axis=0)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    total_path = os.path.join(n_path,'total_'+time_str+".csv")
    all_df.to_csv(total_path,index=False)
    csv2NN(total_path,n_path)
    os.remove(total_path)


    
def search_files(i,df,all_files):
    df_item = df.iloc[i]
    fn = df_item['id'] + '.opcode'
    flag = -1
    ret_val = None
    if fn in all_files.keys():
        if df_item['malware']== 0:
            ret_val = all_files[fn]
            flag = 0
        else:
            ret_val = all_files[fn]
            flag = 1
    else:
        flag = -1
        ret_val = fn
    return  ret_val,flag

def start_process():
    #print 'Starting',multiprocessing.current_process().name
    pass

def multi_handle(df,all_files):
    starttime = datetime.datetime.now()
    neg_list = []
    pos_list = []
    missing_list = []
    partial_search = partial(search_files,df = df,all_files = all_files)
    pool = Pool(processes=3,initializer=start_process)
    ret = pool.map(partial_search,range(df.shape[0]))
    pool.close()
    pool.join() 
    for item in ret:
        if item[1] == 0:
            neg_list.append(item[0])
        elif item[1] == 1:
            pos_list.append(item[0])
        else:
            missing_list.append(item[0])
    endtime = datetime.datetime.now()
    print 'lasting time: '+str((endtime - starttime).seconds)+"  seconds"
    return pos_list,neg_list,missing_list

    
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
    print('Multi processing Searching files in train.csv')
    train_opcode_pos_path,train_opcode_neg_path,train_missing_opcode = multi_handle(df_train,all_files)
    print('Found training opcode: %f'%(len(train_opcode_pos_path)+len(train_opcode_neg_path)))
    print('Missing opcode of training: %f'%len(train_missing_opcode))
    print('Multi processing Searching files in test.csv')
    test_opcode_pos_path,test_opcode_neg_path,test_missing_opcode = multi_handle(df_test,all_files)
    print('Missing opcode of testing: %f'%len(test_missing_opcode))
    print('Found testing opcode: %f'%(len(test_opcode_pos_path)+len(test_opcode_neg_path)))

    label_train_neg = len(train_opcode_neg_path)*[0] 
    label_train_pos =  len(train_opcode_pos_path)*[1] 
    label_test_neg =  len(test_opcode_neg_path)*[0] 
    label_test_pos = len(test_opcode_pos_path)*[1]

    train_pos_csv_path,upper_train_pos_dir= opcode2files(train_opcode_pos_path,label_train_pos,param['dest_dir'],'train_malware_opcode_NN',param)
    train_neg_csv_path,upper_train_neg_dir = opcode2files(train_opcode_neg_path,label_train_neg,param['dest_dir'],'train_normal_opcode_NN',param)
    test_pos_csv_path,upper_test_pos_dir = opcode2files(test_opcode_pos_path,label_test_pos,param['dest_dir'],'test_malware_opcode_NN',param)
    test_neg_csv_path,upper_test_neg_dir = opcode2files(test_opcode_neg_path,label_test_neg,param['dest_dir'],'test_normal_opcode_NN',param)


    if param['split_train_test'] == 0 or param['ignore_split']==1:
        if param['split_good_bad'] ==0 or param['ignore_split']==1:
            #save all in one file, all_opcode.csv
            all_csv_path = [train_pos_csv_path,train_neg_csv_path,test_pos_csv_path,test_neg_csv_path]
            tar_path = os.path.join(param['dest_dir'],'all_opcode_NN')
            if not os.path.isdir(tar_path):
                os.mkdir(tar_path)
            combineCSV(all_csv_path,tar_path)
        if param['split_good_bad'] ==1 or param['ignore_split']==1:
            #split int good and bad
            #normal.csv / bad.csv
            normal_csv_path = [train_neg_csv_path,test_neg_csv_path]
            tar_path = os.path.join(param['dest_dir'],'all_normal_opcode_NN')
            if not os.path.isdir(tar_path):
                os.mkdir(tar_path)
            combineCSV(normal_csv_path,tar_path)
            malware_csv_path = [train_pos_csv_path,test_pos_csv_path]
            tar_path = os.path.join(param['dest_dir'],'all_malware_opcode_NN')
            if not os.path.isdir(tar_path):
                os.mkdir(tar_path)
            combineCSV(malware_csv_path,tar_path)

    if param['split_train_test'] == 1 or param['ignore_split']==1:
        if param['split_good_bad'] ==0 or param['ignore_split']==1:
            #train_opcode.csv, test_opcode.csv
            train_csv_path = [train_neg_csv_path,train_pos_csv_path]
            tar_path = os.path.join(param['dest_dir'],'all_train_opcode_NN')
            if not os.path.isdir(tar_path):
                os.mkdir(tar_path)
            combineCSV(train_csv_path,tar_path)
            test_csv_path = [test_neg_csv_path,test_pos_csv_path]
            tar_path = os.path.join(param['dest_dir'],'all_test_opcode_NN')
            if not os.path.isdir(tar_path):
                os.mkdir(tar_path)
            combineCSV(test_csv_path,tar_path)
        if param['split_good_bad'] ==1 or param['ignore_split']==1:
            #train_malware.csv,train_malware.csv,test_malware.csv,test_normal.csv
            pass
    if(param['split_train_test'] == 0 or param['split_good_bad'] ==0) and (param['ignore_split']==0):
        shutil.rmtree(upper_train_neg_dir)
        shutil.rmtree(upper_train_pos_dir)
        shutil.rmtree(upper_test_neg_dir)
        shutil.rmtree(upper_test_pos_dir)
        






if __name__ == '__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    get_train_test(param)