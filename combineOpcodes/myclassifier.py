# _*_ coding:utf-8_*_
import os
import numpy as np
import pandas as pd
from shutil import copyfile
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
def train_test_df(train_path,test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train,df_test

def search_copy(files,source_path,dest_path):
    for item in files:
        s_path = source_path+item
        d_path = dest_path+item
        copyfile(s_path,d_path)

def files_copy(df_train,df_test,train_path,test_path):
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    print(len(train_files))
    short_name_train = [i[:-7] for i in train_files]
    short_name_test = [i[:-7] for i in test_files]
    print('number of training files:%f'%len(short_name_train))
    print('number of testing files:%f'%len(short_name_test))
    print('total files %f'%len(short_name_train)+len(short_name_test))
    for i,item in enumerate(short_name_train):
        df_item = df_train.loc[df_train['id']==item]
        if(df_item['malware'].values[0]==0):
            dest_train = './datas/train/0/'
        else:
            dest_train = './datas/train/1/'
        if not os.path.isdir(dest_train):
            os.mkdir(dest_train)
        source_train = train_path+'/'+train_files[i]
        dest_train += train_files[i]
        copyfile(source_train,dest_train)

    for i,item in enumerate(short_name_test):
        df_item = df_test.loc[df_test['id']==item]
        if(df_item['malware'].values[0]==0):
            dest_test = './datas/test/0/'
        else:
            dest_test = './datas/test/1/'
        if not os.path.isdir(dest_test):
            os.mkdir(dest_test)
        source_test = test_path+'/'+test_files[i]
        dest_test += test_files[i]
        copyfile(source_test,dest_test)

def file_copy_v2(source_path,dest_path):
    files = os.listdir(source_path)
    for item in files:
        s_path = source_path + item
        d_path = dest_path + item
        copyfile(s_path,d_path)

    
def difference(df,opcode_path,iftrain=True):
    #opcode_path is a dir list
    all_files=[]
    for item in tqdm(opcode_path):
        files = os.listdir(item)
        files = [i[:-7] for i in files]
        all_files +=files
    print(len(all_files))
    df_files = df['id'].values
    sub_files=[]
    for i in tqdm(df_files):
        if i not in all_files:
            sub_files.append(i)
    print('num of difference %f'%len(sub_files))
    malware_sub_files=[]
    normal_sub_files=[]
    for item in tqdm(sub_files):
        if df.loc[df['id']==item,'malware'].values[0] == 1:
            malware_sub_files.append(item)
        else:
            normal_sub_files.append(item)
    tar_path = './datas/abnormal/'
    if not os.path.isdir(tar_path):
        os.mkdir(tar_path)
    if iftrain:
        add_path = 'train/'
    else:
        add_path = 'test/'
    tar_path += add_path
    if not os.path.isdir(tar_path):
        os.mkdir(tar_path)
    malware_ones = np.ones(len(malware_sub_files))
    malware_df = pd.DataFrame({'id':malware_sub_files,'malware':malware_ones})
    normal_zeros = np.zeros(len(normal_sub_files))
    normal_df = pd.DataFrame({'id':normal_sub_files,'malware':normal_zeros})
    malware_df.to_csv(tar_path+'malware.csv',index=False)
    normal_df.to_csv(tar_path+'normal.csv',index=False)


def compare_df_files(df,iftrain = True):
    opcode_path = []
    if iftrain:
        opcode_path.append('../datas/opcode-201707/201705/malicious/train/1/')
        opcode_path.append('../datas/opcode-201707/201705/normal/train/0/')
        opcode_path.append('../datas/opcode-201707/201706-classified/train/0/')
        opcode_path.append('../datas/opcode-201707/201706-classified/train/1/')
    else:
        opcode_path.append('../datas/opcode-201707/201705/malicious/test/1/')
        opcode_path.append('../datas/opcode-201707/201705/normal/test/0/')
        opcode_path.append('../datas/opcode-201707/201706-classified/test/0/')
        opcode_path.append('../datas/opcode-201707/201706-classified/test/1/')
    difference(df,opcode_path,iftrain)

        
#将得到的缺失opcode 的id 与 已知缺失的id对比
def compare_missing(s_path,c_path=None):
    df_comp = pd.read_csv(s_path)
    print(df_comp['abnormal_opcode'].sum(axis = 0))




def multi_processing_copy(source_path,dest_path):
    files = os.listdir(source_path)
    existing_files=os.listdir(dest_path)
    files=[i for i in files if i not in existing_files]
    #using 5 threads
    each_num = len(files)/5
    total_splits=[]
    for i in range(5):
        nl = files[i*each_num:(i+1)*each_num]
        total_splits.append(nl)
    nl = files[5*each_num:]
    print(nl)
    total_splits.append(nl)
    partial_copy = partial(search_copy,source_path = source_path,dest_path = dest_path)
    pool = Pool(5)
    print('starting...')
    print(type(total_splits))
    print(len(total_splits))
    pool.map(partial_copy,total_splits)
    pool.close()
    pool.join()



    

if __name__=='__main__':
    train_path = '../datas/opcode-201707/train-test-set/train.csv'
    test_path = '../datas/opcode-201707/train-test-set/test.csv'
    train_opcode_path = '../datas/opcode-201707/201706/train'
    test_opcode_path = '../datas/opcode-201707/201706/test'
    df_train,df_test = train_test_df(train_path,test_path)
    #handle 201706 opcode into four dirs 
    #files_copy(df_train,df_test,train_opcode_path,test_opcode_path)



    #source_path_malware_test = '../datas/opcode-201707/201705/malicious/test/1/'
    #source_path_malware_train = '../datas/opcode-201707/201705/malicious/train/1/'
    #source_path_normal_test = '../datas/opcode-201707/201705/normal/test/0/'
    #source_path_normal_train = '../datas/opcode-201707/201705/normal/train/0/'
    #dest_path_malware_test = './datas/test/1/'
    #dest_path_malware_train = './datas/train/1/'
    #dest_path_normal_test='./datas/test/0/'
    #dest_path_normal_train='./datas/train/0/'
    #file_copy_v2(source_path_malware_test,dest_path_malware_test)
    #file_copy_v2(source_path_malware_train,dest_path_malware_train)
    #file_copy_v2(source_path_normal_test,dest_path_normal_test)
    #file_copy_v2(source_path_normal_train,dest_path_normal_train)

    #print('starting multi...')
    #multi_processing_copy(source_path_normal_train,dest_path_normal_train)

    # find the ids without corresponding opcode
    compare_df_files(df_train,True)
    compare_df_files(df_test,False)

    #compare finded abnormal opcode with known ids
    #existing_abnormal_path = '../datas/opcode-201707/add_2017-07-06/abnormal_opcode.csv'
    #compare_missing(existing_abnormal_path)



