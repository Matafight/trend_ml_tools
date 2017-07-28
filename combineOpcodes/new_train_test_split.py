# _*_ coding:utf-8_*_
import os
import numpy as np
import pandas as pd
from shutil import copyfile,move
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










def redis_files(train_opcode,test_opcode,df_train,df_test):
    train_pos_files = os.listdir(train_opcode[0])
    train_neg_files = os.listdir(train_opcode[1])
    train_files = train_pos_files+train_neg_files
    
    print('original %f training files'%len(train_files))
    test_pos_files = os.listdir(test_opcode[0])
    test_neg_files = os.listdir(test_opcode[1])
    test_files = test_pos_files+test_neg_files
    print('original %f testing files'%len(test_files))


    df_train_files = df_train['id'].values
    df_test_files = df_test['id'].values

    test2train=[]
    train2test=[]
    missing_training=[]
    missing_testing=[]
    for item in tqdm(df_train_files):
        item +='.opcode'
        if item in train_files:
            pass
        elif item in test_files:
            #需要知道这个样本正例还是负例
            #如果是正例
            if df_train.loc[df_train['id']==item[:-7],'malware'].values[0] == 1:
                #剪切到train_opcode[0]目录下
                #原路径是test_opcode[0]
                source_path = os.path.join(test_opcode[0],item)
                dest_path = os.path.join(train_opcode[0],item)
                move(source_path,dest_path)
            else:
                #剪切到train_opcode[1]下
                #原路径是test_opcode[1]
                source_path = os.path.join(test_opcode[1],item)
                dest_path = os.path.join(train_opcode[1],item)
                move(source_path,dest_path)
            test2train.append(item)
        else:
            missing_training.append(item)
    print('test to train:%f'%len(test2train))

    for item in tqdm(df_test_files):
        item += '.opcode'
        if item in test_files:
            pass
        elif item in train_files:
            if df_test.loc[df_test['id']==item[:-7],'malware'].values[0] == 1:
                source_path = os.path.join(train_opcode[0],item)
                dest_path = os.path.join(test_opcode[0],item)
                move(source_path,dest_path)
            else:
                source_path = os.path.join(train_opcode[1],item)
                dest_path = os.path.join(test_opcode[1],item)
                if not os.path.isfile(source_path):
                    source_path = os.path.join(train_opcode[0],item)
                move(source_path,dest_path)

            train2test.append(item)
        else:
            missing_testing.append(item)
    print('train to test:%f'%len(train2test))

#假设已经做好training testing 划分了，纠正标号
def correct_labels(train_opcode,test_opcode,df_train,df_test):
    train_pos_files = os.listdir(train_opcode[0])
    train_neg_files = os.listdir(train_opcode[1])
    train_files = train_pos_files+train_neg_files
    
    print('original %f training files'%len(train_files))
    test_pos_files = os.listdir(test_opcode[0])
    test_neg_files = os.listdir(test_opcode[1])
    test_files = test_pos_files+test_neg_files
    print('original %f testing files'%len(test_files))


    df_train_files = df_train['id'].values
    df_test_files = df_test['id'].values

    test2train=[]
    train2test=[]
    missing_training=[]
    missing_testing=[]
    #for item in tqdm(df_train_files):
    #    item +='.opcode'
    #    if item in train_files:
    #        #判断label
    #        if df_train.loc[df_train['id']==item[:-7],'malware'].values[0] == 1:
    #            supposed_path = os.path.join(train_opcode[0],item)
    #            real_path = os.path.join(train_opcode[1],item)
    #            if not os.path.isfile(supposed_path):
    #                if os.path.isfile(real_path):
    #                    move(real_path,supposed_path)
    #                    print('train wrong label')
    #        else:
    #            supposed_path = os.path.join(train_opcode[1],item)
    #            real_path = os.path.join(train_opcode[0],item)
    #            if not os.path.isfile(supposed_path):
    #                if os.path.isfile(real_path):
    #                    move(real_path,supposed_path)
    #                    print('train wrong label')


    #    elif item in test_files:
    #        #需要知道这个样本正例还是负例
    #        #如果是正例
    #        #if df_train.loc[df_train['id']==item[:-7],'malware'].values[0] == 1:
    #        #    #剪切到train_opcode[0]目录下
    #        #    #原路径是test_opcode[0]
    #        #    source_path = os.path.join(test_opcode[0],item)
    #        #    dest_path = os.path.join(train_opcode[0],item)
    #        #    move(source_path,dest_path)
    #        #else:
    #        #    #剪切到train_opcode[1]下
    #        #    #原路径是test_opcode[1]
    #        #    source_path = os.path.join(test_opcode[1],item)
    #        #    dest_path = os.path.join(train_opcode[1],item)
    #        #    move(source_path,dest_path)
    #        test2train.append(item)
    #    else:
    #        missing_training.append(item)

    for item in tqdm(df_test_files):
        item += '.opcode'
        if item in test_files:
            if df_test.loc[df_test['id']==item[:-7],'malware'].values[0] == 1:
                supposed_path = os.path.join(test_opcode[0],item)
                real_path = os.path.join(test_opcode[1],item)
                if not os.path.isfile(supposed_path):
                    if os.path.isfile(real_path):
                        move(real_path,supposed_path)
                        print('test wrong label')
            else:
                supposed_path = os.path.join(test_opcode[1],item)
                real_path = os.path.join(test_opcode[0],item)
                if not os.path.isfile(supposed_path):
                    if os.path.isfile(real_path):
                        move(real_path,supposed_path)
                        print('test wrong label')

        elif item in train_files:
            #if df_test.loc[df_test['id']==item[:-7],'malware'].values[0] == 1:
            #    source_path = os.path.join(train_opcode[0],item)
            #    dest_path = os.path.join(test_opcode[0],item)
            #    move(source_path,dest_path)
            #else:
            #    source_path = os.path.join(train_opcode[1],item)
            #    dest_path = os.path.join(test_opcode[1],item)
            #    if not os.path.isfile(source_path):
            #        source_path = os.path.join(train_opcode[0],item)
            #    move(source_path,dest_path)

            train2test.append(item)
        else:
            missing_testing.append(item)
    print('train to test:%f'%len(train2test))
    print('missing testing %f'%len(missing_testing))

if __name__=='__main__':
    train_path = '../datas/Partition/train-test-set-0727/train.csv'
    test_path = '../datas/Partition/train-test-set-0727/test.csv'
    df_train,df_test = train_test_df(train_path,test_path)
    print('finish reading dataframe files...')

    #os.system(r'net use p: \\10.64.24.50\Shaocheng_Guo')
    #os.chdir('p:')
    test_pos_path = '../datas/newdownload/combined/201707/malicious/test/1'
    test_neg_path = '../datas/newdownload/combined/201707/normal/test/0'
    train_pos_path = '../datas/newdownload/combined/201707/malicious/train/1'
    train_neg_path = '../datas/newdownload/combined/201707/normal/train/0'
   

    #test_pos_path = '../datas/newdownload/combined/201707/malicious/test/1'
    #test_neg_path = '../datas/newdownload/combined/201707/normal/test/0'
    #train_pos_path = '../datas/newdownload/combined/201707/malicious/train/1'
    #train_neg_path = '../datas/newdownload/combined/201707/normal/train/0'

    train_opcode = [train_pos_path,train_neg_path]
    test_opcode = [test_pos_path,test_neg_path]
    redis_files(train_opcode,test_opcode,df_train,df_test)    
    #correct_labels(train_opcode,test_opcode,df_train,df_test)






