#coding:utf-8
#给定train test 划分,输出划分好的NN format数据

import pandas as pd
import numpy as np
import argparse
import ConfigParser
import os
from tqdm import tqdm
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    train_path = cf.get('train_test_split','train')
    test_path = cf.get('train_test_split','test')
    features_path = cf.get('source_path','source_features_path')
    labels_path = cf.get('source_path','source_labels_path')
    split_train_test = int(cf.get('source_path','split_train_test'))
    split_malware_normal = int(cf.get('source_path','split_malware_normal'))
    dest_dir = cf.get('dest_dir','dest_dir')
    param = {'train_path':train_path,
             'test_path':test_path,
             'dest_dir':dest_dir,
             'features_path':features_path,
             'labels_path':labels_path,
             'split_train_test':split_train_test,
             'split_malware_normal':split_malware_normal}
    return param

def read_csv(train_path,test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train,df_test

def retrive_opcode(str):
    str = str.strip()
    return os.path.split(str)[1]

#return a dict mapping opcode to index of list, lists contain features and labels
def read_all_data(features_path,labels_path):
    features_lines = []
    label_lines = []
    opcode2index={}
    with open(features_path) as fh:
        features_lines = fh.readlines()
        features_lines = [line.strip() for line in features_lines]
        dim = features_lines[0]
        features_lines = features_lines[1:]
    with open(labels_path) as fh:
        label_lines = fh.readlines()
        label_lines = [line.strip() for line in label_lines]
        assert len(features_lines) == len(label_lines)
        for i,line in enumerate(label_lines):
            opcode = retrive_opcode(line)
            opcode2index[opcode] = i 
    return features_lines,label_lines,dim,opcode2index

def write_files(path,tosave_list,dim = -1):
    with open(path,'w') as fh:
        if len(tosave_list) !=0:
            if dim!=-1:
                fh.write(dim+'\n')
            for item in tosave_list[:-1]:
                fh.write(item+'\n')
            fh.write(tosave_list[-1])

#类别未知label记为0，正例label记为1，负例label记为2
def redistribute_samples(dim,df_train,df_test,opcode2index,features_lines,labels_lines,param):
    train_pos_lines = []
    train_neg_lines = []
    test_pos_lines = []
    test_neg_lines = []

    train_pos_labels = []
    train_neg_labels = []
    test_pos_labels = []
    test_neg_labels = []
    missing_training = []
    missing_testing = []
    print('dealing with train.csv...')
    for ind in tqdm(range(df_train.shape[0])):
        #search in opcode2index
        item = df_train.iloc[ind]
        cur_opcode = item['id'] + '.opcode'
        if cur_opcode in opcode2index.keys():
            cur_label = item['malware']
            if cur_label == 1:
                train_pos_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_list = list(label_line)
                label_list[0] = str(1)
                label_line = ''.join(label_list)
                train_pos_labels.append(label_line)
            else:
                train_neg_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_list = list(label_line)
                label_list[0] = str(2)
                label_line = ''.join(label_list)
                train_neg_labels.append(label_line)
        else:
            missing_training.append(cur_opcode)

    print('found total %f training files'%(len(train_pos_lines)+len(train_neg_lines)))
    print('missing %f training files'%(len(missing_training)))
    print('dealing with test.csv...')
    for ind in tqdm(range(df_test.shape[0])):
        item = df_test.iloc[ind]
        cur_opcode = item['id'] + '.opcode'
        if cur_opcode in opcode2index.keys():
            cur_label = item['malware']
            if cur_label == 1:
                test_pos_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_list = list(label_line)
                label_list[0] = str(1)
                label_line = ''.join(label_list)
                test_pos_labels.append(label_line)
            else:
                test_neg_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_list = list(label_line)
                label_list[0] = str(2)
                label_line = ''.join(label_list)
                test_neg_labels.append(label_line)
        else:
            missing_testing.append(cur_opcode)
    print('found total %f testinging files'%(len(test_pos_lines)+len(test_neg_lines)))
    print('missing %f testing files'%(len(missing_testing)))
    #save into four different file
    #train_pos,train_neg,test_pos,test_neg
    if param['split_train_test'] == 1 and param['split_malware_normal'] == 1:
        train_pos_features_path  = os.path.join(param['dest_dir'],'train_malware_vec.dat')
        train_pos_labels_path = os.path.join(param['dest_dir'],'train_malware_vec.dat_AI')
        train_neg_features_path = os.path.join(param['dest_dir'],'train_normal_vec.dat')
        train_neg_labels_path = os.path.join(param['dest_dir'],'train_normal_vec.dat_AI')
        test_pos_features_path = os.path.join(param['dest_dir'],'test_malware_vec.dat')
        test_pos_labels_path = os.path.join(param['dest_dir'],'test_malware_vec.dat_AI')
        test_neg_features_path = os.path.join(param['dest_dir'],'test_normal_vec.dat')
        test_neg_labels_path = os.path.join(param['dest_dir'],'test_normal_vec.dat_AI')

        write_files(train_pos_features_path,train_pos_lines,dim)
        write_files(train_pos_labels_path,train_pos_labels)
        write_files(train_neg_features_path,train_neg_lines,dim)
        write_files(train_neg_labels_path,train_neg_labels)
        write_files(test_pos_features_path,test_pos_lines,dim)
        write_files(test_pos_labels_path,test_pos_labels)
        write_files(test_neg_features_path,test_neg_lines,dim)
        write_files(test_neg_labels_path,test_neg_labels)
    elif param['split_train_test'] == 1:
        #train test 划分
        train_features_path = os.path.join(param['dest_dir'],'train_vec.dat')
        train_labels_path = os.path.join(param['dest_dir'],'train_vec.dat_AI')
        test_features_path = os.path.join(param['dest_dir'],'test_vec.dat')
        test_labels_path = os.path.join(param['dest_dir'],'test_vec.dat_AI')

        train_features = train_pos_lines + train_neg_lines
        train_labels = train_pos_labels + train_neg_labels
        test_features = test_pos_lines + test_neg_lines
        test_labels = test_pos_labels + test_neg_labels

        write_files(train_features_path,train_features,dim)
        write_files(train_labels_path,train_labels)
        write_files(test_features_path,test_features,dim)
        write_files(test_labels_path,test_labels)
    elif param['split_malware_normal'] == 1:
        malware_features_path = os.path.join(param['dest_dir'],'malware_vec.dat')
        malware_labels_path = os.path.join(param['dest_dir'],'malware_vec.dat_AI')
        normal_features_path = os.path.join(param['dest_dir'],'normal_vec.dat')
        normal_labels_path = os.path.join(param['dest_dir'],'normal_vec.dat_AI')

        malware_features = train_pos_lines + test_pos_lines
        malware_labels = train_pos_labels + test_pos_labels

        normal_feautres = train_neg_lines + test_neg_lines
        normal_labels = train_neg_labels + test_neg_labels

        write_files(malware_features_path,malware_features,dim)
        write_files(malware_labels_path,malware_labels)
        write_files(normal_features_path,normal_feautres,dim)
        write_files(normal_labels_path,normal_labels)
    else:
        print('no split according to the parameters!')

        

    #with open(param['dest_dir'])
    
if __name__ == '__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    df_train,df_test = read_csv(param['train_path'],param['test_path'])
    features_lines,label_lines,dim,opcode2index = read_all_data(param['features_path'],param['labels_path'])
    redistribute_samples(dim,df_train,df_test,opcode2index,features_lines,label_lines,param)





        
