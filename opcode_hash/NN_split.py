#coding:utf-8
#给定train test 划分,输出划分好的NN format数据

import pandas as pd
import numpy as np
import argparse
import ConfigParser
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
    dest_dir = cf.get('dest_dir','dest_dir')
    param = {'train_path':train_path,'test_path':test_path,'features_path':features_path,'labels_path':labels_path}
    return param
def read_csv(train_path,test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train,df_test

def retrive_opcode(str):
    #get opcode 
    pass

#return a dict mapping opcode to index of list, lists contain features and labels
def read_all_data(features_path,labels_path):
    features_lines = []
    label_lines = []
    opcode2index={}
    with open(features_path) as fh:
        features_lines = fh.readlines()
    with open(labels_path) as fh:
        label_lines = fh.readlines()
        for i,line in enumerate(label_lines):
            opcode = retrive_opcode(line)
            opcode2index[opcode] = i 

    return features_lines,label_lines,opcode2index

#类别未知label记为0，正例label记为1，负例label记为2
def redistribute_samples(df_train,df_test,opcode2index,features_lines,labels_lines,param):

    train_pos_lines = []
    train_neg_lines = []
    test_pos_lines = []
    test_neg_lines = []

    train_pos_labels = []
    train_neg_labels = []
    test_pos_labels = []
    test_neg_labels = []
    for ind in tqdm(df_train.shape[0]):
        #search in opcode2index
        item = df_train.iloc[ind]
        cur_opcode = item['id'] + '.opcode'
        if cur_opcode in opcode2index.keys():
            cur_label = item['malware']
            if cur_label == 1:
                train_pos_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_line[0] = 1
                train_pos_labels.append(label_line)
            else:
                train_neg_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_line[0] = 2
                train_neg_labels.append(label_line)
    for ind in tqdm(df_test.shape[0]):
        item = df_test.iloc[ind]
        cur_opcode = item['id'] + '.opcode'
        if cur_opcode in opcode2index.keys():
            cur_label = item['malware']
            if cur_label == 1:
                test_pos_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_line[0] = 1
                test_pos_labels.append(label_line)
            else:
                test_neg_lines.append(features_lines[opcode2index[cur_opcode]])
                label_line = labels_lines[opcode2index[cur_opcode]]
                label_line[0] = 2
                test_neg_labels.append(label_line)
    #save into four different files
    #train_pos,train_neg,test_pos,test_neg
    train_pos_features_path  = os.path.join(param['dest_dir'],'train_malware_vec.dat')
    train_pos_labels_path = os.path.join(param['dest_dir'],'train_malware_vec.AI')
    train_neg_features_path = os.path.join(param['dest_dir'],'train_noraml_vec.dat')
    train_neg_labels_path = os.path.join(param['dest_dir'],'train_normal_vec.AI')
    test_pos_features_path = os.path.join(param['dest_dir'],'test_malware_vec.dat')
    test_pos_labels_path = os.path.join(param['dest_dir'],'test_malware_vec.AI')
    test_neg_features_path = os.path.join(param['dest_dir'],'test_normal_vec.dat')


    with open(param['dest_dir'])
    




        
