#coding:utf-8
#给定train test 划分,输出划分好的NN format数据

import pandas as pd
import numpy as np


def read_csv(train_path,test_path):
    pass

#return a dict mapping opcode to index of list, lists contain features and labels
def read_all_data(features_path,label_path):
    pass

#类别未知label记为0，正例label记为1，负例label记为2
def redistribute_samples(df_train,df_test,opcode2index,features_lines,labels_lines):

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




        
