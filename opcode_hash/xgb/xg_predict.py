# coding=utf-8
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import recall_score,precision_score,accuracy_score
import copy
from sklearn.metrics import confusion_matrix
from tools import read_sample_path
def get_config():
    config = dict()
    config['model_path'] = './Models/2017_07_25_14_06_14.xgmodel'
    config['data_path'] = '../test_NN_format/NN_features.txt.libsvm'
    return config

if __name__ == '__main__':
    config = get_config()
    X,y = load_svmlight_file(config['data_path'])
    X = X.todense()
    print('test_data shape:')
    print(X.shape)
    dtest = xgb.DMatrix(X,label = y)
    model = xgb.Booster(model_file='./Models/2017_07_25_14_06_14.xgmodel')
    pred = model.predict(dtest)
    test_y = y
    pred = model.predict(dtest)
    #split by 0.5
    pos_ind = pred>=0.5
    neg_ind = pred<0.5
    pred[pos_ind] = 1
    pred[neg_ind] = 0
    recall = recall_score(test_y,pred)
    precision = precision_score(test_y,pred)
    accuracy = accuracy_score(test_y,pred)
    print('test recall:'+str(recall))
    print('test precision:'+str(precision))
    print('test accuracy:'+str(accuracy))
    c= confusion_matrix(test_y,pred)
    cm = copy.deepcopy(c)
    cm[0,0] = int(c[1,1])
    cm[0,1] = int(c[1,0])
    cm[1,0] = int(c[0,1])
    cm[1,1] = int(c[0,0])
    print(cm)
   



