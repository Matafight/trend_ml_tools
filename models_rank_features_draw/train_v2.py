# _*_coding:utf-8_*_
import numpy as np
import xgboost as xgb
import pickle
import argparse
import xg_plot
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file

import pandas as pd	
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from operator import itemgetter
import ConfigParser

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    #parser.add_argument('-o', '--output', required=True)
    #parser.add_argument('-v', '--visual', default=30)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    booster = cf.get('xg_conf', 'booster')
    objective = cf.get('xg_conf', 'objective')
    silent = int(cf.get('xg_conf','silent'))
    eta = float(cf.get('xg_conf', 'eta'))
    gamma = float(cf.get('xg_conf', 'gamma'))
    min_child_weight = float(cf.get('xg_conf', 'min_child_weight'))
    max_depth = int(cf.get('xg_conf', 'max_depth'))
    num_round = int(cf.get('xg_conf', 'num_round'))
    save_period = int(cf.get('xg_conf', 'save_period'))
    eval = int(cf.get('xg_conf', 'eval'))
    nthread = int(cf.get('xg_conf', 'nthread'))
    param = {'booster': booster, 'objective': objective, 'silent': silent, 'eta': eta, 'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': max_depth, 'num_round': num_round, 'save_period': save_period, 'eval': eval,
             'nthread': nthread}
    data_path = cf.get('xg_conf', 'train_path')



    #if int(cf.get('xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
    #    try:
    #        label = cf.get('xg_conf', 'label')
    #        save2xgdata(data, label)
    #        data += '.xgmat'
    #    except:
    #        print('convert fail!')
    return data_path, param

def read_label(path):
    with open(path, 'r',encoding = 'utf-8',errors = 'ignore') as label_fi:
        label_data = []
        for line in label_fi.readlines():
            if int(line.split('|')[0]) == 2:
                label = 0
            if int(line.split('|')[0]) == 1:
                label = 1
            label_data.append(label)
        label_data = np.asarray(label_data)

    print('Finished read label')
    return label_data

# xgboost needs data in a certain format:
# label 1:value1 2:value2 5:value5 ...
# load feature from fe_path and label from la_path, save the converted data into save_path
#def save2xgdata(fe_path, la_path):
#    print("reading row features!")
#    label = read_label(la_path)
#    # feature and label is in numpy.asarray format
#    saver = open(fe_path+'.xgmat', 'w')
#    print(fe_path)
#    with open(fe_path, 'r',encoding = 'utf-8',errors = 'ignore') as data_fi:
#    	lines = data_fi.readlines()
#    	dimension = int(lines[0].strip())
#        count = 0
#        for line in lines[1:]:
#            data_index = [int(i) for i in line.split(';')[:-1][3::2]]
#            data_value = [float(i) for i in line.split(';')[:-1][4::2]]
#            data_str = str(label[count])
#            count += 1
#            try:
#                for i in range(len(data_index)):
#                    data_str += (' '+str(data_index[i])+':'+str(data_value[i]))
#            except IndexError:
#                pass
#            if len(data_index)<dimension:
#                data_str += (' '+str(dimension)+':'+str(0))
#            saver.writelines(data_str+'\n')
#    print('Finished read feature')
# Give train data and parameters, return a trained model

def xgb_train(xg_train, param):
    if param['eval']:
        watchlist=[(xg_train, 'train')]
        xg_model = xgb.train(param, xg_train, param['num_round'], watchlist)
    else:
        xg_model = xgb.train(param, xg_train, param['num_round'])
    return xg_model

 
def evaluation(preds, label):
    preds = [int(preds[i]>=0.5) for i in range(len(preds))]
    accuracy = sum([int(preds[i]==label[i]) for i in range(len(preds))]) / (len(preds)+0.0)
    precision = sum([int(preds[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(preds)+0.0)
    recall = sum([int(label[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(label)+0.0)
    return preds,accuracy, precision, recall   

def read_data(path):
    df_data = pd.read_csv(path)
    df_data.drop('id',axis=1,inplace = True)
    predictors = df_data.dtypes.index[df_data.dtypes.index!='label']
    tar = 'label'
    return predictors,df_data[predictors].values,df_data[tar].values


def feat_plot(xg_model,data,label,x_axis_label=None,eval_type='weight',max_num_features=None):
    xg_plot.fea_plot(xg_model,data,label,type = eval_type,max_num_features = max_num_features,x_axis_label = x_axis_label)
    plt.show()

def confusion_mat(xg_model):
    pass

if __name__ == '__main__':

    arg = arg_parser()
    train_path, param = conf_parser(arg.conf)
    predictors,train_data,train_labels = read_data(train_path)
    dtrain = xgb.DMatrix(train_data,label=train_labels)
    xg_model = xgb_train(dtrain,param)
    predictors = predictors.values
    feat_plot(xg_model,train_data,train_labels,eval_type = 'gain',max_num_features = 50,x_axis_label=predictors) 
    
    '''pred= xg_model.predict(xg_test)
    test_labels = xg_test.get_label()
    pred,acc,pre,recall = evaluation(pred,test_labels)
    print('accuracy: %s, precision: % recall:%s'%(str(acc),str(pre),str(recall)))
    import copy
    from sklearn.metrics import confusion_matrix
    c= confusion_matrix(test_labels,pred)
    cm = copy.deepcopy(c)
    cm[0,0] = int(c[1,1])
    cm[0,1] = int(c[1,0])
    cm[1,0] = int(c[0,1])
    cm[1,1] = int(c[0,0])
    print(cm)


    pred_train= xg_model.predict(xg_train)
    train_labels = xg_train.get_label()
    pred_train,acc,pre,recall = evaluation(pred_train,train_labels)
    print('accuracy: %s, precision: % recall:%s'%(str(acc),str(pre),str(recall)))
    import copy
    from sklearn.metrics import confusion_matrix
    c= confusion_matrix(train_labels,pred_train)
    cm = copy.deepcopy(c)
    cm[0,0] = int(c[1,1])
    cm[0,1] = int(c[1,0])
    cm[1,0] = int(c[0,1])
    cm[1,1] = int(c[0,0])
    print(cm)


    eval_type = 'weight'

    predictors = ['cputype', 'certs_info_count', 'export_size', 'sizeofcmds',
       'nspecialslots', 'ncmds', 'filetype', 'onlymacho', 'pagesize',
       'cpusubtype', 'file_size', 'cert_valid', 'ncodeslots',
       'ents_info_count', 'flags', 'codelimit', 'lazy_bind_size', 'upx',
       'cd_version', 'load_commands_count', 'magic', 'weak_bind_size']
    #scaled_train_data = np.loadtxt('./data/train_scaled.csv')
    xg_plot.fea_plot(xg_model, train_data, train_label, max_num_features=None,type = eval_type,x_axis_label = predictors)


    fscore = xg_model.get_score(importance_type='gain')
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = list()
    for key in fscore:
        fea_index.append(int(key[0][1:])) # note that the index of array in Python start from 0
        print(key)
    print(fea_index)

    xg_plot.tsne_plot(xg_model,train_data, train_label,type = eval_type,max_num_features = 15)
    xg_plot.mean_plot(xg_model,train_data,train_label,max_num_features = None,type = eval_type,x_axis_label = predictors)
    #xg_plot.variance_plot(xg_model,train_data,train_label,max_num_features = None,x_axis_label = predictors)
    xg_plot.chi2_plot(xg_model,train_data,train_label,max_num_features = None,type = eval_type,x_axis_label = predictors)

    xg_plot.mean_plot(xg_model,test_data,test_label,max_num_features = None,type = eval_type,x_axis_label = predictors)
    

    #xg_plot.stat_plot(xg_model,data[0].toarray(),data[1],max_num_features = int(arg.visual))
    plt.show()
    #saver = file(arg.output, 'wb')
    #pickle.dump(xg_model, saver)'''



