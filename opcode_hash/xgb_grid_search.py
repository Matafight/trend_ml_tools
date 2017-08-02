#coding=utf-8
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import precision_score,recall_score
import pprint
import pandas as pd
import time
import numpy as np
import ConfigParser
import argparse
import os
from sklearn.datasets import load_svmlight_file
from tools import get_csr_labels,save2xgdata
import log_class
from sklearn.metrics import recall_score,precision_score,accuracy_score
import copy
from sklearn.metrics import confusion_matrix


# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    booster = cf.get('xg_grid_search', 'booster')
    silent = int(cf.get('xg_grid_search','silent'))
    nthread = int(cf.get('xg_grid_search', 'nthread'))
    eta = float(cf.get('xg_grid_search', 'eta'))
    gamma = float(cf.get('xg_grid_search', 'gamma'))
    max_delta_step = float(cf.get('xg_grid_search','max_delta_step'))
    p_lambda = float(cf.get('xg_grid_search', 'lambda'))
    alpha = float(cf.get('xg_grid_search', 'alpha'))
    sketch_eps = float(cf.get('xg_grid_search', 'sketch_eps'))
    refresh_leaf = int(cf.get('xg_grid_search', 'refresh_leaf'))
    max_depth = int(cf.get('xg_grid_search', 'max_depth'))
    subsample = float(cf.get('xg_grid_search', 'subsample'))
    min_child_weight = float(cf.get('xg_grid_search', 'min_child_weight'))
    colsample_bytree = float(cf.get('xg_grid_search', 'colsample_bytree'))
    objective = cf.get('xg_grid_search', 'objective')
    base_score = float(cf.get('xg_grid_search', 'base_score'))
    eval_metric = cf.get('xg_grid_search', 'eval_metric')
    ascend = int(cf.get('xg_grid_search','ascend'))
    seed = int(cf.get('xg_grid_search', 'seed'))



    save_period = int(cf.get('xg_grid_search', 'save_period'))
    eval = int(cf.get('xg_grid_search', 'eval'))
    cv = int(cf.get('xg_grid_search','cv'))

    t_num_round = int(cf.get('xg_grid_search_tune','num_round'))
    t_max_depth = [int(i) for i in cf.get('xg_grid_search_tune','max_depth').split(',')]
    t_subsample = [float(i) for i in cf.get('xg_grid_search_tune','subsample').split(',')]
    t_min_child_weight = [float(i) for i in cf.get('xg_grid_search_tune','min_child_weight').split(',')]
    t_colsample_bytree = [float(i) for i in cf.get('xg_grid_search_tune','colsample_bytree').split(',')]
    t_colsample_bylevel = [float(i) for i in cf.get('xg_grid_search_tune','colsample_bylevel').split(',')]
    t_max_delta_step = [int(i) for i in cf.get('xg_grid_search_tune','max_delta_step').split(',')]
    t_gamma = [float(i) for i in cf.get('xg_grid_search_tune','gamma').split(',')]
    t_param = {'num_round':t_num_round,
                'max_depth':t_max_depth,
                'subsample':t_subsample,
               'min_child_weight':t_min_child_weight,
               'colsample_bytree':t_colsample_bytree,
                'colsample_bylevel':t_colsample_bylevel,
                'max_delta_step':t_max_delta_step,
                'gamma':t_gamma}

    params = {'booster': booster, 'objective': objective, 'silent': silent, 'eta': eta, 'gamma': gamma,
             'max_delta_step':max_delta_step,'lambda':p_lambda,'alpha':alpha,'sketch_eps':sketch_eps,
             'refresh_leaf':refresh_leaf,'base_score':base_score,'max_depth':max_depth,'subsample':subsample,
              'min_child_weight':min_child_weight,'colsample_bytree':colsample_bytree,
              # 'eval_metric':eval_metric,
             'seed':seed,'nthread': nthread}

    others = {'num_round':t_num_round,'cv':cv,'ascend':ascend,'eval_metric':eval_metric}

    data = cf.get('xg_grid_search', 'data')
    dataname = cf.get('xg_grid_search','dataname')
    if int(cf.get('xg_grid_search','xgmat'))==0: # if it is not a xgmat file, than convert it
        label = cf.get('xg_grid_search', 'label')
        save2xgdata(data, label)
        data += '.libsvm'
    else:
        print('training libsvm already exists...')
        data = cf.get('xg_grid_search', 'xgdata')

    pred_test = int(cf.get('xg_grid_search','pred_test'))    
    if pred_test==1:
        test_data = cf.get('xg_grid_search','test_data')
        if int(cf.get('xg_grid_search','xgmat'))==0: # if it is not a xgmat file, than convert it
            label = cf.get('xg_grid_search', 'test_label')
            save2xgdata(test_data, label)
            test_data += '.libsvm'
        else:
            print('testing libsvm already exists...')
            test_data = cf.get('xg_grid_search', 'xgdata_test')
        log_dir = os.path.split(test_data)[1]
    others['log_dir'] = dataname
    return data, test_data, params,t_param,others

def get_negative_positive_ratio(y):
    labels_np = np.array(y)
    neg_num = np.sum(labels_np==0)
    pos_num = np.sum(labels_np==1)
    return neg_num/float(pos_num)

def tune_num_boost_round(params,dtrain,num_boost_round,log,watchlist,eval_metric,feval=None,ascend=True):
    evals_result = {}
    if(feval==None):
        params['eval_metric'] = eval_metric
    xgb.train(params=params,dtrain=dtrain,num_boost_round=num_boost_round,evals=watchlist,feval=feval,evals_result=evals_result)
    evals_result = evals_result['eval'][eval_metric]
    if(ascend==True):
        loc = max(enumerate(evals_result), key=lambda x: x[1])[0]
    else:
        loc = min(enumerate(evals_result), key=lambda x: x[1])[0]
    loc += 1
    log.add("****num_boost_round : "+str(loc)+":"+str(evals_result[loc]))
    print('****  num_boost_round : %s : %s'%(loc,evals_result[loc-1]))
    return loc


def custom_eval_metirc_precison(preds,dtrain):
    labels = dtrain.get_label()
    flag1 = np.prod(preds<=1.0)
    flag2 = np.prod(preds>=0.0)
    flag = flag1*flag2
    assert flag == 1,"预测出来的值不是概率"
    preds = preds>=0.5
    preds = preds.astype(int)
    precison = precision_score(labels,preds)
    return 'precision',precison

def custom_eval_metirc_recall(preds,dtrain):
    labels = dtrain.get_label()
    flag1 = np.prod(preds<=1.0)
    flag2 = np.prod(preds>=0.0)
    flag = flag1*flag2
    assert flag == 1,"预测出来的值不是概率"
    preds = preds>=0.5
    preds = preds.astype(int)
    recall = recall_score(labels,preds)
    return 'recall',recall

def set_custom_eval_metirc(eval_metirc):

    custom_fs = dict(precision=custom_eval_metirc_precison,
                     recall=custom_eval_metirc_recall)
    for k,v in custom_fs.items():
        if(eval_metirc==k):
            return v
    return None

def predict_test(model,test_X,test_y,log):
    dtest = xgb.DMatrix(test_X,label = test_y)
    pred = model.predict(dtest)
    #split by 0.5
    pos_ind = pred>=0.5
    neg_ind = pred<0.5
    pred[pos_ind] = 1
    pred[neg_ind] = 0
    recall = recall_score(test_y,pred)
    precision = precision_score(test_y,pred)
    accuracy = accuracy_score(test_y,pred)
    log.add('test recall:'+str(recall))
    log.add('test precision:'+str(precision))
    log.add('test accuracy'+str(accuracy))
    print('test recall:'+str(recall))
    print('test precision:'+str(precision))
    print('test accuracy:'+str(accuracy))
    c= confusion_matrix(test_y,pred)
    if c.shape[0] >=2:
        cm = copy.deepcopy(c)
        cm[0,0] = int(c[1,1])
        cm[0,1] = int(c[1,0])
        cm[1,0] = int(c[0,1])
        cm[1,1] = int(c[0,0])
        print(cm)
        log.add('confusion matrix:')
        log.add(str(cm[0,0])+' '+str(cm[0,1]))
        log.add(str(cm[1,0])+' '+str(cm[1,1]))

def load_model_test():
    config = dict()
    config['model_path'] = './models/grid_search/2017_07_27_18_11_28.xgmodel'
    #config['data_path'] = '../datas/combined-0707/201707/map_md5_48/test_NN_format/NN_features.txt.libsvm'
    config['data_path']='../datas/Partition/train-test-set-0727/test_vec.dat.libsvm'
    X,y = load_svmlight_file(config['data_path'])
    X = X.todense()
    print('test_data shape:')
    print(X.shape)
    dtest = xgb.DMatrix(X,label = y)
    model = xgb.Booster(model_file=config['model_path'])
    params_xgb = model.attr('gamma')
    print(params_xgb)
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
   
def xg_train_wrapper(parser):
    xgdata,data_test,params,params_t,params_other = conf_parser(parser.conf)
    x,y = load_svmlight_file(xgdata)
    x = x.todense()
    test_x,test_y = load_svmlight_file(data_test)
    test_x = test_x.todense()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    #skf = StratifiedKFold(n_splits = 3,shuffle=True,random_state = 100)
    #train_ind,val_ind = skf.split(x,y)
    #x_train  = x[train_ind]
    #y_train = y[train_ind]
    #x_val = x[val_ind]
    #y_val = y[val_ind]

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, y_val)
    dtrain_whole = xgb.DMatrix(x,label = y)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    watchlist_whole = [(dtrain_whole, 'eval')]
    scale_pos_weight = get_negative_positive_ratio(y)

    params['scale_pos_weight'] = scale_pos_weight
    custom_feval = set_custom_eval_metirc(params_other['eval_metric'])
    log = log_class.log_class('grid_search_xgb',params_other['log_dir'])
    print(params)
    num_round = tune_num_boost_round(params,dtrain,params_other['num_round'],log,watchlist,eval_metric=params_other['eval_metric'],feval=custom_feval,ascend=params_other['ascend'])

    params_t = [dict(max_depth=params_t['max_depth']),
                dict(subsample=params_t['subsample']),
                dict(min_child_weight=params_t['min_child_weight']),
                dict(colsample_bytree=params_t['colsample_bytree']),
                dict(colsample_bylevel = params_t['colsample_bylevel']),
                dict(max_delta_step = params_t['max_delta_step']),
                dict(gamma = params_t['gamma'])]
    for param_t in params_t:
        k = param_t.keys()[0]
        values = param_t[k]
        if(k=='num_round'):
            continue
        log.add("====="+str(k)+"======="+str(values))
        print('========== ',k,' ========== ',values)
        result = []
        if(len(values) == 1):
            params[k] = values[0]
            continue
        for v in values:
            print('**** for : %s ****\n'%(str(v)))
            log.add("**** for :"+str(v)+"****")
            params[k] = v
            if (custom_feval == None):
                params['eval_metric'] = params_other['eval_metric']
            result_df = xgb.cv(params=params,
                               dtrain=dtrain_whole,
                               num_boost_round=num_round,
                               nfold=params_other['cv'],
                               # metrics=params_other['eval_metric'],
                               feval=custom_feval,
                               stratified=True,
                               verbose_eval=False,
                               show_stdv=False,
                               shuffle=True)
            result_df = result_df[['test-'+params_other['eval_metric']+'-mean']]
            assert result_df.columns[0]=='test-'+params_other['eval_metric']+'-mean','choose the correct column\n'
            result_np = result_df.as_matrix()
            result.append(float(result_np[-1][0]))
        print(zip(values,result))
        if(params_other['ascend'] == 1):
            loc = max(enumerate(result),key=lambda x:x[1])[0]
        else:
            loc = min(enumerate(result),key=lambda x:x[1])[0]
        params[k] = values[loc]
        print('%s : %s\n'%(k,params[k]))
        log.add(k)
        log.add(str(params[k]))
    num_round = tune_num_boost_round(params,dtrain_whole,params_other['num_round'],log,watchlist_whole,eval_metric=params_other['eval_metric'],feval=custom_feval,ascend=params_other['ascend'])
    model = xgb.train(params,dtrain_whole,num_round,watchlist_whole,feval=custom_feval)
    pprint.pprint(params)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    if not os.path.isdir('./models/grid_search'):
        os.mkdir('./models/grid_search')
    model.save_model('./models/grid_search' + '/' + time_str + '.xgmodel')
    print('saved : %s' % ('./models/grid_search' + '/' + time_str + '.xgmodel'))
    predict_test(model,test_x,test_y,log)

if __name__ == '__main__':
    arg = arg_parser()
    xg_train_wrapper(arg)
    #load_model_test()