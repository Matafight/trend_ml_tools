#coding:utf-8
import sys
import xgboost as xgb
import numpy as np
import pandas as pd
import argparse
import ConfigParser
from method_config import param_xgb_space
from tools import save2xgdata
from sklearn.datasets import load_svmlight_file
from hyperopt import fmin,hp,tpe,STATUS_OK,Trials,space_eval
import log_class
import time 
import pickle 
from sklearn.metrics import recall_score,precision_score,accuracy_score
import copy
from sklearn.metrics import confusion_matrix
import os
#precision_score(y_true,y_pred)
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-pipe',action = 'store_true',dest = 'pipeline',default = False)
    return parser.parse_args()




# configure parser
def conf_parser(conf_path,pipeline):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    top_dir = cf.get('top_dir','top_dir')
    if False==pipeline:
        print('not in pipiline')
        data = cf.get('xgb_hyopt_xg_conf', 'data')
    else:
        algorithm = cf.get('setup_opcode2hash','algorithm') +'_'+cf.get('setup_opcode2hash','bits')
        alg_bits_path = os.path.join(top_dir,algorithm)
        data = os.path.join(alg_bits_path,r'all_train_opcode_NN/NN_features.txt')
        others['log_dir']= algorithm


    
    n_fold = int(cf.get('xgb_hyopt_xg_conf','n_fold'))
    save_model = int(cf.get('xgb_hyopt_xg_conf','save_model'))
    metrics = cf.get('xgb_hyopt_xg_conf','metrics')
    if_ascend = int(cf.get('xgb_hyopt_xg_conf','if_ascend'))
    if int(cf.get('xgb_hyopt_xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
        if False == pipeline:
            label = cf.get('xgb_hyopt_xg_conf', 'label')
            save2xgdata(data, label)
            data += '.libsvm'
        else:
            label = os.path.join(alg_bits_path,r'all_train_opcode_NN/NN_AI.txt')
            save2xgdata(data,label)
            data += '.libsvm'
    else:
        if False == pipeline:
            data = cf.get('xgb_hyopt_xg_conf', 'xgdata')
        else:
            data = os.path.join(alg_bits_path,r'all_train_opcode_NN/NN_features.txt.libsvm')

        
    pred_test = int(cf.get('xgb_hyopt_xg_conf','pred_test'))
    param_untuned = {'n_fold':n_fold,'data':data,'save_model':save_model,'metrics':metrics,'if_ascend':if_ascend,'pred_test':pred_test}
    if False == pipeline:
        param_untuned['log_dir'] = 'ordinary_data'
    else:
        param_untuned['log_dir'] = algorithm
    #for test data
    if pred_test:
        if False == pipeline:
            test_data = cf.get('xgb_hyopt_test','data')
        else:
            test_data = os.path.join(alg_bits_path,r'all_test_opcode_NN/NN_features.txt')
        if int(cf.get('xgb_hyopt_test','xgmat'))==0: # if it is not a xgmat file, than convert it
            if False == pipeline:
                label = cf.get('xgb_hyopt_test', 'label')
                save2xgdata(test_data, label)
                test_data += '.libsvm'
            else:
                label = os.path.join(alg_bits_path,r'all_test_opcode_NN/NN_AI.txt')
                save2xgdata(test_data,label)
                test_data += '.libsvm'
        else:
            if False == pipeline:
                test_data = cf.get('xgb_hyopt_test', 'xgdata')
            else:
                test_data = os.path.join(alg_bits_path,r'all_test_opcode_NN/NN_features.txt.libsvm')
        param_untuned['test_data'] = test_data
    return param_untuned

    
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

def set_custom_eval_metric(eval_metirc):
    custom_fs = dict(precision=custom_eval_metirc_precison,
                     recall=custom_eval_metirc_recall)
    for k,v in custom_fs.items():
        if(eval_metirc==k):
            return v
    return None

def get_scale_pos_weight(y):
    neg_sum = np.sum(y==0)
    pos_sum = np.sum(y==1)
    return neg_sum/pos_sum

class xgb_hyopt:
    def __init__(self,train_X,train_y,test_X,test_y,param_untuned):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.log = log_class.log_class('xgb',param_untuned['log_dir'])
        self.param_untuned = param_untuned
        self.eval_step = 0

    def construct_model(self,param):
        train_X = self.train_X
        train_y = self.train_y
        dtrain = xgb.DMatrix(train_X,label = train_y)
        #stratified = True maximize = True
        if self.param_untuned['if_ascend'] == 1:
            maximize = True
        else:
            maximize = False
        custom_eval_metric = set_custom_eval_metric(self.param_untuned['metrics'])
        if custom_eval_metric == None:
            df_result = xgb.cv(param,dtrain,
                                num_boost_round = param['num_round'],
                                stratified = False,
                                metrics = self.param_untuned['metrics'],
                                nfold = self.param_untuned['n_fold'])
        else:
            df_result = xgb.cv(param,dtrain,
                                num_boost_round = param['num_round'],
                                stratified = False,
                                maximize = maximize,
                                feval =custom_eval_metric,
                                nfold = self.param_untuned['n_fold'])
        #stratified = True 效果更差？
        #df_result = xgb.cv(param,dtrain,num_boost_round = param['num_round'],stratified = True,maximize = maximize,metrics = self.param_untuned['metrics'],feval = custom_eval_metric,nfold = self.param_untuned['n_fold'])
        column = 'test-'+self.param_untuned['metrics']+'-mean'
        num_row = df_result.shape[0]
        loss = df_result.iloc[[num_row-1]][column].values[0]
        return loss

    def hyperopt_obj(self,param):
        int_params = ['max_depth','num_round']
        for item in int_params:
            param[item] = int(param[item])
        self.eval_step +=1
        self.log.add('eval_step:'+str(self.eval_step))
        self.log.add(param,1)
        loss = self.construct_model(param)
        self.log.add('loss:'+str(loss))
        if self.param_untuned['if_ascend'] == 1:
            loss = -1*loss
        print('eval_step:'+str(self.eval_step))
        print("current parameter score : "+str(loss))
        return {'loss':loss,'status':STATUS_OK}

    def main_tunning(self):
        obj = lambda p:self.hyperopt_obj(p)
        cur_param_space =  param_xgb_space
        train_X = self.train_X
        train_y = self.train_y
        dtrain = xgb.DMatrix(train_X,label = train_y)
        #get scale_pos_weight
        param_xgb_space['scale_pos_weight'] = get_scale_pos_weight(self.train_y)
        self.log.add("scale_pos_weight:"+str(param_xgb_space['scale_pos_weight']))
        self.log.add('eval_metric:'+str(self.param_untuned['metrics']))

        int_params = ['max_depth','num_round']
        trials = Trials()
        best_params = fmin(obj,param_xgb_space,algo=tpe.suggest,max_evals=param_xgb_space['max_evals'],trials=trials)
        print('best params')
        self.log.add("best_params")
        self.log.add(best_params,1)

        for item in int_params:
            best_params[item] = int(best_params[item])
        for item in best_params:
            param_xgb_space[item] = best_params[item]
        #get cv score for best params
        loss =self.construct_model(param_xgb_space)
        self.log.add('loss:'+str(loss))

        ##train new model for best params
        model  = xgb.train(param_xgb_space,dtrain,num_boost_round = best_params['num_round'])
        #save model
        if self.param_untuned['save_model']:
            cur_time = self.log.get_time()
            save2model = './models/xgb_'+cur_time
            pickle.dump(model,open(save2model,'wb'))        

        if self.param_untuned['pred_test'] == 1:
            self.predict_test(model)

    def predict_test(self,model):
        test_X = self.test_X
        test_y = self.test_y
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
        self.log.add('test recall:'+str(recall))
        self.log.add('test precision:'+str(precision))
        self.log.add('test accuracy'+str(accuracy))
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
        self.log.add('confusion matrix:')
        self.log.add(str(cm[0,0])+' '+str(cm[0,1]))
        self.log.add(str(cm[1,0])+' '+str(cm[1,1]))


    
    


if __name__=='__main__':
    #load data and convert data
    parser = arg_parser()
    param_untuned = conf_parser(parser.conf,parser.pipeline)
    print("reading data from")
    print(param_untuned['data'])
    X,y = load_svmlight_file(param_untuned['data'])
    X = X.todense()

    if param_untuned['pred_test'] == 1:
    #load test data
        test_X,test_y = load_svmlight_file(param_untuned['test_data'])
        test_X = test_X.todense()
    else:
        test_X = X
        test_y = y
   
    xgb_model = xgb_hyopt(X,y,test_X,test_y,param_untuned)

    #training new model
    xgb_model.main_tunning()

    #load existing model and predict 
    #model = pickle.load(open('./xgb/Models/2017_07_25_14_06_14.xgmodel','rb'))
    #model = xgb.Booster(model_file='./xgb/Models/2017_07_25_14_06_14.xgmodel')
    #xgb_model.predict_test(model)



    #dtrain = xgb.DMatrix(X,label = y)
    #param = {
    #'booster': 'gbtree',
    #'objective': 'binary:logistic',
    #'num_round': 10,
    #'nthread': 5,
    #'silent':1,
    #}
    #df_result = xgb.cv(param,dtrain,num_boost_round=param['num_round'],nfold=3)
    #print(df_result)
    #print(type(df_result))
    #num_row = df_result.shape[0]
    #column = 'test-error-mean'
    #loss = df_result.iloc[[num_row-1]][column].values[0]
    #print(loss)

    

        
