#_*_coding:utf-8_*_

import xgb_hyopt
import xg_grid_search
import argparse
import ConfigParser
import os
from sklearn.datasets import load_svmlight_file

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-hyperopt',action = 'store_true',dest='hyperopt',default = False)
    parser.add_argument('-pipe',action = 'store_true',dest = 'pipeline',default = False)
    return parser.parse_args()


def hyperopt_model(parser):
    param_untuned = xgb_hyopt.conf_parser(parser.conf,parser.pipeline)
    print("reading data from")
    print(param_untuned['data'])
    X,y = load_svmlight_file(param_untuned['data'])
    X = X.todense()
    if param_untuned['pred_test'] == 1:
    #load test data
        test_X,test_y = load_svmlight_file(param_untuned['test_data'])
        test_X = test_X.todense()
        print('pred_test true')
    else:
        test_X = X
        test_y = y
   
    xgb_model = xgb_hyopt.xgb_hyopt(X,y,test_X,test_y,param_untuned)

    #training new model
    xgb_model.main_tunning()

def grid_search_model(parser):
    xg_grid_search.xg_train_wrapper(parser)



if __name__=='__main__':
    parser = arg_parser()
    if parser.hyperopt==True:
        hyperopt_model(parser)
    else:
        grid_search_model(parser)
