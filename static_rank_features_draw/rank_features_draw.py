# _*_coding:utf-8_*_

import pandas as pd
from plot import mean_plot,variance_plot,chi2_plot
import argparse
import ConfigParser

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

def get_params(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    train_path = cf.get('train_test_path','train_path') 
    test_path = cf.get('train_test_path','test_path')
    return train_path,test_path
    

def draw(data):
    #normalization first
    predictors = data.dtypes.index[data.dtypes.index!='label']
    tar = 'label'
    # not work if the max_value = min_value
    data[predictors] = (data[predictors]-data[predictors].min())/(data[predictors].max()-data[predictors].min())
    data[predictors] = data[predictors].fillna(data[predictors].mean())
    data[predictors]=data[predictors].fillna(0)
    num_feat_each_plot = 50
    #mean_plot(data,predictors,num_feat_each_plot)
    variance_plot(data,predictors,num_feat_each_plot)
    #chi2_plot(data,predictors,num_feat_each_plot)

if __name__ == '__main__':
    parser = arg_parser()
    train_path,test_path = get_params(parser.conf)
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    data = pd.concat([train_data,test_data],axis=0)
    data.drop('id',axis=1,inplace = True)

    draw(data)
   

