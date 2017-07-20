# _*_coding:utf-8_*_

import pandas as pd
from plot import mean_plot,variance_plot,chi2_plot,mutual_info_plot
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
    ranks_dir = cf.get('train_test_path','ranks_dir')
    return train_path,test_path,ranks_dir
    

def draw(data,ranks_dir):
    #normalization first
    predictors = data.dtypes.index[data.dtypes.index!='label']
    tar = 'label'
    # not work if the max_value = min_value
    data[predictors] = (data[predictors]-data[predictors].min())/(data[predictors].max()-data[predictors].min())
    data[predictors] = data[predictors].fillna(data[predictors].mean())
    data[predictors]=data[predictors].fillna(0)
    num_feat_each_plot = 50
    mean_plot(data,predictors,num_feat_each_plot,ranks_dir)
    #variance_plot(data,predictors,num_feat_each_plot,ranks_dir)
    #chi2_plot(data,predictors,num_feat_each_plot,ranks_dir)
    #mutual_info_plot(data,predictors,num_feat_each_plot,ranks_dir)

if __name__ == '__main__':
    parser = arg_parser()
    train_path,test_path,ranks_dir = get_params(parser.conf)
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    data = pd.concat([train_data,test_data],axis=0)
    data.drop('id',axis=1,inplace = True)
    draw(data,ranks_dir)
   

