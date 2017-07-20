
#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from tqdm import  tqdm
import argparse
import ConfigParser
from tools import dirlist,to_NN

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    input_dirs = cf.get('source', 'source_dirs')
    class_type =cf.get('source','class_type')
    output_dir = cf.get('dest', 'dest_dirs')
    input_dir_list = input_dirs.split(';')
    param = {'input_dir_list':input_dir_list,'output_dir':output_dir,'class_type':class_type}
    return param

def combine_files(dir_list):
    X = []
    paths=[]
    for item in dir_list:
        df = pd.read_csv(item)
        new_col = df.columns[df.columns!='opcode_name']
        X.append(df[new_col].as_matrix())
        paths.extend(df['opcode_name'].values.tolist())
    #concat
    data = np.concatenate(X,axis=0)
    return data,paths

def convert2NN(dir_list,dest_dir):
    data,paths = combine_files(dir_list)
    num_col = data.shape[1]
    features = data[:,:num_col-1]
    labels = data[:,num_col-1]
    paths = np.array(paths)
    to_NN(data=features,label=labels,path=paths,NN_name='NN_train.txt',NN_label_name='NNAI.txt')

if __name__=='__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    convert2NN(param['input_dir_list'],param['output_dir'])