
#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from tqdm import  tqdm
import argparse
import ConfigParser
from tools import dirlist,to_NN
import os
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    #判断属于pipeline还是单独运行,由命令行参数给出
    parser.add_argument('-pipe',action = 'store_true',dest = 'pipeline',default = False)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path,pipeline):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    top_dir = cf.get('top_dir','top_dir')
    if pipeline == False:
        training_dirs = cf.get('hashbit2NN_source', 'training_dirs')
        training_dirs = [os.path.join(top_dir,i) for i in training_dirs.split(';')]
        testing_dirs = cf.get('hashbit2NN_source','testing_dirs')
        testing_dirs = [os.path.join(top_dir,i) for i in testing_dirs.split(';')]
        output_dir = cf.get('hashbit2NN_dest', 'dest_dirs')
        output_dir = os.path.join(top_dir,output_dir)
    else:
        #从opcode2hash获取路径
        algorithm = cf.get('opcode2hash_algorithm','algorithm') +'_'+cf.get('opcode2hash_algorithm','bits')
        alg_bits_path = os.path.join(top_dir,algorithm)
        train_malware_path = os.path.join(alg_bits_path,r'train/malware/'+algorithm+'_opcode.csv')
        train_normal_path = os.path.join(alg_bits_path,r'train/normal/'+algorithm+'_opcode.csv')
        training_dirs=[train_malware_path,train_normal_path]
        test_malware_path = os.path.join(alg_bits_path,r'test/malware/'+algorithm+'_opcode.csv')
        test_normal_path = os.path.join(alg_bits_path,r'test/normal/'+algorithm+'_opcode.csv')
        testing_dirs= [test_malware_path,test_normal_path]
        output_dir = alg_bits_path
        print(testing_dirs)
        print(training_dirs)
        print(output_dir)
        print('finished reading in common_pipeline')
        
    param = {'training_dirs':training_dirs,'testing_dirs':testing_dirs,'output_dir':output_dir}
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

def convert2NN(dir_list,dest_dir,train_or_test):
    data,paths = combine_files(dir_list)
    num_col = data.shape[1]
    features = data[:,:num_col-1]
    labels = data[:,num_col-1]
    paths = np.array(paths)
    if train_or_test == 'train':
        n_path = os.path.join(dest_dir,'train_NN_format')
    else:
        n_path = os.path.join(dest_dir,'test_NN_format')
    if not os.path.isdir(n_path):
        os.mkdir(n_path)

    NN_name = os.path.join(n_path,'NN_features.txt')
    NN_label_name = os.path.join(n_path,'NNAI.txt')
    to_NN(data=features,label=labels,path=paths,NN_name=NN_name,NN_label_name=NN_label_name)

def batchconvert2NN(param):
    if len(param['training_dirs'])!=0:
        convert2NN(param['training_dirs'],param['output_dir'],'train')
    if len(param['testing_dirs'])!=0:
        convert2NN(param['testing_dirs'],param['output_dir'],'test')
    
        




if __name__=='__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf,parser.pipeline)
    batchconvert2NN(param)