#coding:utf-8

from opcode2NN_single import arg_parser,conf_parser,get_NN_format
from tools import dirlist2
import argparse
import ConfigParser
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
from functools import partial
import datetime
import time
import shutil

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    algorithm = cf.get('setup_opcode2hash','algorithm')
    bits = int(cf.get('setup_opcode2hash','bits'))
    source_dirs = cf.get('source_opcode2hash','source_dirs')
    dest_features_path = cf.get('dest_opcode2hash','dest_features_path')
    dest_labels_path = cf.get('dest_opcode2hash','dest_labels_path')
    param = {'algorithm':algorithm,'bits':bits,'source_dirs':source_dirs,'dest_features_path':dest_features_path,'dest_labels_path':dest_labels_path}
    return param


def search_files(item,all_files,param):
    with open(all_files[item]) as fh:
        all_files[item] = os.path.normcase(os.path.abspath(all_files[item]))
        with open(all_files[item]) as fh:
            cont = fh.read()
            feature,NN_AI,dim = get_NN_format(param,cont,repr(all_files[item])[1:-1])
    return feature,NN_AI,dim


def start_process():
    #print 'Starting',multiprocessing.current_process().name
    pass

def multi_2NN(param,all_files):
    starttime = datetime.datetime.now()
    neg_list = []
    pos_list = []
    missing_list = []
    partial_search = partial(search_files,all_files = all_files,param = param)
    pool = Pool(processes=3,initializer=start_process)
    ret = pool.map(partial_search,all_files.keys())
    pool.close()
    pool.join() 
    feature_list = []
    AI_list= []
    for item in ret:
        feature_list.append(item[0])
        AI_list.append(item[1])
        dim = item[2]
    #write to files
    print('writing to files...')
    with open(param['dest_features_path'],'w') as fh:
        fh.write(str(dim)+'\n')
        for item in feature_list[0:-1]:
            fh.write(item+'\n')
        fh.write(feature_list[-1])
    with open(param['dest_labels_path'],'w') as fh:
        for item in AI_list[0:-1]:
            fh.write(item+'\n')
        fh.write(AI_list[-1])
    endtime = datetime.datetime.now()
    print 'lasting time: '+str((endtime - starttime).seconds)+"  seconds"



def batch2NN(param):
    all_files = {}
    all_files = dirlist2(param['source_dirs'],all_files)
    print('total number of files %f'%len(all_files))
    feature_list = []
    AI_list = []
    dim  = 0
    print('hashing opcodes...')
    multi_2NN(param,all_files)
    '''for item in tqdm(all_files):
        all_files[item] = os.path.normcase(os.path.abspath(all_files[item]))
        with open(all_files[item]) as fh:
            cont = fh.read()
            feature,NN_AI,dim = get_NN_format(param,cont,repr(all_files[item])[1:-1])
            feature_list.append(feature)
            AI_list.append(NN_AI)
    #write to files
    print('writing to files...')
    with open(param['dest_features_path'],'w') as fh:
        fh.write(str(dim)+'\n')
        for item in feature_list[0:-1]:
            fh.write(item+'\n')
        fh.write(feature_list[-1])
    with open(param['dest_labels_path'],'w') as fh:
        for item in AI_list[0:-1]:
            fh.write(item+'\n')
        fh.write(AI_list[-1])'''

    
            



if __name__=='__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    batch2NN(param)