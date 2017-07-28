# _*_ coding:utf-8 _*_

#要利用反射机制实现工厂模式，算是一个设计模式实战吧
import argparse
import ConfigParser
import pandas as pd
import numpy as np
import hashlib
import sha3
import abc
import os
import re
from tqdm import tqdm
import time
from multiprocessing import Pool
import multiprocessing
from functools import partial
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    top_dir = cf.get('top_dir','top_dir')
    input_dirs = cf.get('opcode2hash_source', 'source_dirs')
    input_dirs = [os.path.join(top_dir,i) for i in input_dirs.split(';')]

    labels = cf.get('opcode2hash_source','labels')
    labels = [int(i) for i in labels.split(';')]

    output_dir = cf.get('opcode2hash_dest', 'dest_dirs')
    output_dir = os.path.join(top_dir,output_dir)

    algorithm = cf.get('opcode2hash_algorithm','algorithm')

    bits = int(cf.get('opcode2hash_algorithm','bits'))

    tasks = cf.get('opcode2hash_source','tasks')
    tasks = tasks.split(';')

    class_types = cf.get('opcode2hash_source','class_types')
    class_types = class_types.split(';')
    
    param = {'input_dirs':input_dirs,'labels':labels,'output_dir':output_dir,'algorithm':algorithm,'bits':bits,'tasks':tasks,'class_types':class_types}
    return param

# sample code 
#algorithm = 'map_shake'
#test_md5 = globals()[algorithm](8)
#print(test_md5.get_hash("hello"))
    
class map_factory():
        __metaclass__ = abc.ABCMeta
        hash_cont = None
        def __init__(self,bits):
            self.bits = bits

        @abc.abstractmethod
        def get_hash(self,str):
            print('factory get hash')


class map_shake(map_factory):
    def __init__(self,bits):
        self.bits = bits
    def get_hash(self,str):
        if self.bits%8!=0:
            raise NameError("bits can not  be divided by 8" )
        bits = self.bits/8
        s = hashlib.shake_128()
        s.update(str)
        hash_cont = s.hexdigest(bits)
        return hash_cont

class map_md5(map_factory):
    def __init__(self,bits):
        self.bits = bits

    def get_hash(self,str):
        s = hashlib.md5()
        s.update(str)
        hash_cont = s.hexdigest()
        return hash_cont



def start_process():
    #print 'Starting',multiprocessing.current_process().name
    pass


def top_level_multi_mapping(i,str2bits_object,filelist):
    return str2bits_object.multi_mapping(i,filelist)

class batch_str2bits():
    algorithm = 'MD5'
    def __init__(self,param = None):
        if None != param:
            self.algorithm = param['algorithm']
            self.input_dirs = param['input_dirs']
            self.output_dir = param['output_dir']
            self.bits = param['bits']
            self.labels = param['labels']
            self.tasks = param['tasks']
            self.class_types = param['class_types']

    def hexstr2bitstr(self,string):
        hex_list = [format(int(c,16),'b') for c in string ]
        re_list = []
        for l in hex_list:
            len_l = len(l)
            for i in range(4-len_l):
                l = '0'+l
            assert len(l) == 4
            re_list.append(l)
        temp = ''.join(re_list)
        temp = [c for c in temp]
        return temp

    
    def str2hash(self,str):
        return globals()[self.algorithm](self.bits).get_hash(str)

    
    def dirlist(self,path,allfile=[],first_vis=False):
        filelist = os.listdir(path)
        if True==first_vis:
            allfile=[]
        for filename in filelist:
            filepath = os.path.join(path, filename)
            if os.path.isdir(filepath):
                dirlist(filepath, allfile)
            else:
                allfile.append(filepath)
        return allfile

    def batch_converting(self,input_dir):
        allfile = self.dirlist(input_dir,first_vis=True)
        print('hashing binary file to bits...')
        hash_list = []
        for item in tqdm(allfile):
            with open(item) as fh:
                #read in binary mode
                cont = fh.read()
                hash_list.append(self.str2hash(cont))
        #converting
        bits_list = []
        for item in hash_list:
            bits_list.append(self.hexstr2bitstr(item))
        return allfile,bits_list

    #多进程处理
    def multi_mapping(self,i,filelist):
        with open(filelist[i]) as fh:
            cont = fh.read()
            hash_cont = self.str2hash(cont)
            return hash_cont
    def multi_handle(self,filelist):

        partial_mapping = partial(top_level_multi_mapping,str2bits_object = self,filelist= filelist)
        pool = Pool(processes=4,initializer=start_process)
        ret = pool.map(partial_mapping,range(len(filelist)))
        pool.close()
        pool.join() 
        return ret

    # called by other functions
    def listfiles2csv(self,filelist,labels,dest_dir,param):
        self.algorithm = param['algorithm']
        self.bits = param['bits']
        hash_list = []
        print('Multi processing hashing...')
        hash_list = self.multi_handle(filelist)
        bits_list = []
        for item in hash_list:
            bits_list.append(self.hexstr2bitstr(item))
        bits_arr = np.array(bits_list)
        num_row,num_column = np.shape(bits_arr)
        labels = np.array(labels).reshape(num_row,1)
        bits_arr = np.concatenate([bits_arr,labels],axis=1)
        columns = ['bit_'+str(i) for i in range(1,num_column+1)]
        columns.append('label')
        df_bits = pd.DataFrame(bits_arr,columns=columns)
        df_bits.insert(0,'opcode_name',filelist)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_path = os.path.join(dest_dir,time_str+"_opcode.csv")
        df_bits.to_csv(save_path,index=False)
        return save_path


    def saving2files(self,input_dir,task,class_type,label):
        allfile,bits_list = self.batch_converting(input_dir)
        bits_arr = np.array(bits_list)
        num_row,num_column = np.shape(bits_arr)
        labels = label*np.ones(num_row).reshape(num_row,1)
        bits_arr = np.concatenate([bits_arr,labels],axis=1)
        columns = ['bit_'+str(i) for i in range(1,num_column+1)]
        columns.append('label')
        df_bits = pd.DataFrame(bits_arr,columns=columns)
        df_bits.insert(0,'opcode_name',allfile)

        new_dir = self.algorithm+'_'+str(self.bits)
        n_path = os.path.join(self.output_dir,new_dir)
        if not os.path.isdir(n_path):
            os.mkdir(n_path)
        n_path = os.path.join(n_path,task)
        if not os.path.isdir(n_path):
            os.mkdir(n_path)
        n_path = os.path.join(n_path,class_type)
        if not os.path.isdir(n_path):
            os.mkdir(n_path)
        df_bits.to_csv(os.path.join(n_path,self.algorithm+'_'+str(self.bits)+'_opcode.csv'),index=False)
    

    def batch_saving2files(self):
        for i,input_dir in enumerate(self.input_dirs):
            print(input_dir)
            self.saving2files(input_dir,self.tasks[i],self.class_types[i],self.labels[i])

        
    
if __name__=='__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)

    batch_map = batch_str2bits(param)
    #batch_map.batch_saving2files()

    #pattern = re.compile(r'([\w]+\.opcode)$')
    #str = '../datas/newdownload/combined/201707/malicious/test/1\004f1efc5ddc00fa51c9a7baa3dbd528e4834da5f26a4ee5299d98a4074c5ee9.opcode'
    #ret = pattern.search(str)
    #正则表达式获取文件夹最后一层目录
    #print(ret.group())
