#coding=utf-8
"""
文件分拣，将文件分为 train  test 两个结合
train下面有子目录 0 和 1，是二分类任务的两个label
"""
import pandas as pd
import shutil
import numpy as np
import os
import glob
from pprint import pprint
from multiprocessing import Pool
import multiprocessing
from functools import partial

#from tools import dirlist
import os

phase = None
result_path3 = None
result_path1 = None
result_path0 = None
result_path2 = None
def start_process():
    print 'Starting',multiprocessing.current_process().name

def search_mv(i,refer,files):
    print("%d / %d\n" % (i, len(refer)))
    item = refer[i]
    file_name = item[0] + '.opcode'
    flag = bool(item[1])
    cases = files[files[:, 1] == file_name]
    if (len(cases) != 0):
        case_file = cases[0]
        if (phase == 'train'):
            re_path = result_path0 if flag == 0  else result_path1
        else:
            re_path = result_path2 if flag == 0  else result_path3
        shutil.copy(case_file[0], re_path)

def search_mv_single_thread(refer,files):
    for i in range(len(refer)):
        print("%d / %d\n" % (i, len(refer)))
        item = refer[i]
        file_name = item[0] + '.opcode'
        flag = bool(item[1])
        cases = files[files[:, 1] == file_name]
        if (len(cases) != 0):
            case_file = cases[0]
            re_path = ''
            if (phase == 'train'):
                re_path = result_path0 if flag == 0  else result_path1
            else:
                re_path = result_path2 if flag == 0  else result_path3
            shutil.copy(case_file[0], re_path)
    print('refer  :  %d\n' % (len(refer)))
    print('origin :  %d\n' % (len(files)))

def get_config():
    config = dict()
    config['phase'] = 'test'
    config['handle_path'] = '../datas/opcode-201707/201706/test'
    config['result_path0'] = './datas/train/0'
    config['result_path1'] = './datas/train/1'
    config['result_path2'] = './datas/test/0'
    config['result_path3'] = './datas/test/1'
    config['train_csv'] = '../datas/opcode-201707/train-test-set/train.csv'
    config['test_csv'] = '../datas/opcode-201707/train-test-set/test.csv'
    config['processes'] = None
    return config

if __name__ == '__main__':

    config = get_config()

    global phase
    global result_path0
    global result_path1
    global result_path2
    global result_path3
    phase = config['phase']

    handle_path = config['handle_path']

    result_path0 = config['result_path0']
    result_path1 = config['result_path1']
    result_path2 = config['result_path2']
    result_path3 = config['result_path3']

    result_paths = [result_path0,result_path1,result_path2,result_path3]

    for r in result_paths:
        if(os.path.exists(r) == False):
            os.makedirs(r)
            pass

    train_csv = config['train_csv']
    test_csv = config['test_csv']

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_np = train_df.as_matrix()
    test_np = test_df.as_matrix()

    #files = dirlist(handle_path,[])
    files = os.listdir(handle_path)
    files = np.array(files)
    files = files[:,np.newaxis]
    files = np.concatenate((files,files),axis=1)
    for i in range(len(files)):
        files[i,1] = files[i,1].split('/')[-1] # 得到文件名
    if(phase == 'train'):
        refer = train_np
    else:
        refer = test_np

    partial_search_mv = partial(search_mv,refer=refer,files=files)
    pool = Pool(processes=config['processes'],initializer=start_process)
    pool.map(partial_search_mv,range(len(refer)))
    pool.close()
    pool.join()

    """
    for i in range(len(refer)):
        print("%d / %d\n"%(i,len(refer)))
        item = refer[i]
        file_name = item[0]+'.opcode'
        flag = bool(item[1])
        cases = files[files[:,1] == file_name]
        if(len(cases)!=0):
            case_file = cases[0]
            re_path = ''
            if(phase=='train'):
                re_path = result_path0 if flag == 0  else result_path1
            else:
                re_path = result_path2 if flag == 0  else result_path3
            shutil.copy(case_file[0],re_path)
    print('refer  :  %d\n'%(len(refer)))
    print('origin :  %d\n'%(len(files)))
    """
