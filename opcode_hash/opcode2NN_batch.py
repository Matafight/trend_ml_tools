#coding:utf-8

from opcode2NN_single import arg_parser,conf_parser,get_NN_format
from tools import dirlist2

def batch2NN(param):
    all_files = dirlist2(param['source_dirs'])
    feature_list = []
    AI_list = []
    dim  = 0
    for item in all_files:
        with open(all_files[item]) as fh:
            cont = fh.read()
            feature,NN_AI,dim = get_NN_format(param,cont,all_files[item])
            feature_list.append(feature)
            AI_list.append(NN_AI)
    #write to files
    with open(param['dest_features_path']) as fh:
        fh.write(str(dim)+'\n')
        for item in feature_list[0:-1]:
            fh.write(item+'\n')
        fh.write(feature_list[-1])
    with open(param['dest_labels_path']) as fh:
        for item in AI_list[0:-1]:
            fh.write(item+'\n')
        fh.write(AI_list[-1])

    

            



if __name__=='__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    batch2NN(param)