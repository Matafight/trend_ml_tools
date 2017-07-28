#coding:utf-8
'''
   input: opcode and its path
   output: NNAI format data
   '''
from tools import return_NN_format
import argparse
import ConfigParser
from opcode2hash import batch_str2bits

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


def get_NN_format(param,opcode,path,label=0):
    batch_map = batch_str2bits(param)
    bit_cont = batch_map.str2bit(opcode)
    dim = len(bit_cont)
    features,line_AI = return_NN_format(bit_cont,label,path)
    return features,line_AI,dim

if __name__ == '__main__':
    parser = arg_parser()
    param = conf_parser(parser.conf)
    path = '8e69d0acef0c987c7ce63d5838782c6fe56f8b1edb357f31d932b444a3036e1a.opcode'
    with open(path) as fh:
        cont = fh.read()
        feature,line_ai = get_NN_format(param,cont,path)
        print(feature)
        print(line_ai)