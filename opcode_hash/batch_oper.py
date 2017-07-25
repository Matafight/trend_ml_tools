import os

# operate on the network location
#os.system(r'net use p: \\10.64.24.50\Shaocheng_Guo')
#upper_dir = 'P:opcode-201707/combined-0707/201707'
#os.system('python opcode2hash.py -c opcode2hash.config')

# step1 : map opcode to hash
#os.system('python opcode2hash.py -c common_pipeline.config')

# step2 : convert hash bits to NN format 
#os.system('python hashbit2NN.py -c common_pipeline.config -pipeline')

#step3: xgboost training
os.system('python xgb_hyopt.py -c common_pipeline.config -pipeline')

#hash2NN
