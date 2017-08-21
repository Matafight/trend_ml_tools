import os

#os.system('python ./opcode2NN/opcode2NN_batch.py -c common_pipeline_shake_512.config')
#os.system('python ./opcode2NN/NN_split.py -c common_pipeline_shake_512.config')
os.system('python xgb_grid_search.py -c common_pipeline_shake_256.config')