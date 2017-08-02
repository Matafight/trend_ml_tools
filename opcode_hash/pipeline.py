import os

os.system('python opcode2NN_batch.py -c common_pipeline.config')
os.system('python NN_split.py -c common_pipeline.config')
os.system('python xgb_grid_search.py -c common_pipeline.config')