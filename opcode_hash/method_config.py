from hyperopt import hp,tpe,STATUS_OK,Trials,space_eval
import numpy as np

xgb_min_num_round = 10
xgb_max_num_round = 100
xgb_num_round_step = 5
xgb_random_seed = 10
# we should use the huber loss function
# for hp.choice function, the best_params returned by fmin is the index of the list, so try not use hp.choice
#booster params
param_xgb_space = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': 5,
    'silent':1,
    'seed': xgb_random_seed,
    "max_evals": 50,
}