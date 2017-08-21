import xgboost as xgb
import argparse
import ConfigParser
import pprint
import pickle
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from tools.tools import get_predictedResults_from_file,get_csr_labels
from xg_predict_comp import compare_models,compare_roc_auc,compare_pr_auc,compare_confusion_matrix,compare_path_score

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=False)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path,is_verbose=False):
    global models_path
    global model_names_list
    global thres_list
    global marker_list
    global dataset_list
    global dataset_f_list
    global save_path


    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    models_path = cf.get('draw','model_paths').split(',')
    model_names_list = cf.get('draw', 'model_names').split(',')
    thres_list = cf.get('draw','thres').split(',')
    thres_list = [float(i) for i in thres_list]
    marker_list = cf.get('draw','markers').split(',')
    dataset_list = cf.get('draw','datasets').split(',')
    dataset_f_list = cf.get('draw','dataset_formats').split(',')
    save_path = cf.get('draw','save_path')
    if is_verbose:
        pprint.pprint(models_path)
        pprint.pprint(model_names_list)
        pprint.pprint(marker_list)
    return

if __name__ == '__main__':

    models_path = None
    model_names_list = None
    thres_list = None
    marker_list = None
    dataset_list = None
    dataset_f_list = None
    label_f_list = None

    arg = arg_parser()
    conf_parser(arg.conf,is_verbose=True)

    labels_list = list()
    pred_scores_list = list()

    for i,f in enumerate(dataset_f_list):
        print('=============================')
        if f  == 'xgboost':
            print(dataset_list[i])
            print(models_path[i])
            X,y = load_svmlight_file(dataset_list[i])
            X = X.todense()
            dtest = xgb.DMatrix(X,label = y)
            labels = dtest.get_label()
            labels = labels.astype(int)
            model_temp = xgb.Booster(model_file=models_path[i])
            labels_pred = model_temp.predict(dtest)
            
            print(labels_pred)
            labels_list.append(labels)
            pred_scores_list.append(labels_pred)

    measures_all = compare_models(labels_list,pred_scores_list,model_names_list,thres_list)
    compare_roc_auc(fid=1,measures=measures_all,marker_list=marker_list,save_path = save_path)
    compare_pr_auc(fid=2, measures=measures_all, marker_list=marker_list,save_path = save_path)
    compare_roc_auc(fid=3, measures=measures_all,axis_interval=[0,1,0.9,1],marker_list=marker_list,save_path = save_path)
    compare_pr_auc(fid=4, measures=measures_all, axis_interval=[0,1,0.9,1],marker_list=marker_list,save_path = save_path)
    compare_confusion_matrix(measures_all)