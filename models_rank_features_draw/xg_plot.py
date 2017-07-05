import numpy as np
import random
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from operator import itemgetter

def get_fea_index(fscore, max_num_features):
    fea_index = list()
    for key in fscore:
        fea_index.append(int(key[0][1:])-1) # note that the index of array in Python start from 0
        #added by guo
        # the get_fscore function also return the features indexed form 0 ,thus it shouldn't add minus one here
        #update on 5/25/2017, the minus one seems necessary for libsvm format data
        #note that for ordinary numpy array , minus one should not be added
    if max_num_features==None:
        pass
    else:
        fea_index = fea_index[0:max_num_features]
    return np.array(fea_index)

def get_axis_label(fea_index,x_axis_labels):
    x_labels = [x_axis_labels[i] for i in fea_index]
    return x_labels

def fea_plot(xg_model, feature, label, type = 'weight', max_num_features = None, x_axis_label = None):
    fig, AX = plt.subplots(nrows=1, ncols=2)
    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)

    #save ranks to files
    save_rank_file = open('../average_rank/ranks/index_'+type + '.txt','w')
    all_feat_index = get_fea_index(fscore,None)
    all_feat_index = [i+1 for i in all_feat_index]
    print('fscore len')
    print(len(all_feat_index))
    if(x_axis_label!=None):
        all_x_axis_label = get_axis_label(all_feat_index,x_axis_label)
    else:
        all_x_axis_label = all_feat_index
    for item in all_x_axis_label:
        save_rank_file.write("%s\n" % item)
    save_rank_file.close()

    if(x_axis_label !=None):
        mapper = {'f{0}'.format(i): v for i, v in enumerate(x_axis_label)}
        mapped = {mapper[k]: v for k, v in xg_model.get_score(importance_type=type).items()}
        xgb.plot_importance(mapped, xlabel = type,ax = AX[0],max_num_features = max_num_features)
    else:
        xgb.plot_importance(xg_model, xlabel=type, importance_type=type, ax=AX[0], max_num_features=max_num_features)

    print(fea_index)
    print(max_num_features)

    feature = feature[:, fea_index]
    dimension = len(fea_index)
    X = range(1, dimension+1)

    Yp = np.mean(feature[np.where(label==1)[0]], axis=0)
    Yn = np.mean(feature[np.where(label!=1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        if(param !=0):
            Yp[i] /= param
            Yn[i] /= param
        else:
            print('oops!seems wrong')
    p1 = AX[1].bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = AX[1].bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    AX[1].legend((p1,p2), ('Malware', 'Normal'))
    AX[1].set_title('Comparison of selected features by their means')
    AX[1].set_xlabel('Feature Index')
    AX[1].set_ylabel('Mean Value')
    AX[1].set_ylim(-1.1, 1.1)
    #update on 5/25/2017, this line should be added or removed according to the inputdata format
    fea_index = [i+1 for i in fea_index]
    if (x_axis_label !=None):
        tar_x_axis_label = get_axis_label(fea_index,x_axis_label)
    else:
        tar_x_axis_label = fea_index
    plt.xticks(X, tar_x_axis_label, rotation=80)
    plt.suptitle('Feature Selection results')

    #seems useless
    SMALL_SIZE =8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def setcolor(index):
    color = np.zeros([index.shape[0], 4])
    for i in range(0, index.shape[0]):
        if index[i] == 1:
            color[i, 0] = 1
        else:
            color[i, 2] = 1
        color[i, 3] = 1
    return color

def tsne_plot(xg_model,feature, label, size=200, type='weight',max_num_features = None):

    #generate fea_index , it seems that this function doesn't need fea_index attribute
    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)


    fig, AX = plt.subplots(nrows=1, ncols=2)
    # select points for plotting in size:size
    positive = feature[np.where(label==1)[0]]; pos_label = label[np.where(label==1)[0]]
    negative = feature[np.where(label!=1)[0]]; neg_label = label[np.where(label!=1)[0]]

    # In positive
    num_pos = positive.shape[0]
    step = np.ceil(num_pos / size)
    pos_index = list()
    # select positive samples with size = size ,default size is 200
    for i in range(1, size+1):
        if i*step<=num_pos:
            pos_index.append((i-1)*step+int(random.uniform(0, step-0.0000001)))
        else:
            pos_index.append(int(random.uniform((i-1)*step, num_pos - 0.0000001)))
    pos_index = np.asarray(pos_index, np.int8)
    positive = positive[pos_index,:]
    pos_label = pos_label[pos_index]

    # In negative
    num_neg = negative.shape[0]
    step = np.ceil(num_neg / size)
    neg_index = list()
    for i in range(1, size+1):
        if i*step<=num_neg:
            neg_index.append((i-1)*step+int(random.uniform(0, step-0.0000001)))
        else:
            neg_index.append(int(random.uniform((i-1)*step, num_neg - 0.0000001)))
    neg_index = np.asarray(neg_index, np.int8)
    negative = negative[neg_index,:]
    neg_label = neg_label[neg_index]

    feature = np.concatenate([positive, negative])
    label = np.concatenate([pos_label, neg_label])

    tsne_origin = TSNE(learning_rate=100).fit_transform(feature, label)
    tsne_trans = TSNE(learning_rate=100).fit_transform(feature[:, fea_index], label)

    p1 = AX[0].scatter(tsne_origin[:size, 0], tsne_origin[:size, 1], c=setcolor(label[:size]))
    p2 = AX[0].scatter(tsne_origin[size:, 0], tsne_origin[size:, 1], c=setcolor(label[size:]))
    AX[0].legend((p1, p2), ('Malware', 'Normal'), scatterpoints=1)
    AX[0].set_title('Low dimensional structure of original feature space')
    p3 = AX[1].scatter(tsne_trans[:size, 0], tsne_trans[:size, 1], c=setcolor(label[:size]))
    p4 = AX[1].scatter(tsne_trans[size:, 0], tsne_trans[size:, 1], c=setcolor(label[size:]))
    AX[1].legend((p3, p4), ('Malware', 'Normal'), scatterpoints=1)
    AX[1].set_title('Low dimensional structure of selected feature space\nSelected by '+type)
    plt.suptitle('Visualized feature space')

def mean_plot(xg_model,feature, label,type = 'weight',max_num_features = None,x_axis_label = None):

    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)

    feature = feature[:, fea_index]
    dimension = len(fea_index)
    X = range(1, dimension + 1)
    fig,ax = plt.subplots()
    Yp = np.mean(feature[np.where(label == 1)[0]], axis=0)
    Yn = np.mean(feature[np.where(label != 1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        Yp[i] /= param
        Yn[i] /= param
    p1 = ax.bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = ax.bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    ax.legend((p1, p2), ('Malware', 'Normal'))
    ax.set_title('Comparison of selected features by their means')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Value')
    ax.set_ylim(-1.1, 1.1)
    #update on 5/25/2017, this line should be added or removed according to the inputdata format
    fea_index = [i+1 for i in fea_index]
    if (x_axis_label !=None):
        tar_x_axis_label = get_axis_label(fea_index,x_axis_label)
    else:
        tar_x_axis_label = fea_index
    plt.xticks(X, tar_x_axis_label, rotation=80)
    ax.set_title('Feature Selection results')

def variance_plot(xg_model,feature,label,type = 'weight',max_num_features= 30,x_axis_label = None):

    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)

    feature = feature[:, fea_index]
    fig,ax = plt.subplots()
    dimension = len(fea_index)
    X = range(1, dimension + 1)
    Yp = np.var(feature[np.where(label == 1)[0]], axis=0)
    Yn = np.var(feature[np.where(label != 1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        Yp[i] /= param
        Yn[i] /= param
    p1 = ax.bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = ax.bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    ax.legend((p1, p2), ('Malware', 'Normal'))
    ax.set_title('Comparison of selected features by their variances')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Variance Value')
    ax.set_ylim(-1.1, 1.1)
    if (x_axis_label !=None):
        tar_x_axis_label = get_axis_label(fea_index,x_axis_label)
    else:
        tar_x_axis_label = fea_index
    plt.xticks(X, tar_x_axis_label, rotation=80)
    plt.suptitle('Feature Selection results')

def chi2_plot(xg_model,feature, label,type = 'weight',max_num_features = 30,x_axis_label = None):

    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)

    #feature += 0.000001
    _feature = feature[:, fea_index]
    dimension = len(fea_index)
    X = range(1, dimension + 1 + 1)
    t = chi2(_feature, label)[0]
    _chi2 = np.zeros(len(t)+1)
    _chi2[0:len(t)] = t
    fig,ax = plt.subplots()
    all = range(0, feature.shape[1])
    _left = list(set(all).difference(set(fea_index)))
    _chi2_left = chi2(feature[:, _left], label)[0]
    _chi2_left_mean = np.mean(_chi2_left)
    _chi2[len(t)] = _chi2_left_mean
    ax.bar(X, _chi2, facecolor='#ff9999', edgecolor='white')
    ax.set_title('Comparison of selected features by their chi2')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('chi2 Value')
    fea_index = fea_index.tolist()
    #fea_index.append('Other')
    #update on 5/25/2017, this line should be added or removed according to the inputdata format
    fea_index = [i+1 for i in fea_index]
    if (x_axis_label !=None):
        tar_x_axis_label = get_axis_label(fea_index,x_axis_label)
    else:
        tar_x_axis_label = fea_index
    tar_x_axis_label.append('Other')
    plt.xticks(X, tar_x_axis_label, rotation=80)
    plt.suptitle('Feature Selection results')

def stat_plot(xg_model,feature, label, type='weight',max_num_features = 30):

    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)

    fig, AX = plt.subplots(nrows=1, ncols=3)
    plt.suptitle('Feature Selection - Statistical Comparison Results\nBy '+type)

    dimension = len(fea_index)
    X = range(1, dimension + 1);
    feature = feature[:, fea_index]

    Yp = np.mean(feature[np.where(label == 1)[0]], axis=0)
    Yn = np.mean(feature[np.where(label != 1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        Yp[i] /= param
        Yn[i] /= param
    p1 = AX[0].bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = AX[0].bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    AX[0].legend((p1, p2), ('Malware', 'Normal'))
    AX[0].set_title('Comparison of selected features by their means')
    AX[0].set_xlabel('Feature Index')
    AX[0].set_ylabel('Mean Value')
    AX[0].set_ylim(-1.1, 1.1)
    plt.sca(AX[0])
    plt.xticks(X, fea_index, rotation=80)

    Yp = np.var(feature[np.where(label == 1)[0]], axis=0)
    Yn = np.var(feature[np.where(label != 1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        Yp[i] /= param
        Yn[i] /= param
    p1 = AX[1].bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = AX[1].bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    AX[1].legend((p1, p2), ('Malware', 'Normal'))
    AX[1].set_title('Comparison of selected features by their variances')
    AX[1].set_xlabel('Feature Index')
    AX[1].set_ylabel('Variance Value')
    AX[1].set_ylim(-1.1, 1.1)
    plt.sca(AX[1])
    plt.xticks(X, fea_index, rotation=80)

    t = chi2(feature, label)[0]
    _chi2 = np.zeros(len(t) + 1)
    _chi2[0:len(t)] = t

    all = range(0, feature.shape[1])
    _left = list(set(all).difference(set(fea_index)))
    _chi2_left = chi2(feature[:, _left], label)[0]
    _chi2_left_mean = np.mean(_chi2_left)
    _chi2[len(t)] = _chi2_left_mean
    X = range(1, dimension + 2);
    AX[2].bar(X, _chi2, facecolor='#ff9999', edgecolor='white')
    AX[2].set_title('Comparison of selected features by their chi2')
    AX[2].set_xlabel('Feature Index')
    AX[2].set_ylabel('chi2 Value')
    fea_index = fea_index.tolist()
    fea_index.append('Other')
    plt.sca(AX[2])
    plt.xticks(X, fea_index, rotation=80)



