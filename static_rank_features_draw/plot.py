import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2


def mean_plot(data,predictors,num_fea_each_plot):
    plot2data = data[predictors]
    plotmean = plot2data.mean()
    dimension = len(predictors)
    
    yp = data.loc[data['label']==1,predictors].mean()
    yn = data.loc[data['label'] == 0,predictors].mean()
    dimension = len(predictors)
    
    #yp = yp + 0.00001
    #yn = yn + 0.00001
    for i in range(dimension):
        param = np.fmax(yp[i],yn[i])
        if(param!=0):
            yp[i] /= param
            yn[i] /= param

    diff = np.abs(yp-yn)
    diff.sort_values(axis=0,ascending=False,inplace = True)
    
    #save indexes
    save_index = diff.index
    thefile = open('../average_rank/ranks/index_mean.txt', 'w')
    for item in save_index:
        thefile.write("%s\n" % item)
    todiv = dimension/num_fea_each_plot
    for i in range(todiv+1):
        fig,ax=plt.subplots()
        x_label =[]
        y_label_p =[]
        y_label_n = []
        if(i==todiv):
            for item in diff.index[i*num_fea_each_plot:dimension]:
                x_label.append(item)
                y_label_p.append(yp[item])
                y_label_n.append(yn[item])
                x_len= dimension-i*num_fea_each_plot
        else:
            for item in diff.index[i*num_fea_each_plot:(i+1)*num_fea_each_plot]:
                x_label.append(item)
                y_label_p.append(yp[item])
                y_label_n.append(yn[item])
                x_len = num_fea_each_plot
        y_label_p = np.array(y_label_p)
        y_label_n = np.array(y_label_n)
        X = range(1,x_len+1)
        p1 = ax.bar(X, +y_label_p, facecolor='#ff9999', edgecolor='white')
        p2 = ax.bar(X, -y_label_n, facecolor='#9999ff', edgecolor='white')
        ax.legend((p1, p2), ('Malware', 'Normal'))
        ax.set_title('Comparison of selected features by their means')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Value')
        ax.set_ylim(-1.1, 1.1)
        plt.xticks(X, x_label, rotation=90)
        ax.set_title('Feature Selection results')
        plt.show()
 
def variance_plot(data,predictors,num_fea_each_plot):
    plot2data = data[predictors]
    plotmean = plot2data.mean()
    dimension = len(predictors)
    
    yp = data.loc[data['label']==1,predictors].var()
    yn = data.loc[data['label'] == 0,predictors].var()
    dimension = len(predictors)
    for i in range(dimension):
        param = np.fmax(yp[i],yn[i])
        yp[i] /= param
        yn[i] /= param
    diff = np.abs(yp-yn)
    diff.sort_values(axis=0,ascending=False,inplace = True)
        #save indexes
    save_index = diff.index
    thefile = open('../average_rank/ranks/index_variance.txt', 'w')
    for item in save_index:
        thefile.write("%s\n" % item)
    todiv = dimension/num_fea_each_plot
    for i in range(todiv+1):
        fig,ax=plt.subplots()
        x_label =[]
        y_label_p =[]
        y_label_n = []
        if(i==todiv):
            for item in diff.index[i*num_fea_each_plot:dimension]:
                x_label.append(item)
                y_label_p.append(yp[item])
                y_label_n.append(yn[item])
                x_len= dimension-i*num_fea_each_plot
        else:
            for item in diff.index[i*num_fea_each_plot:(i+1)*num_fea_each_plot]:
                x_label.append(item)
                y_label_p.append(yp[item])
                y_label_n.append(yn[item])
                x_len = num_fea_each_plot
        y_label_p = np.array(y_label_p)
        y_label_n = np.array(y_label_n)
        X = range(1,x_len+1)
        p1 = ax.bar(X, +y_label_p, facecolor='#ff9999', edgecolor='white')
        p2 = ax.bar(X, -y_label_n, facecolor='#9999ff', edgecolor='white')
        ax.legend((p1, p2), ('Malware', 'Normal'))
        ax.set_title('Comparison of selected features by their means')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Value')
        ax.set_ylim(-1.1, 1.1)
        plt.xticks(X, x_label, rotation=90)
        ax.set_title('Feature Selection results')
        plt.show()
    
def chi2_plot(data,predictors,num_fea_each_plot):
    dimension = len(predictors)
    X = range(1, dimension + 1)
    t = chi2(data[predictors], data['label'])[0]
    

    df = pd.DataFrame({'predictors':predictors,'chi2':t})
    df.fillna(0)
    df.sort_values('chi2',axis=0,ascending = False,inplace=True)
    #t = df['chi2'].values
    save_index = df['predictors']
    thefile = open('../average_rank/ranks/index_chi2.txt', 'w')
    for item in save_index:
        thefile.write("%s\n" % item)
        
    todiv = dimension/num_fea_each_plot
    for i in range(todiv+1):
        
        if(i==todiv):
            y_label = df.iloc[i*num_fea_each_plot:dimension]['chi2'].values
            x_label = df.iloc[i*num_fea_each_plot:dimension]['predictors'].values
        else:
            y_label = df.iloc[i*num_fea_each_plot:(i+1)*num_fea_each_plot]['chi2'].values
            x_label = df.iloc[i*num_fea_each_plot:(i+1)*num_fea_each_plot]['predictors'].values    
        fig,ax=plt.subplots()
        X = range(1,len(x_label)+1)
        ax.bar(X, y_label, facecolor='#ff9999', edgecolor='white')
        ax.set_title('Comparison of selected features by their chi2')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('chi2 Value')

        plt.xticks(X, x_label, rotation=90)
        plt.suptitle('Feature Selection results')
        plt.show()

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train_flags_normalized.csv')
    test_data = pd.read_csv('./data/test_flags_normalized.csv')
    data = pd.concat([train_data,test_data],axis=0)
    data.drop('id',axis=1,inplace = True)
    
    # normalization
    predictors = data.dtypes.index[data.dtypes.index!='label']
    tar = 'label'
    # not work if the max_value = min_value
    data[predictors] = (data[predictors]-data[predictors].min())/(data[predictors].max()-data[predictors].min())
    data[predictors] = data[predictors].fillna(data[predictors].mean())
    data[predictors]=data[predictors].fillna(0)
    num_fea_each_plot = 50
    #mean_plot(data,predictors,num_fea_each_plot)
    #variance_plot(data,predictors,num_fea_each_plot)
    chi2_plot(data,predictors,num_fea_each_plot)




