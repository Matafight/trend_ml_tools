# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import pandas as pd
mean_rank = {}
with  open('./index_mean.txt') as mean_fh:
    lines = mean_fh.readlines()
    rank = 1
    for line in lines:
        if(rank<=247):
            mean_rank[line] = rank
            rank += 1
        else:
            mean_rank[line]= 397

variance_rank= {}
with  open('./index_variance.txt') as variance_fh:
    lines = variance_fh.readlines()
    rank = 1
    for line in lines:
        if(rank<=248):
            variance_rank[line] = rank
            rank += 1
        else:
            variance_rank[line]= 397

chi2_rank= {}
with  open('./index_chi2.txt') as chi2_fh:
    lines = chi2_fh.readlines()
    rank = 1
    for line in lines:
        if(rank<=246):
            chi2_rank[line] = rank
            rank += 1
        else:
            chi2_rank[line]= 397
variance_rank_v2 = {}
for item in mean_rank:
    if item in variance_rank:
        variance_rank_v2[item] = variance_rank[item]
    else:
        variance_rank_v2[item] = 0

chi2_rank_v2={}
for item in mean_rank:
    if item in chi2_rank:
        chi2_rank_v2[item] = chi2_rank[item]
    else:
        chi2_rank_v2[item] = 0


'''total_rank = {}
for item in mean_rank:
    cur_rank = 0
    cur_rank = mean_rank[item] + variance_rank[item] + chi2_rank[item]
    total_rank[item] = cur_rank

sorted(total_rank.iteritems(), key=lambda d:d[1], reverse = False)'''




xg_rank = {}
with  open('./xg_rank.txt') as xg_fh:
    lines =xg_fh.readlines()
    rank = 1
    for line in lines:
        xg_rank[line] = rank
        rank += 1

xg_rank_v2 = {}
for item in mean_rank:
    if item in xg_rank:
        xg_rank_v2[item] = xg_rank[item]
    else:
        xg_rank_v2[item] = 0


mean_rank_key = []
mean_rank_value=[]
variance_rank_v2_key = []
variance_rank_v2_value=[]
chi2_rank_v2_key=[]
chi2_rank_v2_value=[]
xg_rank_v2_key = []
xg_rank_v2_value = []
for item in mean_rank:
    mean_rank_key.append(item)
    mean_rank_value.append(mean_rank[item])

for item in variance_rank_v2:
    variance_rank_v2_key.append(item)
    variance_rank_v2_value.append(mean_rank[item])

for item in chi2_rank_v2:
    chi2_rank_v2_key.append(item)
    chi2_rank_v2_value.append(chi2_rank_v2[item])

for item in xg_rank_v2:
    xg_rank_v2_key.append(item)
    xg_rank_v2_value.append(xg_rank_v2[item])
mean_df = pd.DataFrame({'features':mean_rank_key,'rank':mean_rank_value})
#mean_df.to_csv('mean_df.csv',index=False)

variance_df = pd.DataFrame({'features':variance_rank_v2_key,'variance':variance_rank_v2_value})
#variance_df.to_csv('variance_df.csv',index=False)

chi2_df = pd.DataFrame({'features':chi2_rank_v2_key,'chi2':chi2_rank_v2_value})
#chi2_df.to_csv('chi2_df.csv',index = False)

xg_df = pd.DataFrame({'features':xg_rank_v2_key,'xg_model':xg_rank_v2_value})
#xg_df.to_csv('xg_df.csv',index=  False)

# combine mean_df with variance_df

all_df = mean_df
all_df['variance'] = 0
for i,row in variance_df.iterrows():
    all_df.loc[all_df['features'] == row['features'],'variance'] = int(row['variance'])

all_df['chi2'] = 0
for i,row in chi2_df.iterrows():
    all_df.loc[all_df['features']==row['features'],'chi2'] = int(row['chi2'])
all_df['xg_model'] = 0
for i,row in xg_df.iterrows():
    all_df.loc[all_df['features']==row['features'],'xg_model'] = int(row['xg_model'])

all_df.to_csv('com_rank.csv',index= False)
