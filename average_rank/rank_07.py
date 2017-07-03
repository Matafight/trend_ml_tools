# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import pandas as pd
mean_rank = {}
tar_dir = './ranks/2017-07'
with  open(tar_dir+'/index_mean.txt') as mean_fh:
    lines = mean_fh.readlines()
    rank = 1
    for line in lines:
        mean_rank[line] = rank
        rank += 1

variance_rank= {}
with  open(tar_dir+'/index_variance.txt') as variance_fh:
    lines = variance_fh.readlines()
    rank = 1
    for line in lines:
        variance_rank[line] = rank
        rank += 1

chi2_rank= {}
with  open(tar_dir+'/index_chi2.txt') as chi2_fh:
    lines = chi2_fh.readlines()
    rank = 1
    for line in lines:
        chi2_rank[line] = rank
        rank += 1

variance_rank_v2 = {}
for item in mean_rank:
    if item in variance_rank:
        variance_rank_v2[item] = variance_rank[item]
    else:
        variance_rank_v2[item] = None

chi2_rank_v2={}
for item in mean_rank:
    if item in chi2_rank:
        chi2_rank_v2[item] = chi2_rank[item]
    else:
        chi2_rank_v2[item] = None


'''total_rank = {}
for item in mean_rank:
    cur_rank = 0
    cur_rank = mean_rank[item] + variance_rank[item] + chi2_rank[item]
    total_rank[item] = cur_rank

sorted(total_rank.iteritems(), key=lambda d:d[1], reverse = False)'''




xg_rank = {}
with  open(tar_dir+'/weight_xg_rank.txt') as xg_fh:
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
        xg_rank_v2[item] = None

xg_rank_gain = {}
with  open(tar_dir+'/gain_xg_rank.txt') as xg_fh:
    lines =xg_fh.readlines()
    rank = 1
    for line in lines:
        xg_rank_gain[line] = rank
        rank += 1

xg_rank_v3 = {}
for item in mean_rank:
    if item in xg_rank_gain:
        xg_rank_v3[item] = xg_rank_gain[item]
    else:
        xg_rank_v3[item] = None

xg_rank_cover = {}
with  open(tar_dir+'/cover_xg_rank.txt') as xg_fh:
    lines =xg_fh.readlines()
    rank = 1
    for line in lines:
        xg_rank_cover[line] = rank
        rank += 1

xg_rank_v4 = {}
for item in mean_rank:
    if item in xg_rank_cover:
        xg_rank_v4[item] = xg_rank_cover[item]
    else:
        xg_rank_v4[item] = None

mean_rank_key = []
mean_rank_value=[]
variance_rank_v2_key = []
variance_rank_v2_value=[]
chi2_rank_v2_key=[]
chi2_rank_v2_value=[]
xg_rank_v2_key = []
xg_rank_v2_value = []
xg_rank_v3_key = []
xg_rank_v3_value = []
xg_rank_v4_key = []
xg_rank_v4_value = []
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

for item in xg_rank_v3:
    xg_rank_v3_key.append(item)
    xg_rank_v3_value.append(xg_rank_v3[item])

for item in xg_rank_v4:
    xg_rank_v4_key.append(item)
    xg_rank_v4_value.append(xg_rank_v4[item])

mean_df = pd.DataFrame({'features':mean_rank_key,'rank':mean_rank_value})
#mean_df.to_csv('mean_df.csv',index=False)

variance_df = pd.DataFrame({'features':variance_rank_v2_key,'variance':variance_rank_v2_value})
#variance_df.to_csv('variance_df.csv',index=False)

chi2_df = pd.DataFrame({'features':chi2_rank_v2_key,'chi2':chi2_rank_v2_value})
#chi2_df.to_csv('chi2_df.csv',index = False)

xg_df = pd.DataFrame({'features':xg_rank_v2_key,'xg_model_weight':xg_rank_v2_value})
#xg_df.to_csv('xg_df.csv',index=  False)

xg_df_gain = pd.DataFrame({'features':xg_rank_v3_key,'xg_model_gain':xg_rank_v3_value})

xg_df_cover = pd.DataFrame({'features':xg_rank_v4_key,'xg_model_cover':xg_rank_v4_value})

# combine mean_df with variance_df

all_df = mean_df
all_df['variance'] = 0
for i,row in variance_df.iterrows():
    all_df.loc[all_df['features'] == row['features'],'variance'] = int(row['variance'])

all_df['chi2'] = 0
for i,row in chi2_df.iterrows():
    all_df.loc[all_df['features']==row['features'],'chi2'] = int(row['chi2'])
all_df['xg_model_weight'] = 0
for i,row in xg_df.iterrows():
    all_df.loc[all_df['features']==row['features'],'xg_model_weight'] = row['xg_model_weight']
all_df['xg_model_gain'] = 0
for i,row in xg_df_gain.iterrows():
    all_df.loc[all_df['features']==row['features'],'xg_model_gain'] = row['xg_model_gain']
all_df['xg_model_cover'] = 0
for i,row in xg_df_cover.iterrows():
    all_df.loc[all_df['features']==row['features'],'xg_model_cover'] = row['xg_model_cover']

sum_row = all_df.sum(axis=1)
all_df['aver_rank'] = all_df.mean(axis=1).values
print(all_df.columns)
all_df.sort_values(by = ['aver_rank'],axis=0,inplace=True)
all_df.to_csv('./ranks/comb_rank.csv',index= False)
