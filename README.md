# trend_ml_tools
This is the tool scripts I used in trendmicro

1. average_rank

根据models_rank_features_draw和statistics_rank_features_draw两个文件夹中的代码生成的各个指标的排序，算出综合排序

2. combine_csvs

由于原始特征分散在各个文件中，combine_csvs的主要功能是合并特征

3. datas

输入数据的文件夹

4. models_rank_features_draw

根据xgboost训练的模型输出的各个特征的分数进行排序，并画图比较

5. statistics_rank_features_draw

根据mean,variance,chi2,mutual information等指标算出特征的排序，并画图比较