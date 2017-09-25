# trend_ml_tools
This is the tool scripts I used in trendmicro

# 各个文件夹介绍

## 1. average_rank

根据models_rank_features_draw和statistics_rank_features_draw两个文件夹中的代码生成的各个指标的排序，算出综合排序

## 2. combine_csvs

由于原始特征分散在各个文件中，combine_csvs的主要功能是合并特征


## 3. models_rank_features_draw

根据xgboost训练的模型输出的各个特征的分数进行排序，并画图比较

## 4. statistics_rank_features_draw

根据mean,variance,chi2,mutual information等指标算出特征的排序，并画图比较

## 5. combineOpocdes
根据给定的train.csv和test.csv文件将指定目录下的opcode文件划分重组，使得满足以下条件的opcode各自存储在一个目录下:
- malware and train
- malware and test
- normal and train
- normal and test

## 6. opcodehash
将给定的opcode按照相应hash算法映射成一定长度的01编码，以该01编码为features训练模型

## 7. xgboost_training_ordinary_data
完整的xgboost的交叉验证调参过程，不过输入的 data是 pandas 的dataframe的格式。

# 使用方法
具体各个工具的使用方法详见各个文件夹内的README文件。