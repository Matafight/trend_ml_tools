## 主要功能
传入一个目录，目录下的文件是各个features根据各个指标(mean,variance,chi2,xgboost.plot_importance)的重要性排序。

rank.py的作用是读入各个指标的排序算出综合排序并保存到csv文件中。

## 使用方法
```python
python rank.py
```