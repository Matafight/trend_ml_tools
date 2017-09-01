## 主要功能
先利用xgboost训练一个模型，之后保存xgboost模型对各个features的重要性排序结果(效果boost内部有三种重要性评价指标：weight,cover,gain)

之后调用plot_importance 方法可视化各个特征的score
## 使用方法
```python
python train_v2.py -c xg_train.conf
```