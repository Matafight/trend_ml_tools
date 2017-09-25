## 主要功能
使用xgboost训练模型，对于超参数，使用自动化脚本执行交叉验证求解。

数据的输入格式： 
```bash
feature1, feature2,...,featuren,label
```

具体的参数设置见文件： xgb_grid_search.config
## 使用方法
```python
python xgb_grid_search.py -c xgb_grid_search.config
```