## 主要功能
给定一个目录，目录下都是不同的opcode文件。
再给定一个train.csv和test.csv表示不同的训练集测试集划分，每个样本对应一个opcode
classifier.py的任务是根据train.csv和test.csv将opcode文件划分为一下四个子集:
1. malware and train
2. malware and test
3. normal and train
4. normal and test

## 使用方法
```python
python myclassifier.py
```
