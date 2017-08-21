## opcode_hash 工具简介
### 整个pipeline主要分为三步：
1. 将给定的opcode文件转换成二进制的编码，需要预先设置hash算法并设置位数，详见./opcode2NN/opcode2hash_tool.config 配置文件说明,使用方法详见 ./opcode2NN/README.md。
    
2. 给定训练集和测试集的划分，将第一步的二进制编码划分为训练集和测试集，该功能的配置文件为 ./opcode2NN/NN_split.config, 使用方法详见./opcode2NN/README.md

3. 使用xgboost算法训练第二步的数据。
    - xgb_grid_search.py  主要训练代码
    - xgb_grid_search.config  xgboost算法配置文件
    - 代码会将训练好的模型保存在./models/目录下，并且会将模型在测试集上的混淆矩阵保存在./log/目录下。

### 其他功能
1. 画出ROC和PR曲线并保存到./curves/目录下
    - draw.py   画图主文件
    - draw.config  画图配置文件




