## Requirements
```bash
pip install tqdm
pip install pysha3
```

## opcode2hash_tool.config 配置文件中配置项：
1. source_dir : 源文件夹
2. train: 划分为训练集的csv文件
3. test: 划分为测试集的csv文件
4. algorithm: 选择哈希算法(可选 map_md5和 map_shake)
5. bits :设置hash位数，(md5默认为128位，map_shake可任意设置，不过位数要是8的倍数)
6. split_train_test: 是否将训练测试集分开存放
7. split_good_bad： 是否将正例和负例分开存放
8. ignore_split: 是否忽略上面两个参数，保存所有可能的划分
9. dest_dirs: 保存转换后的数据到该文件夹下