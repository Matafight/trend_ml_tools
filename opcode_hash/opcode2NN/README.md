## 该文件夹下包含两个小脚本
1. opcode2NN_batch.py 将一个目录下的所有opcode文件(通过递归查找)通过哈希算法转换成01编码，并保存为NNAI格式。
    每一个opcode文件转换成的向量为一个样本，由于当前并不知道样本的类标号，将其类标号置为0(0 表示标号未知，1表示正例样本，2表示负例样本)，将样本和类标号分别存储在hashvec.dat和hashvec.dat_AI文件中(可自定义文件名)，hashvec.dat和hashvec.dat.AI的文件格式如下：
    - hashvec.dat
        第一行为样本特征维度
        从第二行开始，每行为一个样本且每行的格式为: num1;num2;非零特征个数;第一个非零特征索引（索引从1开始）;第一个非零特征值;第二个非零特征索引;第二个非零特征值;...,注意这里的num1和num2可以是任意数值。
    - hashvec.dat.AI
        该文件存储hashvec.dat中每行样本对应的类标号和opcode文件路径
        标号|文件路径
    

2. NN_split.py 读取给定的train.csv和test.csv(或其他表示训练集测试集划分的文件)，对第一步保存的hashvec.dat和hashvec.dat.AI进行划分，并根据train.csv和test.csv中提供的类标号重新对样本标号。
    - train.csv和test.csv格式：至少包含两列数据，分别是opcode文件名(id)和类标号(malware)，列名分别是id和malware。



## 两个脚本各自对应这一个配置文件：
1. opcode2hash_tool.config
    需要设置的参数为:
    - source_dirs: 包含opcode文件的目录，程序通过递归地查找source_dirs及其子目录中的opcode文件并对其hash
    - algorithm： 设置hash算法，可选项为:map_md5和map_shake
    - bits: 设置文件hash之后的比特位数(需要能被8整除)，该选项只对map_shake算法有效，当算法设置为map_md5时，位数默认为128
    - dest_features_path: 设置hashvec.dat文件的保存路径，需要具体到文件名
    - dest_labels_path: 设置hashvec.dat.AI文件的保存路径，需要具体到文件名
2. NN_split.config
    需要设置的参数为：
    - train: 被划分为训练集的opcode编号及相应类标号的csv文件路径，具体到文件名
    - test: 被划分为测试集的opcode及其类标号的csv文件路径，具体到文件名
    - source_features_path: 样本全集的特征路径(hashvec.dat路径)
    - source_labels_path: 样本全集的类标号路径(hashvec.dat.AI路径)
    - split_train_test: 是否将全集中的训练集和测试集分开，1表示分开
    - split_malware_normal:是否将全集中malware和normal样本分开，1表示分开
    - dest_dir: 划分后的数据的存放目录，不用具体到文件名


## 使用方法
对于第一个脚本，写好配置文件之后在命令行下执行：
```python
python opcode2NN_batch.py -c opcode2hash_tool.config
```

第二个脚本：
```python
python NN_split.py -c NN_split.config
```

## 依赖项
```bash
pip install pysha3
pip install tqdm
```