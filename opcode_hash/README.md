## pipeline简介
1. opcode2hash.py
2. hash2NN.py
3. xgb_hyopt.py

代码运行分为两种模式：
1. pipeline模式
2. 独立模式

## 输入数据的目录结构
```
├─malicious
│  ├─test
│  │  └─1
│  └─train
│      └─1
├─map_md5_48
│  ├─test
│  │  ├─malware
│  │  └─normal
│  └─train
│      |─malware
|      └─normal
└─normal
    ├─test
    │  └─0
    └─train
       └─0
```
malicious和normal文件夹存储的是原始opcode文件，map_md4_48文件夹内存储的是由opcode文件生成的hash编码,算法支持md5(128 bits)和sha-3(支持任意长度的散列，不过位数要满足8的倍数,以字节为单位生成散列值)