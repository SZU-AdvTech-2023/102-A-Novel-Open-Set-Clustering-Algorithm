# DOS-IN优化: DOS-IN聚类算法优化



## Description

DOS (**D**elta **O**pen **S**et) 聚类算法能够识别复杂形状的簇，但是其对输入参数的依赖性大，且由于其通过特定函数生成规则邻域用于识别开集，导致DOS在处理重叠的集群和高斯集群时表现较差。本文复现的DOS-IN（**I**rregular **N**eighborhoods）聚类算法基于对象之间的相似度生成不规则邻域，能够自适应对象的分布，不仅可以准确地区分重叠的簇，而且具有更少的输入参数。此外，DOS-IN引入了小簇合并机制，解决DOS在识别高斯簇方面的不足。但DOS–IN算法在数据预处理与小簇合并机制设计上仍存在不足，为此，本文在程序中加入了特征工程的数据预处理方法并优化了小簇合并机制，有效提高该算法的准确率。



## Reproduction Instructions

environment can be seen in `env.txt`



to reproduce the Figure 2,3,4:

```shell
cd synthetic_data
python DOS-IN-plot.py
```



to reproduce the Figure 5,7,9,11,13:

```shell
cd feature_engineering/DOS-IN
python picture.py
```


to reproduce the Figure 15,16,17:

```shell
cd feature_engineering/DOS-IN
python PCA_picture.py
```



to reproduce the Figure 6,8,10,12,14,17,18:

```shell
cd synthetic_data
python DOS-IN.py
```

