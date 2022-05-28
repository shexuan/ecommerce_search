## 天池电商搜索挑战赛
https://tianchi.aliyun.com/competition/entrance/531946/information

初赛66名(0.274), 复赛16名(其中召回使用难样本后0.296, 排序最终0.344)

### 安装依赖
初赛(src/recall)使用的bert4keras包，复赛(src/rank)由于官方要求使用的TensorFlow，本代码tf1.14可以跑通。
需要注意的是这份代码中复赛代码和初赛环境不兼容，因此这里初赛复赛使用的是两个不同的环境。
```
pip install jieba
pip install opencc-python-reimplemented
pip install 'h5py==2.10.0' --force-reinstall
pip install pandas
pip install bert4keras
```
其他依赖可根据报错自行安装。

### 运行过程
本仓库代码未进行精细整理，因此运行时需要根据自己实际情况调整数据模型路径。
#### 1、初赛
初赛使用的是simcse模型，分两步进行：
- 第一步使用非监督模式训练，对应 `src/recall/train.sh`;
- 第二步使用第一步训练出来的模型预测得到难样本，然后使用监督模式训练, 对应 `train_hard.sh`。

在监督模式训练时，使用的是3样本元组（query, doc, hard_neg），尝试使用4元组但是在loss设计时出错，后面未细调。

#### 2、复赛
复赛使用的是bert模型，仅尝试了label smoothing和负样本采样。分两步进行：
- 先利用初赛模型生成复赛模型训练数据，对应`src/recall/gen_rank_data.sh`;
- 利用初赛模型生成的样本训练复赛模型，对应`run_train.sh`。

计算资源限制未进行太细致的调参，recall_size设置为10的时候复赛得分0.332，recall_size设置为20的时候复赛得分0.334。






