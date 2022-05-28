#! -*- coding: utf-8 -*-
# 数据读取函数

from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open, _open_
from keras.models import Model
from bert4keras.snippets import DataGenerator, sequence_padding
import tensorflow as tf
import gc, os, json, pickle
import pandas as pd
import random
import faiss


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warmup阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warmup阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)

    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 总共的步数，训练完所有世代的步数epochs * sample_count / batch_size
        self.total_steps = total_steps
        # 全局初始化step
        self.global_step = global_step_init
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 热调整步长，warmup_epoch * sample_count / batch_size
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        # 参数显示  
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []
    #更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
    #更新学习率
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
            

class CustomGeneratorSupervised(DataGenerator):
    """paired训练语料生成器
    """
    def __iter__(self, random):
        batch_token_ids = []
        batch_indexes = []
        for is_end, (q_token_ids, d_token_ids, index) in self.sample(random):
            batch_token_ids.append(q_token_ids)
            batch_token_ids.append(d_token_ids)

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_indexes = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_indexes
                batch_token_ids, batch_indexes = [], []

class CustomGeneratorSupervisedForHardExampleMining(DataGenerator):
    """paired训练语料生成器
    """
    def __iter__(self, random):
        batch_token_ids = []
        batch_indexes = []
        for is_end, (q_token_ids, d_token_ids, hard_d_token_ids, index) in self.sample(random):
            batch_token_ids.append(q_token_ids)
            batch_token_ids.append(d_token_ids)
            batch_token_ids.append(hard_d_token_ids)

            if len(batch_token_ids) == self.batch_size * 3 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_indexes = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_indexes
                batch_token_ids, batch_indexes = [], []

class CustomAugmentGeneratorSupervised(DataGenerator):
    """paired训练语料生成器
    """
    def __init__(self, data, batch_size, tokenizer, dropout_rate=0.1, sample_method="random", max_len=64):
        super(CustomAugmentGeneratorSupervised, self).__init__(data, batch_size)
        self.tokenizer = tokenizer
        self.dropout_rate = dropout_rate
        self.sample_method = sample_method
        self.max_len = max_len
    
    def sample_dtext(self, d_text, comm_text, sample_method="random"):
        """对d_text进行dropout
        两种采样方式：1）从query和doc的交集词汇中进行采样，2）随机从分词中进行采样
        """
        n = len(d_text)
        if n<10:
            return d_text
        else:
            if sample_method=="random":
                sample_idx = random.sample(range(n), k=int(self.dropout_rate*n))
                return [w if idx not in sample_idx else "" for idx, w in enumerate(d_text)]
            else: # 从交集词汇中进行采样
                sample_word = random.sample(comm_text, k=min(int(self.dropout_rate*n), len(comm_text)))
                return [w if idx not in sample_word else "" for idx, w in enumerate(d_text)]
    
    def __iter__(self, random):
        batch_token_ids = []
        batch_indexes = []
        for is_end, (q_text, d_text, comm_text) in self.sample(random):
            d_text = self.sample_dtext(d_text, comm_text, self.sample_method)
            q_token_ids = self.tokenizer.encode(" ".join(q_text), maxlen=self.max_len)[0]
            d_token_ids = self.tokenizer.encode(" ".join(d_text), maxlen=self.max_len)[0]
            
            batch_token_ids.append(q_token_ids)
            batch_token_ids.append(d_token_ids)

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_indexes = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_indexes
                batch_token_ids, batch_indexes = [], []


class CustomGeneratorUnSupervised(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random):
        batch_token_ids, batch_indexes = [], []
        for is_end, (token_ids, index) in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
            
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []
                

class CustomGeneratorForPredict(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_indexes = [], []
        for is_end, (token_ids, index) in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_indexes.append(index)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_indexes
                batch_token_ids, batch_indexes = [], []
                

class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    注:
     评估的时候doc文档必须保证ID从小到大，因为后面计算时将索引当做其ID进行计算；
     query文档可无序；
    """
    def __init__(self, dev_generator=None, doc_generator=None, model_save_dir=None,
                 base_model="bert", train_query_generator=None):
        self.best_mrr = -1.
        self.dev_generator = dev_generator
        self.train_query_generator = train_query_generator
        self.doc_generator = doc_generator
        self.save_dir = model_save_dir
        self.base_model = base_model

    
    def on_epoch_end(self, epoch, logs=None):
        save_dir = "{0}/epoch{1}".format(self.save_dir, epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights("{0}/epoch{1}.weight".format(save_dir, epoch))
        self.dev_emb = l2_normalize(self.predict(self.model, self.dev_generator)[0])
        self.corpus_emb = l2_normalize(self.predict(self.model, self.doc_generator)[0])
        save_embedding(self.dev_emb, save_dir+"/query_embedding", 200001)
        save_embedding(self.corpus_emb, save_dir+"/doc_embedding", 1)
    
    def evaluate(self, query_generator, doc_generator):
        """评估MRR10, TOP10的召回准确率
        """
        query_emb, qids = self.predict(self.model,  query_generator)
        assert (qids>0).all(), "Some query ID <= 0"
        doc_emb, dids = self.predict(self.model, doc_generator)
        
        query_emb = l2_normalize(query_emb).astype(np.float32)
        doc_emb = l2_normalize(doc_emb).astype(np.float32)
        
        # 归一化后dot结果与欧氏距离、余弦距离呈正相关，因此可用dot结果来进行召回
        doc_scores = np.dot(query_emb, doc_emb.T)
        
        # 按dot结果逆向排序，sortidx中的元素为scores各行的索引, 转为doc ID需要+1
        sortidx = np.argsort(doc_scores)[:,::-1] + 1
        del doc_scores
        gc.collect()
        row_idx, gold_idx = np.where(sortidx==qids)
        
        assert len(qids)==len(gold_idx), "部分query的匹配doc ID未找到"
        
        # 求MRR指标，索引从0开始因此需要+1
        gold_idx = gold_idx+1

        mrr10 = np.where(gold_idx>10, 0, 1/gold_idx).mean()
        mrr50 = np.where(gold_idx>50, 0, 1/gold_idx).mean()
        mrr100 = np.where(gold_idx>100, 0, 1/gold_idx).mean()
        
        return mrr10, mrr50, mrr100
    
    @staticmethod
    def predict(model, data_pred_generator):
        """data 包括三列 => ([token_ids, segment_id], label)
        """
        vecs, indexes = [], []
        for token_ids, idx in tqdm(data_pred_generator):
            vec = model.predict(token_ids, verbose=False)
            vecs.append(vec)
            indexes.extend(idx)
        
        indexes = np.array(indexes).reshape(-1, 1)
        
        return np.vstack(vecs), indexes
    
    @staticmethod
    def predict_raw_tokens(model, data_encode, batch_size=512):
        """data 包括三列 => ([token_ids, segment_id], label)
        """
        vecs, indexes = [], []
        batch_tokens, batch_segments = [], []

        for token_ids, idx in tqdm(data_encode):
            batch_tokens.append(token_ids)
            indexes.append(idx)

            if len(batch_tokens)==batch_size:
                batch_tokens = sequence_padding(batch_tokens)
                batch_segments = np.zeros_like(batch_tokens)

                vec = model.predict([batch_tokens,batch_segments], verbose=False)
                vecs.append(vec)
                batch_tokens = []

        # 处理最后剩余部分
        batch_tokens = sequence_padding(batch_tokens)
        batch_segments = np.zeros_like(batch_tokens)

        # simbert预测值有两个cls_output和mlm_output,取第一个cls输出即可
        vec = model.predict([batch_tokens,batch_segments], verbose=False)
        vecs.append(vec)
        batch_token_ids = [], []

        indexes = np.array(indexes).reshape(-1, 1)

        return np.vstack(vecs), indexes
    
    
def get_hard_examples(query_emb, doc_emb, doc_idx, nlist=100, 
                      topK=50, use_gpu=False, emb_size=128):
    """从doc中获取难样本, 使用faiss来查找最近邻向量
    faiss使用参考：https://www.cnblogs.com/sug-sams/p/12607662.html
    embedding需要为np.float32格式.
    nlist: 聚类中心个数；
    topK: 召回近邻个数；
    use_gpu: 是否使用GPU
    """
    use_gpu = False
    # IndexFlatL2表示利用L2距离来比较特征的相似度
    index_l2 = faiss.IndexFlatL2(emb_size) # 向量维度为128
    # 使用GPU
    if use_gpu:
        res = faiss.StandardGpuResources()
        index_l2 = faiss.index_cpu_to_gpu(res, 0, index_l2)

    # 倒排索引进行加速
    index_l2 = faiss.IndexIVFFlat(index_l2, emb_size, nlist, faiss.METRIC_L2)

    # 倒排索引加速需要先训练 k近邻
    index_l2.train(doc_emb)

    # 手动传入向量index
    index_l2 = faiss.IndexIDMap(index_l2)
    index_l2.add_with_ids(doc_emb, np.array(doc_idx).astype('int64'))
    # 查找近邻
    D, I = index_l2.search(query_emb, topK)

    return I
        

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


def hard_supervise_loss(y_true, y_pred):
    """https://www.icode9.com/content-4-1357759.html"""
    row = K.arange(0, K.shape(y_pred)[0], 3)
    col = K.arange(K.shape(y_pred)[0])
    col = K.squeeze(tf.where(K.not_equal(col % 3, 0)), axis=1)
    y_true = K.arange(0, K.shape(col)[0], 2)
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred)) # tf.matmul(y_pred, y_pred, adjoint_b = True)

    similarities = tf.gather(similarities, row, axis=0)
    similarities = tf.gather(similarities, col, axis=1)

    similarities = similarities * 20
    loss = K.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)
    # loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)

def super_hard_supervise_loss(n_samle_per_pair=3):
    """n_samle_per_pair表示每一组的样本数，原始一对样本数量为2，
        加上一个难样本一组则为3, 3个难样本一组则为5
    """
    def loss_fn(y_true, y_pred):
        row = K.arange(0, K.shape(y_pred)[0], n_samle_per_pair)
        col = K.arange(K.shape(y_pred)[0])
        col = K.squeeze(tf.where(K.not_equal(col % n_samle_per_pair, 0)), axis=1)
        y_true = K.arange(0, K.shape(col)[0], n_samle_per_pair - 1)
        y_pred = K.l2_normalize(y_pred, axis=1)
        similarities = K.dot(y_pred, K.transpose(y_pred))  # tf.matmul(y_pred, y_pred, adjoint_b = True)

        similarities = tf.gather(similarities, row, axis=0)
        similarities = tf.gather(similarities, col, axis=1)

        similarities = similarities * 20
        loss = K.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)
        # loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)
    return loss_fn


def save_embedding(embeddings, save_path, start_idx):
    padding = ("0,"*63).strip(",")
    with open(save_path, 'w', encoding="utf-8") as up:
        for idx, feat in enumerate(embeddings):
            _ = up.write('{0}\t{1}\t{2},{3}\n'.format(idx+start_idx, ','.join(["%0.4f"%x for x in feat]), idx, padding))
            
def load_embedding(fpath):
    with open(fpath, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
        index = [line[0] for line in lines]
        emb = [line[1].split(",") for line in lines]
        emb = np.array(emb).astype(np.float32)
        index = np.array(index).astype(np.int32)
    
        return index, emb
    
def get_paired_data(args):
    ROOT_DIR = "../../data/raw_data/"
    # 全部的语料，train_query, dev_query, doc
    # train pair 语料
    df_pair_raw = pd.read_pickle(f"{ROOT_DIR}/train_pairs_cut_rm_stop_words.pkl")[["qid","did","qtext_cut", "dtext_cut"]]
    df_pair_ecom = None
    df_pair_video = None
    df_pair = None

    # 额外电商数据
    if args["use_ecom"]:
        ECOM_DIR = "../../data/ecom/"
        df_pair_ecom = pd.read_pickle(f"{ECOM_DIR}/train_pairs_cut_rm_stop_words.pkl")[["qid","did","qtext_cut", "dtext_cut"]]
        # 添加上额外的电商pair数据
        df_pair = df_pair_raw.append(df_pair_ecom)
        
    # 额外视频数据
    if args["use_video"]:
        VIDEO_DIR = "../../data/video/"
        df_pair_video = pd.read_pickle(f"{VIDEO_DIR}/train_pairs_cut_rm_stop_words.pkl")[["qid","did","qtext_cut", "dtext_cut"]]    
        # 添加上额外的视频数据
        df_pair = df_pair.append(df_pair_video)
    
    return df_pair, df_pair_raw, df_pair_ecom, df_pair_video

def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], float(l[2])))
    return D

def get_encoded_data_list_single(df, tokenizer, pair_dict, mode):
    """加载数据（带标签）
    单条格式：(文本1, 文本ID, label), 若为无监督，则label均设置为0
    """
    texts = df[["id","text_cut"]].values
    data = []
    for id_, sent in texts:
        # 因为只有一个句子，因此这里暂不保留segment_id以节约内存, (文本1, doc文本ID, label)
                                                        # pair_dict.get(id_, 0)
        data.append([tokenizer.encode(" ".join(sent))[0], pair_dict.get(id_, 0)])
    
    return data

def gen_encode_data_for_predict(tokenizer, suffix="roformer_tokenizer"):
    """生成用来预测得到embedding的 dev 和 corpus 的encode数据
    """
    ROOT_DIR = "../../data/raw_data/"
    # 全部语料
    df_all = pd.read_pickle(f"{ROOT_DIR}/docs_cut_rm_stop_words.pkl")[["id", "text_cut", "source"]]
    # 获取编码后的语料
    corpus_encode = get_encoded_data_list_single(df_all.query("source=='doc'"),
                                                 tokenizer, {}, mode="predict")

    # 测试集
    dev_encode = get_encoded_data_list_single(df_all.query("source=='dev'"),
                                              tokenizer, {}, mode="predict")
    with _open_(f"{ROOT_DIR}/doc_encode_{suffix}.pkl", "wb") as f:
        pickle.dump(corpus_encode, f)
    
    with _open_(f"{ROOT_DIR}/dev_encode_{suffix}.pkl", "wb") as f:
        pickle.dump(dev_encode, f)

def get_encoded_paird_from_paired(df, tokenizer, maxlen=64):
    """按对读入数据
    单条格式：(query, doc)
    输出格式：(query_token_id, doc_token_id, doc_id)
    """
    texts = df[["qid","did","qtext_cut", "dtext_cut"]].sort_values(by="qid").values
    data = []
    for qid, did, qtext, dtext in tqdm(texts):
        
        q_token_ids = tokenizer.encode(" ".join(qtext), maxlen=maxlen)[0]
        d_token_ids = tokenizer.encode(" ".join(dtext), maxlen=maxlen)[0]
        data.append([q_token_ids, d_token_ids, 1])
    
    return data

def get_raw_text_paird_from_paired(df):
    """按对读入数据
    单条格式：(query, doc)
    输出格式：(query_token_id, doc_token_id, doc_id)
    """
    texts = df[["qid","did","qtext_cut", "dtext_cut"]].values
    data = []
    for qid, did, qtext, dtext in tqdm(texts):
        data.append([qtext, dtext, list(set(qtext) & set(dtext))])
    
    return data

def get_encoded_paired_from_unpaired(df, tokenizer, sample_num, maxlen=64):
    """读取未配对的doc，半监督数据中的无监督部分
    输入单条格式：(query, id)
    输出：(query_token_id, query_token_id, id)
    """
    if sample_num<=0:
        return []
    df_dev = df.query("source=='dev'")
    doc_n = sample_num - len(df_dev) if sample_num>len(df_dev) else 0
    df_doc = df.query("source=='doc'")
    doc_n = doc_n if doc_n<=len(df_doc) else len(df_doc)
    df_doc = df_doc.sample(n=doc_n).append(df_dev)
    data = []
    for qid, text in df_doc[["id","text_cut"]].values:
        q_token_ids = tokenizer.encode(" ".join(text), maxlen=maxlen)[0]
        data.append([q_token_ids, q_token_ids, 1])
    
    return data

def get_tokenizer(dict_path, pre_tokenize=None):
    """建立分词器
    """
    return Tokenizer(dict_path, do_lower_case=True, pre_tokenize=pre_tokenize)

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def get_encoder(
    config_path,
    checkpoint_path,
    model='bert',
    pooling='first-last-avg',
    dropout_rate=0.1,
    activation=None
    ):
    """建立编码器
    P1：把encoder的最后一层的[CLS]向量拿出来；
    P2：把Pooler（BERT用来做NSP任务）对应的向量拿出来，跟P1的区别是多了个线性变换；
    P3：把encoder的最后一层的所有向量取平均；
    P4：把encoder的第一层与最后一层的所有向量取平均。
    """
    
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']

    if pooling == 'pooler':
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            with_pool='linear',
            dropout_rate=dropout_rate
        )
    else:
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            dropout_rate=dropout_rate
        )

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    elif pooling == 'cls':
        output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    elif pooling == 'pooler':
        output = bert.output
    
    output = keras.layers.Dense(128, activation=activation)(output)
    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder


def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        try:
            token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
            a_token_ids.append(token_ids)
            token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
            b_token_ids.append(token_ids)
            labels.append(d[2])
        except:
            print(d)
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def write_to_json(jpath, data):
    with open(jpath, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)
        print("write json file success!")
        
def read_json(jpath):
    with open(jpath, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        print("loads json file success!")
        return data