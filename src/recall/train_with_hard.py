#! -*- coding: utf-8 -*-

import sys, os
# 使用混合精度训练
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'
# os.environ['TF_KERAS'] = "1"

from utils import *
from glob import glob
import argparse
import pandas as pd
import numpy as np
import pickle
from functools import partial

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding, _open_
import gc
import time
import random
from tqdm import tqdm

RECALL_TOPK = 50
N_HARD_GENERATE = 3  # 每一组数据生成几个难样本备用


class CustomGeneratorSupervisedForHardExampleMining(DataGenerator):
    """paired训练语料生成器
    """

    def __init__(self, data, batch_size, n_hard=1, max_len=64):
        super(CustomGeneratorSupervisedForHardExampleMining, self).__init__(data, batch_size)
        # 使用几个难样本
        self.n_hard = n_hard
        assert self.n_hard >= 1 and self.n_hard <= 3, "hard negative doc must >=1 and <=3"
        self.max_len = max_len

    def __iter__(self, random):
        batch_token_ids = []
        batch_indexes = []
        for is_end, hard_token_ids in self.sample(random):
            batch_token_ids.append(hard_token_ids[0])  # query token_ids
            batch_token_ids.append(hard_token_ids[1])  # true label doc token_ids
            batch_token_ids.append(hard_token_ids[2])  # hard negative token_ids

            if self.n_hard > 1:
                # hard negative token_ids
                for i in range(3, 2 + self.n_hard):
                    batch_token_ids.append(hard_token_ids[i])
            
            if (len(batch_token_ids) == (self.batch_size * (2 + self.n_hard))) or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_indexes = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_indexes
                batch_token_ids, batch_indexes = [], []


def get_data_hard_neg_helper(data_generator, model, doc_emb, doc_encode_data, paired_encode_data):
    """提取原始数据的难样本
    data_generator, query token数据生成器；
    model：用来预测的模型；
    doc_emb: 语料库embedding；
    doc_encode_data: 编码好的doc语料tokens，[[d1_token_ids, id1], [d2_token_ids, id2], ...];
    paired_encode_data: 编码好的pair tokens，[[q1_token_ids, d1_token_ids, id1], [q2_token_ids, d2_token_ids, id2], ...];
    """
    query_emb = l2_normalize(Evaluator.predict(model, data_generator)[0])
    doc_idx = np.arange(len(doc_emb))
    recall_doc_arr = get_hard_examples(query_emb, doc_emb, doc_idx, nlist=100,
                                       topK=RECALL_TOPK, use_gpu=False, emb_size=128)
    print("recall_doc_arr max:", recall_doc_arr.max())
    print("recall_doc_arr min:", recall_doc_arr.min())

    data = []
    for idx, doc_ids in enumerate(recall_doc_arr):
        doc_hard_negs = []
        for i in range(N_HARD_GENERATE):
            doc_hard_neg_idx = doc_ids[HARD_NEG_RANK + i * 10]
            doc_hard_neg = doc_encode_data[doc_hard_neg_idx][0]
            # paired_encode_data[idx][1] 为query的真实结果，若所选难负doc为label，则换一个
            if doc_hard_neg == paired_encode_data[idx][1]:
                doc_hard_neg_idx = doc_ids[HARD_NEG_RANK + i * 10 - 1]
                doc_hard_neg = doc_encode_data[doc_hard_neg_idx][0]
            doc_hard_negs.append(doc_hard_neg)

        paired_sample = [paired_encode_data[idx][0], paired_encode_data[idx][1]]
        paired_sample.extend(doc_hard_negs)
        paired_sample.append(1)
        data.append(paired_sample)

    return data


def get_data_hard_neg(args, encoder, tokenizer, corpus_encode, hard_encode_data_path=None):
    """生成难样本
    """
    if hard_encode_data_path and os.path.exists(hard_encode_data_path):
        print(f"hard triple encode data 【{hard_encode_data_path}】 exists, loading ... ...")
        with _open_(hard_encode_data_path, "rb") as f:
            hard_triple_encode = pickle.load(f)
        return hard_triple_encode

    print("hard triple encode data Not exists, generating ... ...")
    df_pair, df_pair_raw, df_pair_ecom, df_pair_video = get_paired_data(args)

    del df_pair
    gc.collect()
    # 编码
    data_pair_raw_encode = get_encoded_paird_from_paired(df_pair_raw, tokenizer, maxlen=64)
    data_pair_ecom_encode = get_encoded_paird_from_paired(df_pair_ecom, tokenizer, maxlen=64)
    data_pair_video_encode = get_encoded_paird_from_paired(df_pair_video, tokenizer, maxlen=64)

    # query sentence encode，用于预测，提取难样本
    raw_query_encode = [[pair[0], 1] for pair in data_pair_raw_encode]
    ecom_query_encode = [[pair[0], 1] for pair in data_pair_ecom_encode]
    video_query_encode = [[pair[0], 1] for pair in data_pair_video_encode]

    raw_query_generator = CustomGeneratorForPredict(raw_query_encode, 512)
    ecom_query_generator = CustomGeneratorForPredict(ecom_query_encode, 512)
    video_query_generator = CustomGeneratorForPredict(video_query_encode, 512)

    # 载入已提前训练好的doc_embedding, 用于召回难样本
    _, doc_emb = load_embedding(args["stage1_model_path"] + '/doc_embedding')
    # 获取包含hard负样本的三元组
    raw_hard_triple_encode = get_data_hard_neg_helper(raw_query_generator, encoder, doc_emb,
                                                      corpus_encode, data_pair_raw_encode)
    ecom_hard_triple_encode = get_data_hard_neg_helper(ecom_query_generator, encoder, doc_emb,
                                                       corpus_encode, data_pair_ecom_encode)
    video_hard_triple_encode = get_data_hard_neg_helper(video_query_generator, encoder, doc_emb,
                                                        corpus_encode, data_pair_video_encode)

    del raw_query_encode, ecom_query_encode, video_query_encode, doc_emb
    del data_pair_raw_encode, data_pair_ecom_encode, data_pair_video_encode
    gc.collect()

    hard_triple_encode = raw_hard_triple_encode + ecom_hard_triple_encode + video_hard_triple_encode

    # 保存数据，下次直接读取
    print(f"Saving hard triple encode data to {args['stage1_model_path']}/hard_triple_encode.pkl")
    with _open_(args["stage1_model_path"] + "/hard_triple_encode.pkl", "wb") as f:
        pickle.dump(hard_triple_encode, f)

    return hard_triple_encode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimSCE model training.")
    parser.add_argument("--model_name", "-model", type=str,
                        choices=["bert", "robert", "robert_large", "roformer", "simcse",
                                 "roformer_sim", "roformer_sim_ft", "simbert"],
                        default="bert", help="模型名字")
    parser.add_argument("--base_model", type=str,
                        choices=["bert", "roformer"],
                        default="bert", help="模型名字")
    parser.add_argument("--pooling_type", "-pt", type=str,
                        choices=['first-last-avg', 'last-avg', 'cls', 'pooler'],
                        default="cls", help="模型最后一层输出形式")
    parser.add_argument("--train_batch_size", type=int,
                        default=42, help="训练batch_size")
    parser.add_argument("--dropout_rate", "-dropout", type=float,
                        default=0., help="dropout 概率")
    parser.add_argument("--epoch", default=3, type=int,
                        help="是否使用额外的数据")
    parser.add_argument("--mode", type=str,
                        choices=["debug", "train", "predict"],
                        default="train", help="运行模式")
    parser.add_argument("--fgm", action="store_true",
                        default=False, help="是否进行对抗训练")
    parser.add_argument("--sample_num", type=int,
                        default=100000, help="采样无监督数据")
    parser.add_argument("--learning_rate", "-lr", type=float,
                        default=1e-5, help="学习率")
    parser.add_argument("--lr_decay", action="store_true",
                        default=False, help="学习率衰减")
    parser.add_argument("--use_ecom", action="store_true",
                        default=False, help="是否使用额外的电商数据")
    parser.add_argument("--use_video", action="store_true",
                        default=False, help="是否使用额外的视频搜索数据")
    parser.add_argument("--suffix", default="",
                        type=str, help="保存文件的目录名后缀")
    parser.add_argument("--aug", default="false",
                        choices=["false", "random", "intersect"],
                        type=str, help="是否使用数据扩增以及何种数据扩增方式")
    parser.add_argument("--eval", action="store_true",
                        default=False, help="是否保留一部分数据用于离线评估")
    parser.add_argument("--hard_neg_rank", type=int,
                        default=9, help="取召回的第几个作为难样本")
    parser.add_argument("--use_n_neg", type=int, choices=[1, 2, 3],
                        default=1, help="每一组训练样本使用几个难样本")
    parser.add_argument("--hard_encode_data_path", type=str,
                        default="", help="已处理好的难样本数据")
    parser.add_argument("--stage1_model_path", type=str,
                        default=None, help="正常训练时候的模型路径")

    args = vars(parser.parse_args())
    print("args:")
    print(args)

    global HARD_NEG_RANK, ROOT_DIR
    ROOT_DIR = "../../data/raw_data/"
    HARD_NEG_RANK = args["hard_neg_rank"]

    run_mode = args["mode"]
    MODEL_SAVE_DIR = ("../../data/models/{model_name}_lr{lr}_drop{drop}_pt_{pooling}_"
                      "ecom_{ecom}_video_{video}_fgm_{fgm}_lrdecay_{lr_decay}_hard_rank{hard_neg_rank}"
                      "_neg{n_neg}_{suffix}"
                      .format(model_name=args["model_name"],
                              sample_num=args["sample_num"],
                              lr=args["learning_rate"],
                              drop=args["dropout_rate"],
                              pooling=args["pooling_type"],
                              ecom=args["use_ecom"],
                              video=args["use_video"],
                              fgm=args["fgm"],
                              suffix=args["suffix"],
                              lr_decay=args["lr_decay"],
                              hard_neg_rank=args["hard_neg_rank"],
                              n_neg=args["use_n_neg"])
                      )
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    ############################################
    # 模型相关
    MODEL_PATH_DICT = {
        "robert": "../../pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12",
        "roformer_sim_ft": "../../pretrained_model/chinese_roformer-sim-char-ft_L-12_H-768_A-12",
        "simbert": "../../pretrained_model/chinese_simbert_L-12_H-768_A-12",
    }
    model_name = args["model_name"]
    assert model_name in MODEL_PATH_DICT, "INVALID model name 【{}】".format(model_name)
    MODEL_PATH = MODEL_PATH_DICT[model_name]
    config_path = f'{MODEL_PATH}/bert_config.json'
    checkpoint_path = f'{MODEL_PATH}/bert_model.ckpt'
    dict_path = f'{MODEL_PATH}/vocab.txt'

    pooling = args["pooling_type"]
    maxlen = 64
    dropout_rate = args["dropout_rate"]

    tokenizer = get_tokenizer(dict_path)
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model=args["base_model"],
        pooling=pooling,
        dropout_rate=0  # dropout_rate
    )

    # 载入无难样本的二元组训练的模型
    weight_name = [w for w in os.listdir(args["stage1_model_path"]) if w.endswith("weight")][0]
    print(f"load weight from 【{args['stage1_model_path']}/{weight_name}】")
    encoder.load_weights(f"{args['stage1_model_path']}/{weight_name}")

    ###########################################
    # 输入相关
    # 测试集
    with _open_(f"{ROOT_DIR}/dev_encode_roformer_tokenizer.pkl", "rb") as f:
        dev_encode = pickle.load(f)
    # 获取编码后的doc语料
    with _open_(f"{ROOT_DIR}/doc_encode_roformer_tokenizer.pkl", "rb") as f:
        corpus_encode = pickle.load(f)

    ############ 生成难样本
    hard_triple_encode = get_data_hard_neg(args, encoder, tokenizer, corpus_encode, args["hard_encode_data_path"])

    # 训练集
    train_generator = CustomGeneratorSupervisedForHardExampleMining(
        hard_triple_encode,
        args["train_batch_size"],
        n_hard=args["use_n_neg"]
    )

    ###############################################################################################
    # 模型训练及评估
    dev_generator = CustomGeneratorForPredict(dev_encode, 256)
    corpus_generator = CustomGeneratorForPredict(corpus_encode, 512)
    evaluator = Evaluator(dev_generator,
                          corpus_generator,
                          MODEL_SAVE_DIR,
                          base_model=args["base_model"])
    call_backs = [evaluator]

    # 几个难样本，一个则n_samle_per_pair=3，3个则n_samle_per_pair=5
    # simcse_loss_fn = partial(super_hard_supervise_loss, n_samle_per_pair=2 + args["use_n_neg"])
    simcse_loss_fn = super_hard_supervise_loss(3)
    # 设置衰减学习率
    if args["lr_decay"]:
        sample_count = len(train_generator)
        # 总共的步长
        total_steps = int(args["epoch"] * sample_count / args["train_batch_size"])
        # 预热步长
        warmup_epoch = 1
        warmup_steps = int(warmup_epoch * sample_count / args["train_batch_size"])
        # 学习率
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=args["learning_rate"],
                                                total_steps=total_steps,
                                                warmup_learning_rate=1e-7,
                                                warmup_steps=warmup_steps,
                                                hold_base_rate_steps=0,
                                                min_learn_rate=1e-6
                                                )
        call_backs.append(warm_up_lr)
        # decay_steps = len(train_generator) * args["epoch"]
        # print("decay_steps:", decay_steps)
        # lr_sch = tf.keras.experimental.CosineDecay(args["learning_rate"], decay_steps, alpha=0.1)
        encoder.compile(optimizer=Adam(learning_rate=args["learning_rate"]), loss=simcse_loss_fn)
    else:
        # AdamW = extend_with_weight_decay(Adam, 'AdamW')
        # opt = AdamW(learning_rate=args["learning_rate"], weight_decay_rate=0.01)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        #     opt,
        #     loss_scale='dynamic')
        opt = Adam(learning_rate=args["learning_rate"])
        encoder.compile(loss=simcse_loss_fn, optimizer=opt)
    # 启用对抗训练
    if args["fgm"]:
        adversarial_training(encoder, 'Embedding-Token', 0.5)
    encoder.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        callbacks=call_backs,
        epochs=args["epoch"]
    )

    # os.system("shutdown")