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
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
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
HARD_NEG_RANK = 9


# NOTE: embedding中的doc token需要去除一个起始的[CLS]标志

def encode_sentence(texts, tokenizer, is_doc):
    """编码句子
    is_doc: 是否是doc的句子，对于doc句子编码后需要去除开头的[CLS]

    返回：句子idx和其token_ids，[[idx0, token_ids], [idx1, token_ids], ...]
    """
    data = []
    for idx, sent in enumerate(texts):
        token_ids = tokenizer.encode(" ".join(sent))[0]
        if is_doc:
            # 对于doc句子编码后需要去除开头的[CLS]
            token_ids = token_ids[1:]
        data.append([idx, token_ids])

    return data


def save_pickle(data, outpath):
    with _open_(outpath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_path):
    with _open_(input_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_pos_pair_encode(df, tokenizer, max_len=128):
    """对正样本对进行编码
    df: 包含正样本对的dataframe，列名 ["qid", "did", "qtext_cut", "dtext_cut"]
    tokenizer: 编码
    """
    data = []
    texts = df[["qid", "did", "qtext_cut", "dtext_cut"]].sort_values(by="qid").values
    for qid, did, qtext, dtext in tqdm(texts):
        token_ids, segment_ids = tokenizer.encode(" ".join(qtext), " ".join(dtext), maxlen=max_len)
        masks = [1] * len(token_ids)
        data.append([token_ids, segment_ids, masks])
    return data


def get_data_hard_neg_helper(data_generator, model, doc_emb, doc_encode_data, paired_encode_data):
    """提取原始数据的难样本
    data_generator, query token数据生成器；
    model：用来预测的模型；
    doc_emb: 语料库embedding；
    doc_encode_data: 编码好的doc语料tokens，[[d1_token_ids, id1], [d2_token_ids, id2], ...];
    paired_encode_data: 编码好的pair tokens，[[q1_token_ids, d1_token_ids, id1], [q2_token_ids, d2_token_ids, id2], ...];

    返回数据: [[query_idx0, [neg_doc_id1,neg_doc_id2,neg_doc_id3,...]], ...] , query_idx从0开始，neg_doc_idx从1开始
    """
    query_emb = l2_normalize(Evaluator.predict(model, data_generator)[0])
    doc_idx = np.arange(len(doc_emb))
    recall_doc_arr = get_hard_examples(query_emb, doc_emb, doc_idx, nlist=100,
                                       topK=RECALL_TOPK, use_gpu=False, emb_size=128)
    print("recall_doc_arr max:", recall_doc_arr.max())
    print("recall_doc_arr min:", recall_doc_arr.min())

    data = []
    for idx, doc_ids in tqdm(enumerate(recall_doc_arr)):
        true_doc_encode = paired_encode_data[idx][1]
        sample_doc_ids = []
        for doc_id in doc_ids:
            if doc_encode_data[doc_id][0] != true_doc_encode:
                sample_doc_ids.append(doc_id)
        # 一个query配29个难负样本待使用
        data.append([idx, sample_doc_ids[:29]])

    return data

def get_data_for_rank(args, encoder, recall_tokenizer, rank_tokenizer, outdir="./"):
    """生成精排阶段的数据
    一共包括6个数据：
        query正样本对，已编码好 [[token_ids, segment_ids, masks], ...]：
            - raw_pos_pair_encode_rank.pkl
            - ecom_pos_pair_encode_rank.pkl
            - video_pos_pair_encode_rank.pkl
        query对应的候选doc负样本，[[query_idx0, [neg_doc_id1,neg_doc_id2,neg_doc_id3,...]], ...] ：
            - raw_neg_doc_candidates.pkl
            - ecom_neg_doc_candidates.pkl
            - video_neg_doc_candidates.pkl
        query和doc编码数据，[[idx0, token_ids], [idx1, token_ids], ...], 其中query tokens包含[CLS]和[SEP]，而doc不包含开头的[CLS]
            - raw_train_query_encode_rank.pkl
            - ecom_train_query_encode_rank.pkl
            - video_train_query_encode_rank.pkl
            - dev_query_encode_rank.pkl
            - doc_encode_rank.pkl
    """
    ROOT_DIR = "../../data/raw_data/"
    # 全部语料
    df_all = pd.read_pickle(f"{ROOT_DIR}/docs_cut_rm_stop_words.pkl")[["id", "text_cut", "source"]]
    # 获取编码后的语料
    corpus_encode_recall = get_encoded_data_list_single(df_all.query("source=='doc'"),
                                                        recall_tokenizer, {}, mode="predict")
    df_pair, df_pair_raw, df_pair_ecom, df_pair_video = get_paired_data(args)

    del df_pair
    gc.collect()

    ##################################
    # 精排部分 对doc和query分别编码 [[idx0, token_ids], [idx1, token_ids], ...]
    # 其中query包含起始的[CLS]和结尾的[SEP]
    # doc仅包含结尾的[SEP]
    raw_train_query_encode_rank = encode_sentence(df_pair_raw["qtext_cut"].values, rank_tokenizer, False)
    ecom_train_query_encode_rank = encode_sentence(df_pair_ecom["qtext_cut"].values, rank_tokenizer, False)
    video_train_query_encode_rank = encode_sentence(df_pair_video["qtext_cut"].values, rank_tokenizer, False)
    dev_query_encode_rank = encode_sentence(df_all.query("source=='dev'")["text_cut"].values, rank_tokenizer, False)
    corpus_encode_rank = encode_sentence(df_all.query("source=='doc'")["text_cut"].values, rank_tokenizer, True)

    print("精排部分 对doc和query分别编码")
    save_pickle(raw_train_query_encode_rank, f"{outdir}/raw_train_query_encode_rank.pkl")
    save_pickle(ecom_train_query_encode_rank, f"{outdir}/ecom_train_query_encode_rank.pkl")
    save_pickle(video_train_query_encode_rank, f"{outdir}/video_train_query_encode_rank.pkl")
    save_pickle(dev_query_encode_rank, f"{outdir}/dev_query_encode_rank.pkl")
    save_pickle(corpus_encode_rank, f"{outdir}/doc_encode_rank.pkl")

    del df_all, raw_train_query_encode_rank, ecom_train_query_encode_rank,
    del video_train_query_encode_rank, dev_query_encode_rank, corpus_encode_rank
    gc.collect()

    ############################
    print("精排模型正样本对编码")
    # 精排模型正样本对编码，[[token_ids, segment_ids, masks], ...]
    rank_raw_pos_pair_encode = get_pos_pair_encode(df_pair_raw, rank_tokenizer, max_len=128)
    rank_ecom_pos_pair_encode = get_pos_pair_encode(df_pair_ecom, rank_tokenizer, max_len=128)
    rank_video_pos_pair_encode = get_pos_pair_encode(df_pair_video, rank_tokenizer, max_len=128)

    save_pickle(rank_raw_pos_pair_encode, f"{outdir}/raw_pos_pair_encode_rank.pkl")
    save_pickle(rank_ecom_pos_pair_encode, f"{outdir}/ecom_pos_pair_encode_rank.pkl")
    save_pickle(rank_video_pos_pair_encode, f"{outdir}/video_pos_pair_encode_rank.pkl")

    del rank_raw_pos_pair_encode, rank_ecom_pos_pair_encode, rank_video_pos_pair_encode
    gc.collect()

    ###############################
    print("利用召回模型提取精排模型负doc样本候选ID")
    # 利用召回模型提取精排模型负doc样本候选ID
    # 编码
    data_pair_raw_encode_recall = get_encoded_paird_from_paired(df_pair_raw, recall_tokenizer, maxlen=64)
    data_pair_ecom_encode_recall = get_encoded_paird_from_paired(df_pair_ecom, recall_tokenizer, maxlen=64)
    data_pair_video_encode_recall = get_encoded_paird_from_paired(df_pair_video, recall_tokenizer, maxlen=64)

    # query sentence encode，用于预测，提取难样本
    recall_raw_query_encode = [[pair[0], 1] for pair in data_pair_raw_encode_recall]
    recall_ecom_query_encode = [[pair[0], 1] for pair in data_pair_ecom_encode_recall]
    recall_video_query_encode = [[pair[0], 1] for pair in data_pair_video_encode_recall]

    raw_query_generator = CustomGeneratorForPredict(recall_raw_query_encode, 512)
    ecom_query_generator = CustomGeneratorForPredict(recall_ecom_query_encode, 512)
    video_query_generator = CustomGeneratorForPredict(recall_video_query_encode, 512)

    print("载入已提前训练好的doc_embedding, 用于召回难样本")
    # 载入已提前训练好的doc_embedding, 用于召回难样本
    _, doc_emb = load_embedding(args["recall_model_path"] + '/doc_embedding')
    # 提取精排训练所用的负样本doc对，一个query对应多个负doc
    # 返回数据 [[query_idx0, [neg_doc_id1,neg_doc_id2,neg_doc_id3,...]], ...]
    raw_recall_doc_idx_topk = get_data_hard_neg_helper(raw_query_generator, encoder, doc_emb,
                                                       corpus_encode_recall, data_pair_raw_encode_recall)
    ecom_recall_doc_idx_topk = get_data_hard_neg_helper(ecom_query_generator, encoder, doc_emb,
                                                        corpus_encode_recall, data_pair_ecom_encode_recall)
    video_recall_doc_idx_topk = get_data_hard_neg_helper(video_query_generator, encoder, doc_emb,
                                                         corpus_encode_recall, data_pair_video_encode_recall)
    save_pickle(raw_recall_doc_idx_topk, f"{outdir}/raw_neg_doc_candidates.pkl")
    save_pickle(ecom_recall_doc_idx_topk, f"{outdir}/ecom_neg_doc_candidates.pkl")
    save_pickle(video_recall_doc_idx_topk, f"{outdir}/video_neg_doc_candidates.pkl")
    print("提取精排训练所用的负样本doc对，一个query对应多个负doc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimSCE model training.")
    parser.add_argument("--recall_model", type=str,
                        choices=["bert", "robert", "robert_large", "roformer", "recall",
                                 "roformer_sim", "roformer_sim_ft", "simbert"],
                        default="roformer_sim_ft", help="召回模型名字")
    parser.add_argument("--rank_model", type=str,
                        choices=["bert", "robert", "robert_large", "roformer", "recall",
                                 "roformer_sim", "roformer_sim_ft", "simbert"],
                        default="simbert", help="精排模型名字")
    parser.add_argument("--recall_base_model", type=str,
                        choices=["bert", "roformer"],
                        default="bert", help="模型名字")
    parser.add_argument("--pooling_type", "-pt", type=str,
                        choices=['first-last-avg', 'last-avg', 'cls', 'pooler'],
                        default="cls", help="模型最后一层输出形式")
    parser.add_argument("--suffix", default="",
                        type=str, help="保存文件的目录名后缀")
    parser.add_argument("--use_ecom", action="store_true",
                        default=True, help="是否使用额外的电商数据")
    parser.add_argument("--use_video", action="store_true",
                        default=True, help="是否使用额外的视频搜索数据")
    parser.add_argument("--eval", action="store_true",
                        default=False, help="是否使用额外的视频搜索数据")
    parser.add_argument("--recall_model_path", type=str,
                        default="", help="召回模型目录")
    parser.add_argument("--outdir", type=str,
                        default="", help="输出目录")

    args = vars(parser.parse_args())
    print("args:")
    print(args)

    # global ROOT_DIR
    # ROOT_DIR = "../../data/raw_data/"

    ############################################
    # 模型相关
    MODEL_PATH_DICT = {
        "bert": "../../pretrained_model/chinese_bert_wwm_ext_L-12_H-768_A-12",
        "robert": "../../pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12",
        "robert_large": "../../pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16",
        "roformer_sim_ft": "../../pretrained_model/chinese_roformer-sim-char-ft_L-12_H-768_A-12",
        "simbert": "../../pretrained_model/chinese_simbert_L-12_H-768_A-12",
        "simcse": "../../pretrained_model/chinese_simcse_roberta_wwm_ext"
    }
    model_name = args["recall_model"]
    assert model_name in MODEL_PATH_DICT, "INVALID model name 【{}】".format(model_name)
    MODEL_PATH = MODEL_PATH_DICT[model_name]
    config_path = f'{MODEL_PATH}/bert_config.json'
    checkpoint_path = f'{MODEL_PATH}/bert_model.ckpt'
    dict_path = f'{MODEL_PATH}/vocab.txt'

    pooling = args["pooling_type"]
    maxlen = 64

    recall_tokenizer = get_tokenizer(dict_path)
    rank_tokenizer = get_tokenizer(MODEL_PATH_DICT[args["rank_model"]] + "/vocab.txt")
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model=args["recall_base_model"],
        pooling=pooling,
        dropout_rate=0  # dropout_rate
    )

    # 载入无难样本的二元组训练的模型
    weight_name = [w for w in os.listdir(args["recall_model_path"]) if w.endswith("weight")][0]
    print(f"load weight from 【{args['recall_model_path']}/{weight_name}】")
    encoder.load_weights(f"{args['recall_model_path']}/{weight_name}")

    if not os.path.exists(args["outdir"]):
        os.makedirs(args["outdir"])

    get_data_for_rank(args, encoder, recall_tokenizer, rank_tokenizer, outdir=args["outdir"])
