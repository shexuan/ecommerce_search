#! -*- coding: utf-8 -*-

import sys, os
# 使用混合精度训练
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'
# os.environ['TF_KERAS'] = "1"

from utils import *
from glob import glob
import argparse
import pandas as pd
import numpy as np

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
import gc
import time
import random
from tqdm import tqdm
# import jieba
# jieba.initialize()


    

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
                        default=64, help="训练batch_size")
    parser.add_argument("--dropout_rate", "-dropout", type=float,
                        default=0.3, help="dropout 概率")
    parser.add_argument("--epoch", default=3, type=int,
                        help="是否使用额外的数据")
    parser.add_argument("--mode", type=str, 
                        choices=["debug","train","predict"],
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

    args = vars(parser.parse_args())
    print("args:")
    print(args)
    
    run_mode = args["mode"]
    MODEL_SAVE_DIR = ("../../data/models/{model_name}_sample_10w_{sample_num}_lr{lr}_drop{drop}_pt_{pooling}_"
                      "ecom_{ecom}_video_{video}_fgm_{fgm}_lrdecay_{lr_decay}_aug{aug}{suffix}/"
                      .format(model_name=args["model_name"], 
                              sample_num=args["sample_num"],
                              lr=args["learning_rate"],
                              drop=args["dropout_rate"],
                              pooling=args["pooling_type"],
                              ecom=args["use_ecom"],
                              video=args["use_video"],
                              fgm=args["fgm"],
                              suffix="_"+args["suffix"],
                              aug=args["aug"],
                              lr_decay=args["lr_decay"])
                     )
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    
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
    
    ###########################################
    # 输入相关
    df_pair, df_pair_raw, _, _ = get_paired_data(args)
    pair_dict = dict(df_pair_raw[["qid","did"]].values)

    # 全部语料
    df_all = pd.read_pickle(f"{ROOT_DIR}/docs_cut_rm_stop_words.pkl")[["id", "text_cut", "source"]]
    
    # 获取编码后的语料
    corpus_encode = get_encoded_data_list_single(df_all.query("source=='doc'"), 
                                                 tokenizer, pair_dict, mode=run_mode)
    # 这里query只取100个作为验证集
    # query_encode = get_encoded_data_list_single(df_all.query("source=='query'"), 
    #                                             tokenizer, pair_dict, mode=run_mode)
    dev_encode = get_encoded_data_list_single(df_all.query("source=='dev'"),
                                              tokenizer, pair_dict, mode=run_mode)
    
    if args["aug"]=="false":
        paired_encode = get_encoded_paird_from_paired(df_pair, tokenizer, maxlen=64)

        # train unpair 语料
        # df_unpaired = pd.read_pickle(f"{ROOT_DIR}/unpaired_corpus.pkl")[["id", "text_cut", "source"]]
        # unpaired_encode = get_encoded_paired_from_unpaired(df_unpaired, tokenizer, 
        #                                                    sample_num=args["sample_num"], maxlen=64)
        # paired_encode = paired_encode + unpaired_encode
        train_generator = CustomGeneratorSupervised(paired_encode, args["train_batch_size"])
        # del df_unpaired
    else:
        raw_pair_data = get_raw_text_paird_from_paired(df_pair)
        train_generator = CustomAugmentGeneratorSupervised(raw_pair_data, batch_size=args["train_batch_size"], 
                                                           tokenizer=tokenizer, dropout_rate=args["dropout_rate"], 
                                                           sample_method=args["aug"], max_len=64)
    
    del df_pair,  df_all, df_pair_raw
    gc.collect()
    
    ###############################################################################################
    # 模型训练及评估
    dev_generator = CustomGeneratorForPredict(dev_encode, 256)
    corpus_generator = CustomGeneratorForPredict(corpus_encode, 256)
    evaluator = Evaluator(dev_generator,
                          corpus_generator,
                          MODEL_SAVE_DIR,
                          base_model=args["base_model"])
    call_backs = [evaluator]
    
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
                                                min_learn_rate = 1e-6
                                                )
        call_backs.append(warm_up_lr)
        # decay_steps = len(train_generator) * args["epoch"]
        # print("decay_steps:", decay_steps)
        # lr_sch = tf.keras.experimental.CosineDecay(args["learning_rate"], decay_steps, alpha=0.1)
        encoder.compile(optimizer=Adam(learning_rate=args["learning_rate"]), loss=simcse_loss)
    else:
        AdamW = extend_with_weight_decay(Adam, 'AdamW')
        opt = AdamW(learning_rate=args["learning_rate"], weight_decay_rate=0.01)
        # opt = Adam(learning_rate=args["learning_rate"])
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            opt,
            loss_scale='dynamic')
        encoder.compile(loss=simcse_loss, optimizer=opt)
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
        