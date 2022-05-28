from __future__ import print_function
import os, pickle
import numpy as np
from tqdm import tqdm
import argparse

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

from bert4keras.snippets import open, _open_
from bert4keras.layers import Loss
from bert4keras.optimizers import Adam, extend_with_weight_decay
from utils import *


class UnilmGenerator(DataGenerator):
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
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids = []


class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 20  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


class UnilmEvaluator(keras.callbacks.Callback):
    def __init__(self, dev_generator=None, doc_generator=None,
                 model_save_dir=None, encoder=None):
        self.dev_generator = dev_generator
        self.doc_generator = doc_generator
        self.save_dir = model_save_dir
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        save_dir = "{0}/epoch{1}".format(self.save_dir, epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.encoder.save_weights("{0}/epoch{1}.weight".format(save_dir, epoch))

        dev_emb = l2_normalize(self.predict(self.encoder, self.dev_generator)[0])
        corpus_emb = l2_normalize(self.predict(self.encoder, self.doc_generator)[0])
        save_embedding(dev_emb, save_dir + "/query_embedding", 200001)
        save_embedding(corpus_emb, save_dir + "/doc_embedding", 1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimSCE model training.")
    parser.add_argument("--model_name", "-model", type=str,
                        choices=["robert", "roformer", "simbert"],
                        default="bert", help="模型名字")
    parser.add_argument("--train_batch_size", type=int,
                        default=64, help="训练batch_size")
    parser.add_argument("--epoch", default=3, type=int,
                        help="是否使用额外的数据")
    parser.add_argument("--fgm", action="store_true",
                        default=False, help="是否进行对抗训练")
    parser.add_argument("--learning_rate", "-lr", type=float,
                        default=1e-5, help="学习率")
    parser.add_argument("--use_ecom", action="store_true",
                        default=False, help="是否使用额外的电商数据")
    parser.add_argument("--use_video", action="store_true",
                        default=False, help="是否使用额外的视频搜索数据")
    parser.add_argument("--final_activation", type=str,
                        default="tanh", help="bert cls层后面的降维使用的激活函数")
    parser.add_argument("--suffix", default="",
                        type=str, help="保存文件的目录名后缀")

    args = vars(parser.parse_args())
    print("args:")
    print(args)

    global HARD_NEG_RANK, ROOT_DIR
    ROOT_DIR = "../../data/raw_data/"

    MODEL_SAVE_DIR = ("../../data/models/{model_name}_lr{lr}_"
                      "ecom_{ecom}_video_{video}_fgm_{fgm}_"
                      "act_{activation}_{suffix}"
                      .format(model_name=args["model_name"],
                              lr=args["learning_rate"],
                              ecom=args["use_ecom"],
                              video=args["use_video"],
                              fgm=args["fgm"],
                              suffix=args["suffix"],
                              activation=args["final_activation"]
                              )
                      )
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    ############################################
    # 模型相关
    MODEL_PATH_DICT = {
        "robert": "../../pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12",
        "roformer": "../../pretrained_model/chinese_roformer-sim-char-ft_L-12_H-768_A-12",
        "simbert": "../../pretrained_model/chinese_simbert_L-12_H-768_A-12",
    }
    model_name = args["model_name"]
    assert model_name in MODEL_PATH_DICT, "INVALID model name 【{}】".format(model_name)
    MODEL_PATH = MODEL_PATH_DICT[model_name]
    config_path = f'{MODEL_PATH}/bert_config.json'
    checkpoint_path = f'{MODEL_PATH}/bert_model.ckpt'
    dict_path = f'{MODEL_PATH}/vocab.txt'

    tokenizer = get_tokenizer(dict_path)

    # 建立加载模型
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model=args["model_name"],
        with_pool='linear',
        # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        # return_keras_model=False
        application='unilm'
    )

    # bert.model.outputs = [cls_output, mlm_output]
    print(bert.outputs)
    cls_output = bert.outputs[0]
    emb_output = keras.layers.Dense(128, activation=args["final_activation"])(cls_output)
    # 最后的编码器
    encoder = keras.models.Model(bert.inputs, emb_output)
    # seq2seq解码器
    seq2seq = keras.models.Model(bert.inputs, bert.outputs[1])
    # final outputs
    bert_outputs = [emb_output, bert.outputs[1]]

    outputs = TotalLoss([2])(bert.inputs + bert_outputs)
    model = keras.models.Model(bert.inputs, outputs)

    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    optimizer = AdamW(learning_rate=args["learning_rate"], weight_decay_rate=0.01)
    model.compile(optimizer=optimizer)
    model.summary()

    ###########################################
    # 输入相关
    # 测试集
    with _open_(f"{ROOT_DIR}/dev_encode_roformer_tokenizer.pkl", "rb") as f:
        dev_encode = pickle.load(f)
    # 获取编码后的doc语料
    with _open_(f"{ROOT_DIR}/doc_encode_roformer_tokenizer.pkl", "rb") as f:
        corpus_encode = pickle.load(f)

    # 训练集
    df_pair, df_pair_raw, df_pair_ecom, df_pair_video = get_paired_data(args)
    paired_encode = get_encoded_paird_from_paired(df_pair, tokenizer, maxlen=64)
    train_generator =  UnilmGenerator(paired_encode, args["train_batch_size"])

    del df_pair, df_pair_raw, df_pair_ecom, df_pair_video
    gc.collect()
    ###############################################################################################
    # 模型训练及评估
    dev_generator = CustomGeneratorForPredict(dev_encode, 256)
    corpus_generator = CustomGeneratorForPredict(corpus_encode, 512)

    evaluator = UnilmEvaluator(dev_generator,
                               corpus_generator,
                               MODEL_SAVE_DIR,
                               encoder)
    call_backs = [evaluator]

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator]
    )

    # os.system("shutdown")
