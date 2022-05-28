from __future__ import print_function
import os, pickle
import numpy as np
from tqdm import tqdm
import argparse
from collections import Counter
from bert4keras.snippets import open, _open_
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import AutoRegressiveDecoder
from utils import get_paired_data, save_embedding, l2_normalize, \
    gen_encode_data_for_predict, CustomGeneratorForPredict


def truncate(text, maxlen=128):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]


class SimBertGenerator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size=80, tokenizer=None, max_len=128):
        super(SimBertGenerator, self).__init__(data, batch_size)
        self.max_len = max_len
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.some_samples = []

    def __iter__(self, random):
        batch_token_ids, batch_segment_ids = [], []
        # 用来交换位置，使得每个batch中doc预测query和query预测doc的数量是一致的
        for is_end, texts in self.sample(random):
            qtext, dtext = texts
            # query - doc
            token_ids, segment_ids = self.tokenizer.encode(
                qtext, dtext, maxlen=self.max_len
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # doc - query
            token_ids, segment_ids = self.tokenizer.encode(
                dtext, qtext, maxlen=self.max_len
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


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


class Evaluate(keras.callbacks.Callback):
    def __init__(self, dev_generator=None, doc_generator=None, model_save_dir=None,
                 base_model="bert", encoder=None):
        self.dev_generator = dev_generator
        self.doc_generator = doc_generator
        self.save_dir = model_save_dir
        self.base_model = base_model
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        save_dir = "{0}/epoch{1}".format(self.save_dir, epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights("{0}/epoch{1}.weight".format(save_dir, epoch))

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
            # simbert预测值有两个cls_output和mlm_output,取第一个cls输出即可
            vec = model.predict(token_ids, verbose=False)[0]
            vecs.append(vec)
            indexes.extend(idx)

        indexes = np.array(indexes).reshape(-1, 1)

        return np.vstack(vecs), indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimSCE model training.")
    parser.add_argument("--model_name", "-model", type=str,
                        choices=["robert", "roformer_sim_ft", "simbert"],
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

    # parser.add_argument("--hard_neg_rank", type=int,
    #                     default=9, help="取召回的第几个作为难样本")
    # parser.add_argument("--use_n_neg", type=int, choices=[1, 2, 3],
    #                     default=1, help="每一组训练样本使用几个难样本")
    # parser.add_argument("--hard_encode_data_path", type=str,
    #                     default="", help="已处理好的难样本数据")
    # parser.add_argument("--stage1_model_path", type=str,
    #                     default=None, help="正常训练时候的模型路径")

    args = vars(parser.parse_args())
    print("args:")
    print(args)

    global HARD_NEG_RANK, ROOT_DIR
    ROOT_DIR = "../../data/raw_data/"

    MODEL_SAVE_DIR = ("../../data/models/{model_name}_lr{lr}_"
                      "ecom_{ecom}_video_{video}_fgm_{fgm}_"
                      "{suffix}"
                      .format(model_name=args["model_name"],
                              lr=args["learning_rate"],
                              ecom=args["use_ecom"],
                              video=args["use_video"],
                              fgm=args["fgm"],
                              suffix=args["suffix"]
                              )
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

    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    # 建立加载模型
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        with_pool='linear',
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        return_keras_model=False
    )

    # bert.model.outputs = [cls_output, mlm_output]
    cls_output = bert.model.outputs[0]
    emb_output = keras.layers.Dense(128, activation=args["final_activation"])(cls_output)
    # 最后的编码器
    encoder = keras.models.Model(bert.model.inputs, emb_output)
    # seq2seq解码器
    seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])
    # final outputs
    bert_outputs = [emb_output, bert.model.outputs[1]]

    outputs = TotalLoss([2, 3])(bert.model.inputs + bert_outputs)
    model = keras.models.Model(bert.model.inputs, outputs)
    
    
    # 载入无难样本的二元组训练的模型
    weight_name = f"{MODEL_SAVE_DIR}/epoch0/epoch0.weight" 
    #[w for w in os.listdir(args["stage1_model_path"]) if w.endswith("weight")][0]
    print(weight_name)
    # print(f"load weight from 【{args['stage1_model_path']}/{weight_name}】")
    model.load_weights(weight_name)


    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    optimizer = AdamW(learning_rate=args["learning_rate"], weight_decay_rate=0.001)
    model.compile(optimizer=optimizer)
    model.summary()
    
    

    #######################################################
    # 训练和预测数据
    # 测试集
    suffix = "simbert_tokenizer"
    if not os.path.exists(f"{ROOT_DIR}/doc_encode_{suffix}.pkl"):
        gen_encode_data_for_predict(tokenizer, suffix="simbert_tokenizer")

    with _open_(f"{ROOT_DIR}/doc_encode_{suffix}.pkl", "rb") as f:
        doc_encode = pickle.load(f)
    with _open_(f"{ROOT_DIR}/dev_encode_{suffix}.pkl", "rb") as f:
        dev_encode = pickle.load(f)

    dev_generator = CustomGeneratorForPredict(dev_encode, 256)
    doc_generator = CustomGeneratorForPredict(doc_encode, 512)

    # 训练集
    df_pair, df_pair_raw, df_pair_ecom, df_pair_video = get_paired_data(args)
    train_generator = SimBertGenerator(df_pair[["qtext_cut", "dtext_cut"]].values,
                                       args["train_batch_size"], tokenizer)
    evaluator = Evaluate(dev_generator=dev_generator, doc_generator=doc_generator,
                         model_save_dir=MODEL_SAVE_DIR,
                         base_model="bert", encoder=encoder)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=args["epoch"],
        callbacks=[evaluator]
    )

    # os.system("shutdown")
