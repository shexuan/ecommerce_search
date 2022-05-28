import json
import os
import random
import collections
import tensorflow as tf
from tqdm import tqdm
import pickle

# from bert import tokenization


class InputExample(object):
    def __init__(self, a_token_ids, b_token_ids, label=None):
        # self.qid = qid  # qid##docid
        self.a_token_ids = a_token_ids
        self.b_toke_ids = b_token_ids
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, token_type_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def truncate_seq_pair(a_ids, b_ids, max_seq_length):
    while True:
        total_length = len(a_ids) + len(b_ids)
        if total_length <= max_seq_length:
            break
        if len(a_ids) > len(b_ids):
            # 索引-1是[SEP]，所以要删除-2
            del a_ids[-2]
        else:
            del b_ids[-2]


def convert_single_example(example: InputExample, max_len=128):
    """生成负训练样本
    """
    a_token_ids = example.a_token_ids
    b_token_ids = example.b_toke_ids
    # account for [CLS], [SEP], [SEP], max_len-3
    truncate_seq_pair(a_token_ids, b_token_ids, max_len - 3)

    input_ids = a_token_ids + b_token_ids
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(a_token_ids) + [1] * len(b_token_ids)

    # padding
    while len(input_ids) < max_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len

    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids,
                            label_id=example.label, is_real_example=True)

    return feature


class DataProcessor(object):
    """
    精排阶段的数据
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

    def __init__(self, input_dir, n_neg=1, rank_idx=9, max_len=128, random_neg_idx=False):
        """
        n_neg: 每个正样本配几个负样本
        rank_idx: neg doc id 的起始index，每隔10个取一个负样本
        """
        self.max_len = max_len
        self.n_neg = n_neg
        self.rank_idx = rank_idx
        self.random_neg_idx = random_neg_idx

        # query和doc的单个句子的token，仅句子token_ids，不包含segment_ids和mask_ids
        self.doc_token_ids = self.load_pickle(f"{input_dir}/doc_encode_rank.pkl")
        self.raw_token_ids = self.load_pickle(f"{input_dir}/raw_train_query_encode_rank.pkl")
        self.ecom_token_ids = self.load_pickle(f"{input_dir}/ecom_train_query_encode_rank.pkl")
        self.video_token_ids = self.load_pickle(f"{input_dir}/video_train_query_encode_rank.pkl")

        # query和doc的样本对token，包含句子token_ids和encode_token_ids
        self.raw_pos_pair_encode = self.load_pickle(f"{input_dir}/raw_pos_pair_encode_rank.pkl")
        self.ecom_pos_pair_encode = self.load_pickle(f"{input_dir}/ecom_pos_pair_encode_rank.pkl")
        self.video_pos_pair_encode = self.load_pickle(f"{input_dir}/video_pos_pair_encode_rank.pkl")

        # query 的doc候选负样本id
        self.raw_doc_candidate = self.load_pickle(f"{input_dir}/raw_neg_doc_candidates.pkl")
        self.ecom_doc_candidate = self.load_pickle(f"{input_dir}/ecom_neg_doc_candidates.pkl")
        self.video_doc_candidate = self.load_pickle(f"{input_dir}/video_neg_doc_candidates.pkl")

    def get_pos_train_features(self, data):
        """获取正样本对
        data: pos pair encode data, [[token_ids, segment_ids, masks], ...]
        """
        features = []
        tf.logging.info("生成正样本数据 >>> >>>")
        for input_ids, segment_ids, mask_ids in tqdm(data):
            while len(input_ids) < self.max_len:
                input_ids.append(0)
                segment_ids.append(0)
                mask_ids.append(0)
            features.append(
                InputFeatures(input_ids=input_ids, input_mask=mask_ids, token_type_ids=segment_ids,
                              label_id=1, is_real_example=True)
            )
        return features

    def load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_neg_train_features(self, query_token_ids, candidate_data):
        """ 提取负样本
        query_token_ids: query的编码数据
        candidate_date: query对应的doc负样本候选数据
        """
        features = []
        tf.logging.info("生成负样本数据 >>> >>>")
        for qid, dids in tqdm(candidate_data):
            # 非随机生成负样本
            if not self.random_neg_idx:
                for i in range(self.n_neg):
                    candidate_doc_id = dids[self.rank_idx + i * 10]
                    example = InputExample(query_token_ids[qid][1], self.doc_token_ids[candidate_doc_id][1], 0)
                    features.append(convert_single_example(example, self.max_len))
            # 随机生成负样本
            else:
                neg_indexes = random.sample(range(self.rank_idx, len(dids)), self.n_neg)
                for i in neg_indexes:
                    candidate_doc_id = dids[i]
                    example = InputExample(query_token_ids[qid][1], self.doc_token_ids[candidate_doc_id][1], 0)
                    features.append(convert_single_example(example, self.max_len))

        return features

    def get_train_features(self, use_video=True):
        """获取正、负样本features"""
        # 正样本
        raw_pos_features = self.get_pos_train_features(self.raw_pos_pair_encode)
        ecom_pos_features = self.get_pos_train_features(self.ecom_pos_pair_encode)
        video_pos_features = []
        if use_video:
            video_pos_features = self.get_pos_train_features(self.video_pos_pair_encode)

        # 负样本
        raw_neg_features = self.get_neg_train_features(self.raw_token_ids, self.raw_doc_candidate)
        ecom_neg_features = self.get_neg_train_features(self.ecom_token_ids, self.ecom_doc_candidate)
        video_neg_features = []
        if use_video:
            video_neg_features = self.get_neg_train_features(self.video_token_ids, self.video_doc_candidate)

        features = raw_pos_features + ecom_pos_features + video_pos_features + \
                   raw_neg_features + ecom_neg_features + video_neg_features

        return features

    def get_inputs(self, features):
        input_ids_lst = []
        input_mask_lst = []
        token_type_ids_lst = []
        label_ids_lst = []
        for feature in features:
            input_ids_lst.append(feature.input_ids)
            input_mask_lst.append(feature.input_mask)
            token_type_ids_lst.append(feature.token_type_ids)
            label_ids_lst.append(feature.label_id)
        # print("input_ids_lst: ", input_ids_lst[0])
        # print("input_mask_lst: ", input_mask_lst[0])
        # print("token_type_ids_lst: ", token_type_ids_lst[0])
        # print("label_ids_lst: ", label_ids_lst[0])
        return input_ids_lst, input_mask_lst, token_type_ids_lst, label_ids_lst
