import os
import json
import argparse
import tensorflow as tf
import time
import random
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
while logger.handlers:
    logger.handlers.pop()

from data import DataProcessor
from bert import modeling
from rank_model import Ranker


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.bert_config_path = args.bert_config_path
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.warmup_proportion = args.warmup_proportion
        self.max_seq_length = args.max_seq_length
        self.learning_rate = args.learning_rate
        self.bert_model_path = args.bert_model_path
        self.processor = DataProcessor(args.data_dir, n_neg=args.n_neg, rank_idx=args.rank_idx,
                                       max_len=args.max_seq_length, random_neg_idx=args.random_neg_idx)
        self.use_video = args.use_video
        self.label_smooth = args.label_smooth

    def create_model(self):
        model = Ranker(self.bert_config_path, is_training=self.args.is_training,
                       num_train_steps=self.num_train_steps, num_warmup_steps=self.num_warmup_steps,
                       learning_rate=self.learning_rate, label_smooth=self.label_smooth)
        return model

    def train(self):
        train_features = self.processor.get_train_features(self.use_video)
        random.shuffle(train_features)
        self.num_train_steps = int(len(train_features) // self.batch_size * self.num_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info(" Num examples = %d", len(train_features))
        tf.logging.info(" Batch size = %d", self.batch_size)
        tf.logging.info(" Num steps = %d", self.num_train_steps)

        num_batches = len(train_features) // self.batch_size
        self.model = self.create_model()
        tot_step = 0

        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       self.bert_model_path)
            tf.train.init_from_checkpoint(self.bert_model_path, assignment_map)
            # tf.logging.info("***** Trainable Variables *****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)
            sess.run(tf.variables_initializer(tf.global_variables()))

            starttime = time.time()
            for i in range(self.num_epochs):
                print("***** epoch-{} *******".format(i))
                if i > 0:
                    tf.logging.info("**** get train_examples ****")
                    random.shuffle(train_features)
                input_ids_lst, input_mask_lst, token_type_ids_lst, label_ids_lst = self.processor.get_inputs(
                    train_features)
                total_loss = 0.
                current_step = 0

                for j in range(num_batches):
                    start = j * self.batch_size
                    end = start + self.batch_size
                    batch_features = {"input_ids": input_ids_lst[start: end], "input_mask": input_mask_lst[start: end],
                                      "token_type_ids": token_type_ids_lst[start: end],
                                      "label_ids": label_ids_lst[start: end]}
                    loss, logits, prob = self.model.train(sess, batch_features)
                    total_loss += loss
                    avg_loss = total_loss / (current_step + 1)
                    if current_step % self.args.eval_steps == 0:
                        tf.logging.info("*****【%s】, training_step: %d 【%s】, loss: %f",
                                        datetime.now().strftime("%H:%M:%S"), current_step,
                                        str(round(100 * current_step / self.num_train_steps, 2)) + "%", avg_loss)
                    current_step += 1
                    tot_step += 1
                    # if current_step % self.args.save_steps == 0:
                    #     # 每一个epoch结束保存模型
                    #     tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                    #     ckpt_name_prefix = "models"
                    #     save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                    #     self.model.saver.save(sess, save_path, global_step=tot_step)

                # 每训练完一轮保存一次模型
                tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                ckpt_name_prefix = "models-epoch"
                save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                self.model.saver.save(sess, save_path, global_step=tot_step)
            tf.logging.info("total training time: %f", time.time() - starttime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="simbert",
                        help="The name of pretrained model.")
    parser.add_argument("--output_dir", type=str,
                        default="./rank_models/",
                        help="The path of checkpoint you want to save")
    parser.add_argument("--data_dir", type=str,
                        default="./",
                        help="The path of checkpoint you want to save")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--n_neg", type=int, default=1, help="每个正样本配几个负样本")
    parser.add_argument("--rank_idx", type=int, default=9,
                        help="doc负样本已按召回排序，rank_idx 表示 neg doc id的起始index，每隔10个取一个负样本")
    parser.add_argument("--use_video", action="store_true", default=False)
    parser.add_argument("--label_smooth", default=0.1, type=float, help="标签平滑值")
    parser.add_argument("--random_neg_idx", default=False, action="store_true", 
                        help="负样本是否进行随机采样")
    parser.add_argument("--suffix", default="", type=str, help="保存模型的后缀")
    parser.add_argument("--is_training", action="store_true", default=True)
    

    args = parser.parse_args()
    print(args)

    # 模型相关
    MODEL_PATH_DICT = {
        "robert": "../../pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12",
        "roformer_sim_ft": "../../pretrained_model/chinese_roformer-sim-char-ft_L-12_H-768_A-12",
        "simbert": "../../pretrained_model/chinese_simbert_L-12_H-768_A-12",
    }
    args.bert_model_path = MODEL_PATH_DICT[args.model_name] + "/bert_model.ckpt"
    args.bert_config_path = MODEL_PATH_DICT[args.model_name] + "/bert_config.json"
    args.bert_vocab_path = MODEL_PATH_DICT[args.model_name] + "/vocab.txt"

    args.output_dir = ("{outdir}/{model}_video_{video}_negRank{rank}_neg{n_neg}_"
                       "lr{lr}_sm{label_smooth}_randomNeg{random_neg}_{suffix}/"
                       .format(outdir=args.output_dir,
                               model=args.model_name,
                               video=args.use_video,
                               rank=args.rank_idx,
                               n_neg=args.n_neg,
                               lr=args.learning_rate,
                               suffix=args.suffix,
                               label_smooth=args.label_smooth,
                               random_neg=args.random_neg_idx)
                       )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer = Trainer(args)
    trainer.train()

    os.system("shutdown")
