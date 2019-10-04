# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from Utils.utils import get_device
from Utils.load_datatsets import load_data, load_testdata

from BertOrigin import args
from train_evalute import train, evaluate, evaluate_save, save_testLabel

if __name__ =="__main__":
    label_list = ['0','1']
    output_dir = ".newsTF_output/"
    test_datadir = '/home/liuyang/TextClassification/dataset/compet/test.csv'

    label_list = ['0', '1']
    data_dir = "/home/liuyang/TextClassification/dataset/compet/"
    output_dir = ".newsTF_output/"
    cache_dir = ".newsTF_cache/"
    log_dir = ".newsTF_log/"

    model_times = "model_1/"  # 第几次保存的模型，主要是用来获取最佳结果

    # bert-base
    bert_vocab_file = "/home/liuyang/bert/bert-base-chinese-vocab.txt"
    bert_model_dir = "/home/liuyang/bert/bert/chinese_L-12_H-768_A-12"

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

    output_model_file = os.path.join(config.output_dir, config.save_name, WEIGHTS_NAME)
    output_config_file = os.path.join(config.output_dir, config.save_name, CONFIG_NAME)

    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    test_dataloader, _ = load_testdata(config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test", label_list)

    # 加载模型
    bert_config = BertConfig(output_config_file)

    if config.model_name == "BertOrigin":
        from BertOrigin.BertOrigin import BertOrigin
        model = BertOrigin(bert_config, num_labels=num_labels)
    elif config.model_name == "BertCNN":
        from BertCNN.BertCNN import BertCNN
        filter_sizes = [int(val) for val in config.filter_sizes.split()]
        model = BertCNN(bert_config, num_labels=num_labels,
                        n_filters=config.filter_num, filter_sizes=filter_sizes)
    elif config.model_name == "BertATT":
        from BertATT.BertATT import BertATT
        model = BertATT(bert_config, num_labels=num_labels)
    elif config.model_name == "BertRCNN":
        from BertRCNN.BertRCNN import BertRCNN
        model = BertRCNN(bert_config, num_labels, config.hidden_size, config.num_layers, config.bidirectional, config.dropout)
    elif config.model_name == "BertCNNPlus":
        from BertCNNPlus.BertCNNPlus import BertCNNPlus
        filter_sizes = [int(val) for val in config.filter_sizes.split()]
        model = BertCNNPlus(bert_config, num_labels=num_labels,
                            n_filters=config.filter_num, filter_sizes=filter_sizes)

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    print("--------------- Test -------------")
    savefile = '/home/liuyang/TextClassification/dataset/compet/label.csv'
    save_testLabel(model, test_dataloader, device, savefile)



