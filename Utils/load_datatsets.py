import os
import sys
import csv

from .Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors, convert_testfeatures_to_tensor


def read_tsv(filename):
    with open(filename, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def load_tsv_dataset(filename, set_type):
    """
    文件内数据格式: sentence  label
    """
    examples = []
    lines = read_tsv(filename)
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = line[1]
        label = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def load_testNoLabel(filename,set_type):
    """
    test没有label时的调用
    :param filename:
    :param set_type:
    :return:
    """
    examples = []
    lines = read_tsv(filename)
    for (i , line) in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = line[0]
        examples.append(InputExample(guid = guid, text_a=text_a,text_b=None,label='0'))
    return examples

def load_data(data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
    if format_type == 0:
        load_func = load_tsv_dataset

    if data_type == "train":
        train_file = os.path.join(data_dir, 'train.csv')
        examples = load_func(train_file, data_type)
    elif data_type == "dev":
        dev_file = os.path.join(data_dir, 'dev.csv')
        examples = load_func(dev_file, data_type)
    elif data_type == "test":
        test_file = os.path.join(data_dir, 'test.csv')
        examples = load_testNoLabel(test_file, data_type)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len

def load_testdata(data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
    if format_type == 0:
        load_func = load_tsv_dataset

    if data_type == "train":
        train_file = os.path.join(data_dir, 'train.csv')
        examples = load_func(train_file, data_type)
    elif data_type == "dev":
        dev_file = os.path.join(data_dir, 'dev.csv')
        examples = load_func(dev_file, data_type)
    elif data_type == "test":
        test_file = os.path.join(data_dir, 'test.csv')
        examples = load_testNoLabel(test_file, data_type)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)

    dataloader = convert_testfeatures_to_tensor(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len

