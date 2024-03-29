# coding=utf-8
from multi_main import main

if __name__ == "__main__":

    model_name = "BertHAN"
    data_dir = "/home/liuyang/TextClassification/dataset/cnews/"
    output_dir = ".cnews_output/"
    cache_dir = ".cnews_cache/"
    log_dir = ".cnews_log/"

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取多次最佳结果

    bert_vocab_file = "/home/liuyang/bert/bert-base-chinese-vocab.txt"  # 需改
    bert_model_dir = "/home/liuyang/bert/bert/chinese_L-12_H-768_A-12"

    from Processors.MultiNewsProcessor import NewsProcessor
    if model_name == "BertHAN":
        from BertHAN import args

    main(args.get_args(data_dir, output_dir, cache_dir,
                       bert_vocab_file, bert_model_dir, log_dir),
         model_times, NewsProcessor)
