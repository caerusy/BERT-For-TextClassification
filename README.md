- 情感分类： 采用 IMDB， SST-2, 以及 Yelp 数据集。
- 问题分类： 采用 TREC 和 Yahoo! Answers 数据集。
- 主题分类： 采用 AG's News，DBPedia 以及 CNews。

### SST-2

```
python3 run_SST2.py --max_seq_length=65 --num_train_epochs=5.0 --do_train --gpu_ids="1" --gradient_accumulation_steps=8 --print_step=100  # train and test
python3 run_SST2.py --max_seq_length=65   # test
```

| 模型                 | loss  | acc    | f1     |
| -------------------- | ----- | ------ | ------ |
| BertOrigin(base)     | 0.170 | 94.458 | 94.458 |
| BertCNN (5,6) (base) | 0.148 | 94.607 | 94.62  |
| BertATT (base)       | 0.162 | 94.211 | 94.22  |
| BertRCNN (base)      | 0.145 | 95.151 | 95.15  |
| BertCNNPlus (base)   | 0.160 | 94.508 | 94.51  |








