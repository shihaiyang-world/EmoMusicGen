
import torch
import collections
import torch.optim as optim
import torch.nn as nn
# from models import LSTM_SeqLabel,LSTM_SeqLabel_True
import argparse
import models.MECTGan as MECTGan


parser = argparse.ArgumentParser()

parser.add_argument('--status', default='train', choices=['train'])
parser.add_argument('--msg', default='_')
parser.add_argument('--train_clip', default=False, help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--see_convergence', default=False)
parser.add_argument('--see_param', default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--test_train', default=False)
parser.add_argument('--number_normalized', type=int, default=0,
                    choices=[0, 1, 2, 3], help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name', default='yj', choices=['lk', 'yj'])
parser.add_argument('--update_every', default=1, type=int)
parser.add_argument('--use_pytorch_dropout', type=int, default=0)

parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)

parser.add_argument('--word_min_freq', default=1, type=int)

# 状态，训练还是测试

# 是否存储训练好模型

# 存储目录

# 预测是是否加载训练好模型

# 加载目录

args = parser.parse_args()

if __name__ == '__main__':
    # 加载数据集

    # 加载模型
    model = MECTGan(input_size=1, hidden_size=1, output_size=1, num_layers=1)

    # 训练

    # 生成音乐

    print("Hello, World!", args.status)