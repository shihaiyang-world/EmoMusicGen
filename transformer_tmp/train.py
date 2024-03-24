'''
定一个最简单的Transformer，带入自己的模型，进行训练过程
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pickle
import numpy as np
from torch.utils.data import DataLoader
from data_processor import PEmoDataset
import os
from public_layer import network_paras
from models import TransformerModel
import time, datetime
import saver
import utils
import json

import argparse
parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument("--path_train_data", default='emopia', type=str)
parser.add_argument("--data_root", default='../co-representation/', type=str)
parser.add_argument("--load_dict", default="dictionary.pkl", type=str)
parser.add_argument("--exp_name", default='output' , type=str)
parser.add_argument("--path_gendir", default='midigen' , type=str)
parser.add_argument("--emo_tag", default=2, type=int)
parser.add_argument('--epoch', default=2, type=int)
parser.add_argument('--layers', default=4, type=int)
parser.add_argument('--batch', default=4, type=int)
parser.add_argument('--heads', default=8, type=int)
args = parser.parse_args()

path_data_root = args.data_root
path_exp = 'exp/' + args.exp_name
path_gendir = 'exp/' + args.path_gendir
emotion_tag = args.emo_tag
epochs = args.epoch
batch_size = args.batch
D_MODEL = 512
HEADS = args.heads
ENCODER_LAYER = args.layers
DECODER_LAYER = args.layers
learning_rate = 0.0001

path_train_data = os.path.join(path_data_root, args.path_train_data + '_data.npz')
path_dictionary =  os.path.join(path_data_root, args.load_dict)
path_train_idx = os.path.join(path_data_root, args.path_train_data + '_fn2idx_map.json')
path_train_data_cls_idx = os.path.join(path_data_root, args.path_train_data + '_idx.npz')

assert os.path.exists(path_train_data)
assert os.path.exists(path_dictionary)
assert os.path.exists(path_train_idx)



def prep_dataloader(batch_size, n_jobs=0):
    dataset = PEmoDataset(path_train_data, path_train_data_cls_idx)

    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader



def train():
    myseed = 42069
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    start_time = time.time()
    # 加载数据集
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_loader = prep_dataloader(batch_size)

    # create saver  保存模型，loss等
    saver_agent = saver.Saver(path_exp)

    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]  with key: [... , 25]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    # 几个类别的token数量，以及embedding的维度
    # [56,   127,   18,      4,    85,    18,       41,       5]
    # tempo  chord  barbeat  type  pitch  duration  velocity  emotion
    # [128,   256,   64,     32,   512,   128,      128,      128]
    emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128]

    # 定义模型
    transformer_model = TransformerModel(D_MODEL, HEADS, ENCODER_LAYER, DECODER_LAYER, emb_sizes, n_class)
    transformer_model.cuda()
    transformer_model.train()
    n_parameters = network_paras(transformer_model)
    print('n_parameters: {:,}'.format(n_parameters))

    optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs):

        num_batch = len(train_loader)
        epoch_loss = 0

        for i, (src_seq, tgt_seq, mask) in enumerate(train_loader):
            src_seq = src_seq.cuda()
            tgt_seq = tgt_seq.cuda()
            # 用于处理非定长序列的padding mask（非官方命名）；
            # 用于防止标签泄露的sequence mask（非官方命名）。
            # batch_mask 是padding mask，每个seq是1024，但是不是所有的都有数据，所以需要mask batch_mask==1的是有数据的，==0的是padding的
            batch_mask = mask.cuda()

            # out是经过softmax之后的概率分布，最大的值对应的index，然后再去字典中查询，就知道预测的词是什么了。
            losses = transformer_model(src_seq, tgt_seq, batch_mask)

            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6] + losses[7]) / 8

            # Update
            transformer_model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 50 == 0:
                print('epoch:{}/{}  batch:{}/{} | Loss: {:06f} \r'.format((epoch), epochs, i, num_batch, loss))

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = epoch_loss / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))

    # save model, with policy  保存模型
    saver_agent.save_model(transformer_model, name='loss_high')
    print('Train Finished, Model saved.')





if __name__ == '__main__':
    train()
