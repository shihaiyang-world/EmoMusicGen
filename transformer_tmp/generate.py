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
parser.add_argument("--emo_tag", default=1, type=int)
args = parser.parse_args()

path_data_root = args.data_root
path_exp = 'exp/' + args.exp_name
path_gendir = 'exp/' + args.path_gendir
emotion_tag = args.emo_tag
epochs = 10
batch_size = 32
D_MODEL = 512
HEADS = 8
ENCODER_LAYER = 2
DECODER_LAYER = 2
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



def generate():

    # 加载模型pth
    # path
    path_ckpt = path_exp  # path to ckpt dir
    name = 'loss_high'
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    dataset = PEmoDataset(path_train_data, path_train_data_cls_idx)
    init = dataset.train_x[0][0:10, :]

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir 生成音乐存储地址
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []  # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128]
    is_training = False

    # init model 初始化模型 用来生成
    transformer_model = TransformerModel(D_MODEL, HEADS, ENCODER_LAYER, DECODER_LAYER, emb_sizes, n_class, is_training=False)
    transformer_model.cuda()
    transformer_model.eval()
    print('[*] load model from:', path_saved_ckpt)
    load = torch.load(path_saved_ckpt)
    transformer_model.load_state_dict(load, strict=False)

    # 生成音乐
    song_time_list = []
    words_len_list = []
    start_time = time.time()

    path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format(str(emotion_tag), utils.get_random_string(10)))

    # 生成音乐
    final_res, generated_key = transformer_model.generate_from_scratch(dictionary, emotion_tag)
    np.save(path_outfile + '.npy', final_res)
    utils.write_midi(final_res, path_outfile + '.mid', word2event)
    print('save to:', path_outfile + '.mid')
    song_time = time.time() - start_time
    word_len = len(final_res)
    print('song time:', song_time)
    print('word_len:', word_len)





if __name__ == '__main__':
    generate()