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
    transformer_model.load_state_dict(torch.load(path_saved_ckpt))

    # 生成音乐
    song_time_list = []
    words_len_list = []
    start_time = time.time()

    path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format(str(emotion_tag), utils.get_random_string(10)))
    # res, _ = transformer_model.generate_from_scratch(dictionary, emotion_tag)

    target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]

    init = np.array([
        target_emotion,  # emotion
        [0, 0, 1, 2, 0, 0, 0, 0]  # bar
    ])
    init_t = torch.from_numpy(init).long().cuda()
    inp = init_t.unsqueeze(0)

    classes = word2event.keys()
    def print_word_cp(cp):
        result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
        for r in result:
            print('{:15s}'.format(str(r)), end=' | ')
        print('')

    cnt_bar = 0

    # 采样类型
    for step in range(1, 1000):
        # 采样类型
        memory = transformer_model.encode(inp)
        y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, state = transformer_model.decode_and_output(inp, memory)

        y_tempo = y_tempo[:, -1, :].unsqueeze(0)
        y_chord = y_chord[:, -1, :].unsqueeze(0)
        y_type = y_type[:, -1, :].unsqueeze(0)
        y_barbeat = y_barbeat[:, -1, :].unsqueeze(0)
        y_pitch = y_pitch[:, -1, :].unsqueeze(0)
        y_duration = y_duration[:, -1, :].unsqueeze(0)
        y_velocity = y_velocity[:, -1, :].unsqueeze(0)
        y_emotion = y_emotion[:, -1, :].unsqueeze(0)
        y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion = nn.Softmax(dim=-1)(y_tempo), nn.Softmax(dim=-1)(y_chord), nn.Softmax(dim=-1)(y_type), nn.Softmax(dim=-1)(
            y_barbeat), nn.Softmax(dim=-1)(y_pitch), nn.Softmax(dim=-1)(y_duration), nn.Softmax(dim=-1)(
            y_velocity), nn.Softmax(dim=-1)(y_emotion)


        cur_word_type = utils.sampling(y_type, p=0.90, is_training=is_training)
        # 温度采样  softmax   temperature越大，越随机(超出原来的概率分布范围)，越小，越确定(接近原来的概率分布范围)
        cur_word_tempo = utils.sampling(y_tempo, t=0.9, p=0.9, is_training=is_training)
        cur_word_barbeat = utils.sampling(y_barbeat, t=0.9, is_training=is_training)
        cur_word_chord = utils.sampling(y_chord, p=0.1, is_training=is_training)
        cur_word_pitch = utils.sampling(y_pitch, p=0.1, is_training=is_training)
        cur_word_duration = utils.sampling(y_duration, t=0.9, p=0.9, is_training=is_training)
        cur_word_velocity = utils.sampling(y_velocity, t=0.9, is_training=is_training)

        # cur_word_emotion =utils.sampling(y_emotion, t=1, is_training=is_training)
        cur_word_emotion = 0
        next_arr = np.array([
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_type,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
            cur_word_emotion
        ])

        print('bar:', cnt_bar, end='  ==')
        print_word_cp(next_arr)
        if word2event['type'][next_arr[3]] == 'EOS':
            continue

        if word2event['bar-beat'][next_arr[2]] == 'Bar':
            cnt_bar += 1

        # cuda
        arr = torch.from_numpy(next_arr).long().cuda()
        arr = arr.unsqueeze(0).unsqueeze(0)
        inp = torch.cat(
            [inp, arr],
            dim=1
        )

        #
        # if res is None:
        #     return
        #
        # inp 转numpy
    inp = inp.squeeze().cpu().detach().numpy()
    np.save(path_outfile + '.npy', inp)
    utils.write_midi(inp, path_outfile + '.mid', word2event)
    print('save to:', path_outfile + '.mid')
        #
    song_time = time.time() - start_time
    word_len = len(inp)
    print('song time:', song_time)
    print('word_len:', word_len)





if __name__ == '__main__':
    generate()