'''
定一个最简单的Transformer，带入自己的模型，进行训练过程
'''

import torch
import torch.nn as nn
from public_layer import PositionalEncoding, MultiHeadAttention, network_paras
import numpy as np
import math
import utils
import torch.nn.functional as F
from fast_transformers.builders import RecurrentDecoderBuilder as RecurrentDecoderBuilder_local
from fast_transformers.builders import TransformerEncoderBuilder as TransformerEncoderBuilder_local
from fast_transformers.builders import TransformerDecoderBuilder as TransformerDecoderBuilder_local
from fast_transformers.masking import TriangularCausalMask as TriangularCausalMask_local
from fast_transformers.masking import LengthMask as LengthMask_local
from fast_transformers.masking import BaseMask as BaseMask_local


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 对model的根号倍数进行缩放
        # 乘以 根号 d_model 其实就是起到了一个标准化的作用
        return self.lut(x) * math.sqrt(self.d_model)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, emb_sizes, n_class, dropout=0.1, d_ff=2048, is_training=True):
        super(TransformerModel, self).__init__()
        self.n_class = n_class
        self.emb_sizes = emb_sizes
        self.d_model = d_model
        self.is_training = is_training
        self.n_head = nhead
        # d_ff 最后一层位置前馈网络中内层的维数
        self.d_ff = d_ff
        self.e_layer = num_encoder_layers
        self.d_layer = num_decoder_layers

        # encoder
        self.get_encoder('encoder')

        self.word_emb_tempo = Embeddings(self.n_class[0], self.emb_sizes[0])
        self.word_emb_chord = Embeddings(self.n_class[1], self.emb_sizes[1])
        self.word_emb_barbeat = Embeddings(self.n_class[2], self.emb_sizes[2])
        self.word_emb_type = Embeddings(self.n_class[3], self.emb_sizes[3])
        self.word_emb_pitch = Embeddings(self.n_class[4], self.emb_sizes[4])
        self.word_emb_duration = Embeddings(self.n_class[5], self.emb_sizes[5])
        self.word_emb_velocity = Embeddings(self.n_class[6], self.emb_sizes[6])
        self.word_emb_emotion = Embeddings(self.n_class[7], self.emb_sizes[7])

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        self.linear = nn.Linear(np.sum(emb_sizes), d_model)
        self.posAtten = PositionalEncoding(d_model)

        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        # individual output
        self.proj_tempo = nn.Linear(self.d_model, self.n_class[0])
        self.proj_chord = nn.Linear(self.d_model, self.n_class[1])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_class[2])
        self.proj_type = nn.Linear(self.d_model, self.n_class[3])
        self.proj_pitch = nn.Linear(self.d_model, self.n_class[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_class[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_class[6])
        self.proj_emotion = nn.Linear(self.d_model, self.n_class[7])

    def encode(self, src, src_mask=None):
        src_embedded = self.common_embedding(src)
        if self.is_training:
            # mask 邻接矩阵  句子中词是否mask  对角线矩阵 计算因果mask
            attn_mask = TriangularCausalMask_local(src_embedded.size(1), device=src.device)
            tensor_len = src_mask.sum(dim=1)
            length_mask = LengthMask_local(tensor_len, max_len=src_mask.size(1), device=src.device)
            enc_output_memory = self.encoder(src_embedded, attn_mask, length_mask=length_mask)
        else:
            enc_output_memory = self.encoder(src_embedded)
        return enc_output_memory

    # encoder层  src的输出作为decoder的输入
    def decode(self, tgt, memory=None, src_mask=None, is_training=True, state=None):
        tgt_embedded = self.common_embedding(tgt)
        if is_training:
            # 计算因果mask
            tgt_mask = TriangularCausalMask_local(tgt.size(1), device=tgt.device)
            tensor_len = src_mask.sum(dim=1)
            length_mask = LengthMask_local(tensor_len, max_len=src_mask.size(1), device=tgt.device)
            dec_output = self.decoder(tgt_embedded, memory, x_mask=tgt_mask, x_length_mask=length_mask)
            return dec_output, None
        else:
            tgt_embedded = tgt_embedded.squeeze(0)
            dec_output, state = self.decoder(tgt_embedded, memory, state=state)
            return dec_output, state

    # loss_mask 是padding mask，1024长度序列中，没有数据被padding的位置
    def forward(self, src, tgt, loss_mask=None):
        # 定义src_mask，即所有的词都是有效的，没有填充词
        encoder_memory = self.encode(src, loss_mask)

        # 预测生成时，返回的是最后一个时间步的结果，而不是loss
        y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, state = self.decode_and_output(tgt, encoder_memory, src_mask=loss_mask)

        # decoder 重构出来的结果，这个结果与真实目标target对比，产生loss
        if self.is_training:
            # 计算loss的逻辑
            # reshape (b, s, f) -> (b, f, s)
            y_tempo = y_tempo[:, ...].permute(0, 2, 1)
            y_chord = y_chord[:, ...].permute(0, 2, 1)
            y_type = y_type[:, ...].permute(0, 2, 1)
            y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
            y_pitch = y_pitch[:, ...].permute(0, 2, 1)
            y_duration = y_duration[:, ...].permute(0, 2, 1)
            y_velocity = y_velocity[:, ...].permute(0, 2, 1)
            y_emotion = y_emotion[:, ...].permute(0, 2, 1)
            # loss
            # 最后过一层softmax  因为是多个embedding在一起，用多个proj计算所有的loss  取平均
            loss_tempo = self.compute_loss(y_tempo, tgt[..., 0], loss_mask)
            loss_chord = self.compute_loss(y_chord, tgt[..., 1], loss_mask)
            loss_type = self.compute_loss(y_type, tgt[..., 3], loss_mask)
            loss_barbeat = self.compute_loss(y_barbeat, tgt[..., 2], loss_mask)
            loss_pitch = self.compute_loss(y_pitch, tgt[..., 4], loss_mask)
            loss_duration = self.compute_loss(y_duration, tgt[..., 5], loss_mask)
            loss_velocity = self.compute_loss(y_velocity, tgt[..., 6], loss_mask)
            loss_emotion = self.compute_loss(y_emotion, tgt[..., 7], loss_mask)

            # 返回各个Loss
            return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_emotion
        else:
            # 生成 ，返回的是比例分布
            y_tempo = y_tempo[:, -1, :].unsqueeze(0)
            y_chord = y_chord[:, -1, :].unsqueeze(0)
            y_type = y_type[:, -1, :].unsqueeze(0)
            y_barbeat = y_barbeat[:, -1, :].unsqueeze(0)
            y_pitch = y_pitch[:, -1, :].unsqueeze(0)
            y_duration = y_duration[:, -1, :].unsqueeze(0)
            y_velocity = y_velocity[:, -1, :].unsqueeze(0)
            y_emotion = y_emotion[:, -1, :].unsqueeze(0)
            return nn.Softmax(dim=-1)(y_tempo), nn.Softmax(dim=-1)(y_chord), nn.Softmax(dim=-1)(y_type), nn.Softmax(dim=-1)(y_barbeat), nn.Softmax(dim=-1)(y_pitch), nn.Softmax(dim=-1)(y_duration), nn.Softmax(dim=-1)(y_velocity), nn.Softmax(dim=-1)(y_emotion), state

    # 这里确实需要一个type加强一下，要不然生成的乱七八糟啊。
    def decode_and_output(self, tgt, memory, src_mask=None, state=None):
        if self.is_training:
            decoder_outputs, _ = self.decode(tgt, memory=memory, src_mask=src_mask, is_training=True)
        else:
            # tgt = tgt.squeeze(0)
            decoder_outputs, state = self.decode(tgt, memory=memory, src_mask=src_mask, is_training=False, state=state)
        '''
        for training
        '''
        # tf_skip_emption = self.word_emb_emotion(y[..., 7])
        tf_skip_type = self.word_emb_type(tgt[..., 3])
        # project other  沿着最后一维堆叠  增强一下type
        if not self.is_training:
            decoder_outputs = decoder_outputs.unsqueeze(0)
            encoder_cat_type = torch.cat([decoder_outputs, tf_skip_type], dim=-1)
        else:
            encoder_cat_type = torch.cat([decoder_outputs, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(encoder_cat_type)

        # individual output  做了一个全连接
        y_type = self.proj_type(decoder_outputs)
        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)
        return y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, state

    def compute_loss(self, predict, target, loss_mask=None):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def common_embedding(self, x, emotion=None):
        # Embedding
        emb_tempo = self.word_emb_tempo(x[..., 0])
        emb_chord = self.word_emb_chord(x[..., 1])
        emb_barbeat = self.word_emb_barbeat(x[..., 2])
        emb_type = self.word_emb_type(x[..., 3])
        emb_pitch = self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])
        emb_emotion = self.word_emb_emotion(x[..., 7])

        # emb_emotion = self.word_emb_emotion(emotion)

        # 把特征concat  shape torch.Size([32, 1024, 1376])
        embs = torch.cat(
            [emb_tempo, emb_chord, emb_barbeat, emb_type, emb_pitch, emb_duration, emb_velocity, emb_emotion], dim=-1)
        # embs_ shape torch.Size([32, 1024, 512])
        embs_ = self.linear(embs)
        # encoding_input shape torch.Size([32, 1024, 512])
        encoding_input = self.posAtten(embs_)
        return encoding_input

    def get_encoder(self, TYPE):

        if self.is_training:
            self.encoder = TransformerEncoderBuilder_local.from_kwargs(
                n_layers=self.e_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",  # 因果mask
            ).get()
            self.decoder = TransformerDecoderBuilder_local.from_kwargs(
                n_layers=self.d_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=self.d_ff,
                activation='gelu',
                dropout=0.1,
            ).get()
        else:
            self.encoder = TransformerEncoderBuilder_local.from_kwargs(
                n_layers=self.e_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
            ).get()
            self.decoder = RecurrentDecoderBuilder_local.from_kwargs(
                n_layers=self.d_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=self.d_ff,
                activation='gelu',
                dropout=0.1,
            ).get()

    def generate_from_scratch(self, dictionary, emotion_tag, key_tag=None, n_token=8, display=True):
        event2word, word2event = dictionary

        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        generated_key = None

        target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]

        init = np.array([
            target_emotion,  # emotion
            [0, 0, 1, 2, 0, 0, 0, 0]  # bar
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            # encoder的结果
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')

            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                # 采样类型

            inp = init_t.unsqueeze(0)
            memory = self.encode(inp)

            print('------ generate ------')
            state = None
            while (True):
                # sample others  采样其他的，音高，音长，力度等
                next_arr, state = self.froward_output_sampling(input_, memory, state=state)
                if next_arr is None:
                    return None, None

                final_res.append(next_arr[None, ...])

                if display:
                    print('bar:', cnt_bar, end='  ==')
                    print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_ = input_.unsqueeze(0).unsqueeze(0)

                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)

        return final_res, generated_key


    def froward_output_sampling(self, input_, memory, is_training=False, state=None):
        y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, state = self.decode_and_output(input_, memory, state=state)

        cur_word_type = utils.sampling(y_type, p=0.99, is_training=is_training)
        # sampling gen_cond
        cur_word_tempo = utils.sampling(y_tempo, t=1.2, p=0.9, is_training=is_training)
        cur_word_barbeat = utils.sampling(y_barbeat, t=1.2, is_training=is_training)
        cur_word_chord = utils.sampling(y_chord, p=0.99, is_training=is_training)
        cur_word_pitch = utils.sampling(y_pitch, p=0.9, is_training=is_training)
        cur_word_duration = utils.sampling(y_duration, t=2, p=0.9, is_training=is_training)
        cur_word_velocity = utils.sampling(y_velocity, t=5, is_training=is_training)

        curs = [
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity
        ]

        if None in curs:
            return None, None

        if is_training:
            cur_word_emotion = torch.from_numpy(np.array([0])).long().cuda().squeeze(0)
            # collect
            next_arr = torch.tensor([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
            ])

        else:
            cur_word_emotion = 0

            # collect
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

        return next_arr, state


class Generator(nn.Module):
    """
    Decoder的输出会送到Generator中做最后的预测。
    """
    def __init__(self):
        """
        d_model: dimension of model. 这个值其实就是word embedding的维度。
                 例如，你把一个词编码成512维的向量，那么d_model就是512
        vocab: 词典的大小。
        """
        super(Generator, self).__init__()

    def forward(self, x):
        return F.softmax(self.proj(x), dim=-1)